"""Decoder arm: prompted Qwen3-0.6B, fixed 8-shot prompt, cached predictions.

One forward pass per example: greedy-generate up to 4 tokens while recording
first-token logits; class confidence = softmax over the 4 label first-tokens.
Predictions are cached to JSONL so re-analysis never re-runs the model.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
SHOT_TMPL = "Text: {text}\nClass: {cls}\n\n"
Q_TMPL = "Text: {text}\nClass:"


def build_prompt(tokenizer, shots, doc):
    user = (
        "Classify each news text into exactly one of: "
        + ", ".join(CLASS_NAMES)
        + ".\n\n"
        + "".join(SHOT_TMPL.format(text=s["text"][:400], cls=CLASS_NAMES[s["label"]]) for s in shots)
        + Q_TMPL.format(text=doc[:400])
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--test-file", type=Path, default=Path("../data/agnews_test.jsonl"))
    ap.add_argument("--shots-file", type=Path, default=Path("../data/agnews_shots.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("../results/decoder_predictions.jsonl"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    shots = [json.loads(l) for l in open(args.shots_file)]
    rows = [json.loads(l) for l in open(args.test_file)]
    # Continuation after "Class:" usually starts with a space; take, per class,
    # the max logit over both spaced and unspaced first-token variants.
    label_variants = [[" World", "World"], [" Sports", "Sports"],
                      [" Business", "Business"], [" Sci", "Sci-Tech", "Sci"]]
    label_ids = [
        [tokenizer.encode(t, add_special_tokens=False)[0] for t in variants]
        for variants in label_variants
    ]

    done = set()
    if args.cache.exists():
        done = {json.loads(l)["idx"] for l in open(args.cache)}
    out_f = open(args.cache, "a")

    import time

    latencies = []
    for i in range(0, len(rows), args.batch_size):
        batch = [(j, r) for j, r in enumerate(rows[i : i + args.batch_size], start=i) if j not in done]
        if not batch:
            continue
        prompts = [build_prompt(tokenizer, shots, r["text"]) for _, r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        gen = model.generate(
            **enc, max_new_tokens=4, do_sample=False, output_scores=True,
            return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id,
        )
        latencies.append((time.perf_counter() - t0) * 1000 / len(batch))
        first_logits = gen.scores[0].float()
        restricted = torch.stack(
            [first_logits[:, ids].max(dim=-1).values for ids in label_ids], dim=-1
        )
        probs = torch.softmax(restricted, dim=-1)
        new_tokens = gen.sequences[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for (j, r), text, p in zip(batch, texts, probs):
            out_f.write(json.dumps({
                "idx": j, "label": r["label"], "gen": text.strip(),
                "pred": int(p.argmax()), "conf": float(p.max()),
            }) + "\n")
        out_f.flush()
        print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    out_f.close()

    # Score from cache
    preds = [None] * len(rows)
    for l in open(args.cache):
        d = json.loads(l)
        preds[d["idx"]] = d
    from sklearn.metrics import f1_score
    import numpy as np

    y = [d["label"] for d in preds]
    yhat = [d["pred"] for d in preds]
    conf = np.array([d["conf"] for d in preds])
    correct = np.array([int(a == b) for a, b in zip(y, yhat)])
    ece = 0.0
    bins = np.minimum((conf * 10).astype(int), 9)
    for b in range(10):
        m = bins == b
        if m.sum():
            ece += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    summary = {
        "arm": "decoder_qwen06b_8shot",
        "macro_f1": f1_score(y, yhat, average="macro"),
        "acc": float(np.mean(correct)),
        "ece": float(ece),
        "ms_per_example_batch1": float(np.mean(latencies)) if latencies else None,
        "n": len(preds),
    }
    print(json.dumps(summary))
    (args.cache.parent / "decoder_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
