"""Decoder arm on CPU: Yahoo-shift evaluation (first 100) + batch-1 latency on
CPU — the deployment-relevant latency number for the 07 question. Reuses the
cached-prediction protocol; writes to results/decoder_cpu_*.json.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arm_decoder import build_prompt

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
RES = Path("../results")


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    shots = [json.loads(l) for l in open("../data/agnews_shots.jsonl")]
    label_variants = [[" World", "World"], [" Sports", "Sports"],
                      [" Business", "Business"], [" Sci", "Sci-Tech", "Sci"]]
    label_ids = [[tok.encode(t, add_special_tokens=False)[0] for t in vs] for vs in label_variants]

    # --- Yahoo shift (first 100) ---
    rows = [json.loads(l) for l in open("../data/yahoo_shift.jsonl")][:100]
    preds, golds = [], []
    bs = 16
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        prompts = [build_prompt(tok, shots, r["text"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id)
        first = gen.scores[0].float()
        restricted = torch.stack([first[:, ids].max(dim=-1).values for ids in label_ids], dim=-1)
        preds.extend(restricted.argmax(-1).tolist())
        golds.extend(r["label"] for r in batch)
        print(f"{min(i+bs, len(rows))}/{len(rows)}", flush=True)
    from sklearn.metrics import f1_score
    shift_f1 = f1_score(golds, preds, average="macro")
    shift_acc = float(sum(p == g for p, g in zip(preds, golds)) / len(golds))

    # --- batch-1 latency on CPU (30 examples from the main test set) ---
    test_rows = [json.loads(l) for l in open("../data/agnews_test.jsonl")][:30]
    lats = []
    for r in test_rows:
        prompt = build_prompt(tok, shots, r["text"])
        enc = tok([prompt], return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=4, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        lats.append((time.perf_counter() - t0) * 1000)

    out = {
        "arm": "decoder_qwen06b_8shot_cpu",
        "shift_macro_f1": shift_f1, "shift_acc": shift_acc, "n_shift": len(rows),
        "ms_per_example_batch1_cpu": sum(lats) / len(lats),
        "ms_p90_cpu": sorted(lats)[int(0.9 * len(lats))],
    }
    print(json.dumps(out, indent=2))
    (RES / "decoder_cpu.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
