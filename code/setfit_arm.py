"""SetFit-style contrastive arm: few-shot fine-tuning of the frozen encoder
itself (cosine-similarity loss on same-class/different-class pairs), then a
logistic head. This is the "true few-shot fine-tuning" control that the
frozen-encoder arm lacks. CPU-friendly (MiniLM, few hundred pairs, 1 epoch).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def subsample_labeled(pool, k, seed):
    rng = random.Random(1000 + seed)
    by_class = {c: [r for r in pool if r["label"] == c] for c in range(4)}
    out = []
    for c in range(4):
        out += rng.sample(by_class[c], k)
    return out


def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = len(conf)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum():
            e += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="8,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    args = ap.parse_args()

    pool = load_jsonl(args.data_dir / "agnews_train.jsonl")
    test = load_jsonl(args.data_dir / "agnews_test.jsonl")
    shift = load_jsonl(args.data_dir / "yahoo_shift.jsonl")
    test_docs = [r["text"] for r in test]
    test_labels = np.array([r["label"] for r in test])
    shift_docs = [r["text"] for r in shift]
    shift_labels = np.array([r["label"] for r in shift])

    out_path = Path("../results/curves_classifiers.jsonl")
    for k in [int(x) for x in args.ks.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            rng = random.Random(2000 + seed)
            sub = subsample_labeled(pool, k, seed)
            by_class = {c: [r for r in sub if r["label"] == c] for c in range(4)}

            examples = []
            for c in range(4):
                same = by_class[c]
                for i in range(len(same)):
                    for j in range(i + 1, len(same)):
                        examples.append(InputExample(texts=[same[i]["text"], same[j]["text"]], label=1.0))
            while True:
                a, b = rng.choice(sub), rng.choice(sub)
                if a["label"] != b["label"]:
                    examples.append(InputExample(texts=[a["text"], b["text"]], label=0.0))
                if sum(1 for e in examples if e.label == 0.0) >= len([e for e in examples if e.label == 1.0]):
                    break

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            train_dl = DataLoader(examples, shuffle=True, batch_size=32)
            loss = losses.CosineSimilarityLoss(model)
            t0 = time.perf_counter()
            model.fit(train_objectives=[(train_dl, loss)], epochs=1, warmup_steps=8, show_progress_bar=False)
            fit_s = time.perf_counter() - t0

            def embed(docs):
                return normalize(model.encode(docs, batch_size=256, show_progress_bar=False, convert_to_numpy=True))

            X = embed([r["text"] for r in sub])
            clf = LogisticRegression(C=10.0, max_iter=2000).fit(X, [r["label"] for r in sub])
            proba = clf.predict_proba(embed(test_docs))
            pred = proba.argmax(1)
            sp = clf.predict_proba(embed(shift_docs)).argmax(1)
            row = {
                "arm": "setfit_minilm", "k": k, "seed": seed,
                "macro_f1": f1_score(test_labels, pred, average="macro"),
                "acc": float((pred == test_labels).mean()),
                "ece": ece(proba.max(1), pred == test_labels),
                "shift_macro_f1": f1_score(shift_labels, sp, average="macro"),
                "finetune_seconds": round(fit_s, 1),
            }
            print(json.dumps(row), flush=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
