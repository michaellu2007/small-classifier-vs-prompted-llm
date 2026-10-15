"""Learning-curve experiments for the classifier arms (TF-IDF, frozen encoder).

For each k in {8,32,128} labels/class and seed in {0,1,2}: subsample a labeled
set from the pool, fit the head, evaluate on the fixed 500-example test set.
Reports macro-F1, accuracy, ECE(10-bin), and batch-1 latency (ms/example).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


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


class TfidfArm:
    name = "tfidf_logres"

    def fit(self, docs, labels):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = self.vec.fit_transform(docs)
        self.clf = LogisticRegression(C=10.0, max_iter=2000)
        self.clf.fit(X, labels)

    def predict(self, docs):
        X = self.vec.transform(docs)
        proba = self.clf.predict_proba(X)
        return proba.argmax(1), proba.max(1), None

    def latency_one(self, doc):
        t0 = time.perf_counter()
        X = self.vec.transform([doc])
        self.clf.predict_proba(X)
        return (time.perf_counter() - t0) * 1000


class EncoderArm:
    name = "minilm_logres"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.enc = SentenceTransformer(model, device="cpu")

    def _embed(self, docs, batch_size=256):
        return normalize(self.encode_fn(docs, batch_size))

    def fit(self, docs, labels):
        self.encode_fn = lambda ds, bs: self.enc.encode(
            ds, batch_size=bs, show_progress_bar=False, convert_to_numpy=True
        )
        X = self._embed(docs)
        self.clf = LogisticRegression(C=10.0, max_iter=2000)
        self.clf.fit(X, labels)

    def predict(self, docs):
        X = self._embed(docs)
        t0 = time.perf_counter()
        proba = self.clf.predict_proba(X)
        head_ms = (time.perf_counter() - t0) * 1000
        return proba.argmax(1), proba.max(1), head_ms / len(docs)

    def latency_one(self, doc):
        t0 = time.perf_counter()
        X = self._embed([doc])
        self.clf.predict_proba(X)
        return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="tfidf,encoder")
    ap.add_argument("--ks", default="8,32,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../results/curves_classifiers.jsonl"))
    ap.add_argument("--shift-file", type=Path, default=Path("../data/yahoo_shift.jsonl"),
                    help="if it exists, also eval each (k=32) fit on the shift set")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pool = load_jsonl(args.data_dir / "agnews_train.jsonl")
    test = load_jsonl(args.data_dir / "agnews_test.jsonl")
    shift = load_jsonl(args.shift_file) if args.shift_file.exists() else []
    test_docs = [r["text"] for r in test]
    test_labels = np.array([r["label"] for r in test])
    shift_docs = [r["text"] for r in shift]
    shift_labels = np.array([r["label"] for r in shift])

    arm_classes = {"tfidf": TfidfArm, "encoder": EncoderArm}
    results = []
    for arm_name in args.arms.split(","):
        arm = arm_classes[arm_name]()
        for k in [int(x) for x in args.ks.split(",")]:
            for seed in [int(x) for x in args.seeds.split(",")]:
                sub = subsample_labeled(pool, k, seed)
                arm.fit([r["text"] for r in sub], [r["label"] for r in sub])
                pred, conf, _ = arm.predict(test_docs)
                lat = np.mean([arm.latency_one(d) for d in test_docs[:100]])
                row = {
                    "arm": arm.name, "k": k, "seed": seed,
                    "macro_f1": f1_score(test_labels, pred, average="macro"),
                    "acc": float((pred == test_labels).mean()),
                    "ece": ece(conf, pred == test_labels),
                    "ms_per_example_batch1": lat,
                }
                if shift:
                    spred, _, _ = arm.predict(shift_docs)
                    row["shift_macro_f1"] = f1_score(shift_labels, spred, average="macro")
                results.append(row)
                print(json.dumps(row), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
