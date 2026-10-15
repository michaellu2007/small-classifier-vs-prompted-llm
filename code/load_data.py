"""Download AG News + Yahoo! Answers Topics and build fixed pools.

Outputs (data/):
  agnews_train.jsonl   deduped train pool (8 few-shot rows removed)
  agnews_shots.jsonl   8 fixed few-shot examples (2/class) for the decoder arm
  agnews_test.jsonl    500-example stratified test subset (125/class)
  yahoo_shift.jsonl    500-example stratified Yahoo subset mapped to AG classes

Label mapping Yahoo -> AG News: Sports(5)->Sports(1),
Business & Finance(6)->Business(2), Science & Mathematics(1) and
Computers & Internet(4)->Sci/Tech(3), Politics & Government(9)->World(0).
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
YAHOO_TO_AG = {5: 1, 6: 2, 1: 3, 4: 3, 9: 0}


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} rows")


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(7)

    train = load_dataset("fancyzhx/ag_news", split="train")
    test = load_dataset("fancyzhx/ag_news", split="test")
    train = [{"text": r["text"], "label": r["label"]} for r in train]
    test = [{"text": r["text"], "label": r["label"]} for r in test]

    # Dedupe by text (AG News train contains duplicates), keep first occurrence.
    seen, pool = set(), []
    for r in train:
        if r["text"] not in seen:
            seen.add(r["text"])
            pool.append(r)

    by_class = {c: [r for r in pool if r["label"] == c] for c in range(4)}
    shots, shot_ids = [], set()
    for c in range(4):
        for r in rng.sample(by_class[c], 2):
            shots.append(r)
            shot_ids.add(id(r))
    train_pool = [r for r in pool if id(r) not in shot_ids]
    dump(OUT / "agnews_shots.jsonl", shots)
    dump(OUT / "agnews_train.jsonl", train_pool)

    test_by_class = {c: [r for r in test if r["label"] == c] for c in range(4)}
    test_subset = []
    for c in range(4):
        test_subset += rng.sample(test_by_class[c], 125)
    rng.shuffle(test_subset)
    dump(OUT / "agnews_test.jsonl", test_subset)

    yahoo = load_dataset("community-datasets/yahoo_answers_topics", split="test")
    shift_by_class = {c: [] for c in range(4)}
    for r in yahoo:
        if r["topic"] not in YAHOO_TO_AG:
            continue
        text = " ".join(filter(None, [r["question_title"], r["question_content"], r["best_answer"]])).strip()
        if text:
            shift_by_class[YAHOO_TO_AG[r["topic"]]].append(
                {"text": text, "label": YAHOO_TO_AG[r["topic"]]}
            )
    shift = []
    for c in range(4):
        shift += rng.sample(shift_by_class[c], 125)
    rng.shuffle(shift)
    dump(OUT / "yahoo_shift.jsonl", shift)


if __name__ == "__main__":
    main()
