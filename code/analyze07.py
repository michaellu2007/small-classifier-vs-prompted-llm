"""Aggregate 07 results: learning-curve table + plot + deployment verdict."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"


def load_rows():
    rows = [json.loads(l) for l in open(RES / "curves_classifiers.jsonl")]
    return rows


def pivot_f1(rows):
    tab = defaultdict(list)
    for r in rows:
        tab[(r["arm"], r["k"])].append(r["macro_f1"])
    return {key: (float(np.mean(v)), float(np.std(v))) for key, v in tab.items()}


def main():
    rows = load_rows()
    dec = json.load(open(RES / "decoder_summary.json"))
    tab = pivot_f1(rows)

    all_ks = sorted({k for (_, k) in tab})
    print(f"{'arm':16s} " + " ".join(f"k={k:<4d}" for k in all_ks))
    for arm in ("tfidf_logres", "minilm_logres", "setfit_minilm"):
        cells = [f"{tab[(arm, k)][0]:.3f}±{tab[(arm, k)][1]:.3f}" if (arm, k) in tab else "  —  " for k in all_ks]
        print(f"{arm:16s} " + " ".join(cells))
    print(f"{'decoder_8shot':16s} {dec['macro_f1']:.3f} (水平线, 零训练预算)")
    import os
    cpu = json.load(open(RES / "decoder_cpu.json")) if (RES / "decoder_cpu.json").exists() else None
    if cpu:
        print(f"\ndeoder CPU: Yahoo 域移 F1={cpu['shift_macro_f1']:.3f} | CPU 延迟 {cpu['ms_per_example_batch1_cpu']:.0f} ms/例(p90 {cpu['ms_p90_cpu']:.0f})")

    # SetFit summary row
    sf = [r for r in rows if r["arm"] == "setfit_minilm"]
    if sf:
        print("\nSetFit 臂(对比微调, CPU ~112s/次):")
        for k in sorted({r["k"] for r in sf}):
            rs = [r for r in sf if r["k"] == k]
            print(f"  k={k:3d}: F1={np.mean([r['macro_f1'] for r in rs]):.3f}±{np.std([r['macro_f1'] for r in rs]):.3f} "
                  f"ECE={np.mean([r['ece'] for r in rs]):.3f} 域移F1={np.mean([r['shift_macro_f1'] for r in rs]):.3f}")

    # Crossover: smallest k where each classifier arm's mean F1 beats decoder.
    for arm in ("tfidf_logres", "minilm_logres", "setfit_minilm"):
        for k in (8, 32, 128):
            if tab[(arm, k)][0] > dec["macro_f1"]:
                print(f">>> {arm} 在 k={k} 处反超 decoder ({tab[(arm,k)][0]:.3f} vs {dec['macro_f1']:.3f})")
                break

    print("\n校准 (ECE) 与延迟:")
    ece = defaultdict(list)
    lat = defaultdict(list)
    for r in rows:
        ece[(r["arm"], r["k"])].append(r["ece"])
        if "ms_per_example_batch1" in r:
            lat[r["arm"]].append(r["ms_per_example_batch1"])
    for arm in ("tfidf_logres", "minilm_logres", "setfit_minilm"):
        lat_s = f"{np.mean(lat[arm]):.1f} ms/例" if lat.get(arm) else "—(训练耗时见上)"
        print(f"{arm:16s} ECE@32={np.mean(ece[(arm,32)]):.3f}  延迟={lat_s}")
    print(f"{'decoder':16s} ECE={dec['ece']:.3f}  延迟={dec['ms_per_example_batch1']:.1f} ms/例(batch1)")

    # Domain shift at k=32
    shift = defaultdict(list)
    for r in rows:
        if "shift_macro_f1" in r and r["k"] == 32:
            shift[r["arm"]].append((r["macro_f1"], r["shift_macro_f1"]))
    if shift:
        print("\n域移 (k=32, AG News → Yahoo 对齐标签):")
        for arm, v in shift.items():
            in_dist = np.mean([x[0] for x in v])
            ood = np.mean([x[1] for x in v])
            print(f"{arm:16s} in={in_dist:.3f} ood={ood:.3f} Δ={ood-in_dist:+.3f}")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    styles = {"tfidf_logres": ("o-", "TF-IDF + LogRes"),
              "minilm_logres": ("s-", "MiniLM frozen + LogRes"),
              "setfit_minilm": ("^-", "SetFit (contrastive FT)")}
    for arm, (style, label) in styles.items():
        ks = sorted({k for (a, k) in tab if a == arm})
        if not ks:
            continue
        means = [tab[(arm, k)][0] for k in ks]
        stds = [tab[(arm, k)][1] for k in ks]
        ax.errorbar(ks, means, yerr=stds, fmt=style, capsize=3, label=label)
    ax.axhline(dec["macro_f1"], ls="--", color="crimson",
               label=f"Qwen3-0.6B 8-shot ({dec['macro_f1']:.2f})")
    ax.set_xscale("log", base=2)
    all_ks = sorted({k for (_, k) in tab})
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])
    ax.set_xlabel("labels per class")
    ax.set_ylabel("Macro-F1 (500-test, AG News)")
    ax.set_title("Small classifiers vs prompted 0.6B decoder")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "learning_curves.png", dpi=150)
    print(f"\n图已存: {RES/'learning_curves.png'}")


if __name__ == "__main__":
    main()
