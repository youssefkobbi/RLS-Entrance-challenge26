"""Aggregate results over seeds: mean +- std tables and the E1 comparison plot.

Usage (after training seeds 0 1 2 of both configs):
    python experiments/aggregate.py --names baseline blind --split test --out experiments/E1_blind

Reads runs/<name>_seed<k>/results_<split>.json for every seed k it finds and writes
  <out>/summary_<split>.md   markdown table (paste it into README.md / the report)
  <out>/summary_<split>.json raw aggregated numbers
  <out>/attribute_accuracy_<split>.png   per-attribute bars, all names side by side
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

METRICS = [
    ("exact match", lambda r: r["exact_match"]),
    ("size acc", lambda r: r["attribute_acc"]["size"]),
    ("color acc", lambda r: r["attribute_acc"]["color"]),
    ("shape acc", lambda r: r["attribute_acc"]["shape"]),
    ("relation acc", lambda r: r["attribute_acc"]["relation"]),
    ("#objects acc", lambda r: r["attribute_acc"]["n_objects"]),
    ("letter acc (TF, all)", lambda r: r["letter_acc_teacher_forced"]),
    ("letter acc (TF, first of part)", lambda r: r["letter_acc_first_of_part"]),
    ("letter acc (TF, inside part)", lambda r: r["letter_acc_inside_part"]),
]


def load(runs_dir, name, split):
    files = sorted(glob.glob(f"{runs_dir}/{name}_seed*/results_{split}.json"))
    return [json.load(open(f)) for f in files]


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--names", nargs="+", default=["baseline", "blind"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="experiments/E1_blind")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.names:
        results = load(args.runs_dir, name, args.split)
        if not results:
            print(f"no results for {name!r} in {args.runs_dir}/")
            continue
        summary[name] = {"n_seeds": len(results)}
        for label, fn in METRICS:
            m, s = mean_std([fn(r) for r in results])
            summary[name][label] = {"mean": m, "std": s}

    names = list(summary)
    lines = [f"Split: `{args.split}`, mean +- std over n seeds", "",
             "| metric | " + " | ".join(f"{n} (n={summary[n]['n_seeds']})" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for label, _ in METRICS:
        cells = [f"{summary[n][label]['mean']:.3f} +- {summary[n][label]['std']:.3f}" for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    (out / f"summary_{args.split}.md").write_text(table + "\n")
    (out / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m[0] for m in METRICS]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        xs = [j + i * width for j in range(len(labels))]
        ax.bar(xs, [summary[n][l]["mean"] for l in labels], width, yerr=[summary[n][l]["std"] for l in labels],
               capsize=2, label=n)
    ax.set_xticks([j + width * (len(names) - 1) / 2 for j in range(len(labels))])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set(ylabel="accuracy", ylim=(0, 1.05), title=f"Per-metric accuracy on {args.split} (mean +- std over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"attribute_accuracy_{args.split}.png", dpi=140)
    print("wrote", out / f"summary_{args.split}.md", "and the plot")


if __name__ == "__main__":
    main()
