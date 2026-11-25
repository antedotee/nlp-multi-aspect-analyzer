"""Build the report figures from the benchmark results and verify that nothing inside them overlaps.

Run from the repository root:  python report/make_figures.py
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = Path("artifacts/official_run/results_runs.csv")
ORDER = ["BERT-base", "ModernBERT-base", "RoBERTa-base", "DeBERTa-v3-base", "RoBERTa-base + R-Drop"]
SHORT = ["BERT", "Modern-\nBERT", "RoBERTa", "DeBERTa-\nv3", "RoBERTa\n+R-Drop"]
COL_W = 3.45  # IEEE single-column width in inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def check_no_overlap(fig, name):
    """Fail if tick labels, axis labels, legend or annotations overlap each other or leave the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if lab.get_visible() and lab.get_text():
                boxes.append((f"tick '{lab.get_text()}'", lab.get_window_extent(r)))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text():
                boxes.append((f"axis label '{lab.get_text()}'", lab.get_window_extent(r)))
        for t in ax.texts:
            boxes.append((f"text '{t.get_text()}'", t.get_window_extent(r)))
        if ax.get_legend():
            boxes.append(("legend", ax.get_legend().get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    problems = [f"{n} leaves the canvas" for n, b in boxes if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    problems += [f"{a} overlaps {b}" for (a, ba), (b, bb) in itertools.combinations(boxes, 2) if ba.overlaps(bb)]
    # legend must not cover any bar or line in its own axes
    for ax in fig.axes:
        leg = ax.get_legend()
        if not leg:
            continue
        lb = leg.get_window_extent(r)
        ab = ax.get_window_extent(r)
        if lb.x0 < ab.x0 - 0.5 or lb.x1 > ab.x1 + 0.5:
            problems.append("legend sticks out of its axes")
        for p in ax.patches:
            if p.get_height() > 0 and p.get_window_extent(r).overlaps(lb):
                problems.append("legend covers a bar")
                break
        for ln in ax.lines:
            xy = ax.transData.transform(np.column_stack(ln.get_data()))
            if any(lb.contains(x, y) for x, y in xy):
                problems.append(f"legend covers line '{ln.get_label()}'")
    if problems:
        sys.exit(f"{name}: " + "; ".join(sorted(set(problems))))
    print(f"{name}: OK ({len(boxes)} text elements, no overlaps)")


def bars(ax, groups, series, colors, hatches, labels, means, stds):
    x, w = np.arange(len(groups)), 0.8 / len(series)
    for k, s in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, means[s], w, yerr=stds[s], capsize=1.8, color=colors[k],
               hatch=hatches[k], edgecolor="black", linewidth=0.4, label=labels[k], error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)


def main():
    runs = pd.read_csv(RUNS)

    # Fig. 2: strict E2E-F1 per domain
    g = runs.groupby(["model", "domain"])["e2e_f1"].agg(["mean", "std"]) * 100
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    doms = ["laptops", "restaurants"]
    bars(ax, SHORT, doms, ["#4C72B0", "#DD8452"], ["", "///"], ["Laptop14", "Rest14"],
         {d: [g.loc[(m, d), "mean"] for m in ORDER] for d in doms}, {d: [g.loc[(m, d), "std"] for m in ORDER] for d in doms})
    ax.set_ylabel("Strict E2E-F1 (%)")
    ax.set_ylim(50, 90)
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout(pad=0.4)
    check_no_overlap(fig, "fig_results")
    fig.savefig(OUT / "fig_results.pdf")
    plt.close(fig)

    # Fig. 3: per-polarity F1 on the pooled test set
    pooled = runs[runs.domain == "pooled"]
    cls = ["f1_POS", "f1_NEG", "f1_NEU"]
    gm = pooled.groupby("model")[cls].mean() * 100
    gs = pooled.groupby("model")[cls].std() * 100
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    bars(ax, SHORT, cls, ["#55A868", "#C44E52", "#8C8C8C"], ["", "///", "..."], ["Positive", "Negative", "Neutral"],
         {c: [gm.loc[m, c] for m in ORDER] for c in cls}, {c: [gs.loc[m, c] for m in ORDER] for c in cls})
    ax.set_ylabel("F1 by polarity (%)")
    ax.set_ylim(30, 100)
    ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout(pad=0.4)
    check_no_overlap(fig, "fig_polarity")
    fig.savefig(OUT / "fig_polarity.pdf")
    plt.close(fig)

    # Fig. 4: grouping precision / recall / F1 versus the clustering threshold
    sweep = json.loads((OUT / "grouping_sweep.json").read_text())
    t = [r["t"] for r in sweep["sweep"]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for key, lab, sty, col in [("p", "Precision", "--", "#4C72B0"), ("r", "Recall", ":", "#DD8452"), ("f1", "F1", "-", "black")]:
        ax.plot(t, [r[key] for r in sweep["sweep"]], sty, color=col, linewidth=1.1, marker="o", markersize=2.2, label=lab)
    ax.plot([0.09, 0.46], [sweep["old"]["f1"]] * 2, color="#C44E52", linewidth=0.9, linestyle="-.", label="Original F1")
    ax.plot([0.25, 0.25], [0, 1.08], color="gray", linewidth=0.6, label="_chosen")  # stops below the legend
    ax.text(0.253, 0.06, r"chosen $\tau$=0.25", fontsize=6.5, color="dimgray", ha="left", va="bottom")
    ax.set_xlabel(r"Cosine-distance threshold $\tau$")
    ax.set_ylabel("Pairwise score")
    ax.set_xlim(0.09, 0.46)
    ax.set_xticks(np.round(np.arange(0.10, 0.451, 0.05), 2))
    ax.set_ylim(0, 1.32)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(linewidth=0.3, alpha=0.6)
    ax.legend(loc="upper center", frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    check_no_overlap(fig, "fig_grouping")
    fig.savefig(OUT / "fig_grouping.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
