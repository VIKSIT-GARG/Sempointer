"""Generate vector figures from the same completed runs as the SemPointer tables."""

from __future__ import annotations

from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_results import BENCHES, OUT, load, paired, rows_for
from analysis.stats import bootstrap_mean_ci

BLUE = "#1765a5"
ORANGE = "#c65d22"
GRAY = "#65727d"
LABELS = {"ruler": "RULER", "longbench": "LongBench", "locomo": "LoCoMo"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


def frontier(data: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 6.3), constrained_layout=True)
    for ax, (bench, run) in zip(axes, data.items()):
        systems = [("sempointer_v2", "SemPointer", BLUE, "o"),
                   ("bm25_m4", "BM25-4", ORANGE, "s")]
        if bench == "ruler":
            systems.append(("bm25_m2", "BM25-2", GRAY, "^"))
        for sys, label, color, marker in systems:
            rows = rows_for(run, sys)
            score, lo, hi = bootstrap_mean_ci([r["score"] for r in rows], seed=42)
            tokens = mean(r["active_tokens"] for r in rows)
            ax.errorbar(tokens, score * 100,
                        yerr=[[100 * (score - lo)], [100 * (hi - score)]],
                        fmt=marker, markersize=6, capsize=3, color=color, label=label)
        ax.set_title(f"{LABELS[bench]} ({BENCHES[bench]} examples)", loc="left", fontweight="bold")
        ax.set_ylabel("Score (%)")
        ax.grid(axis="both", alpha=.18)
        ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=7)
        ax.margins(x=.16, y=.34)
    axes[-1].set_xlabel("Mean active prompt tokens")
    fig.savefig(OUT / "frontier.pdf")
    plt.close(fig)


def paired_effects(data: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 5.1), constrained_layout=True)
    positions = np.arange(len(data))[::-1]
    specs = (
        ("score", 100, "SemPointer minus BM25-4 score (points)", BLUE, False),
        ("active_tokens", 1, "Tokens saved by SemPointer", ORANGE, True),
        ("latency_ms", 1, "Answer milliseconds saved by SemPointer", GRAY, True),
    )
    for ax, (field, scale, title, color, savings) in zip(axes, specs):
        for pos, (bench, run) in zip(positions, data.items()):
            _, point, lo, hi, _ = paired(run, "sempointer_v2", run, "bm25_m4", field, scale)
            if savings:
                point, lo, hi = -point, -hi, -lo
            ax.errorbar(point, pos, xerr=[[point - lo], [hi - point]], fmt="o",
                        color=color, capsize=3, markersize=5)
        ax.axvline(0, color="#333333", linewidth=.7)
        ax.set_yticks(positions, [LABELS[b] for b in data])
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=.18)
        ax.margins(x=.16, y=.35)
    fig.savefig(OUT / "paired_effects.pdf")
    plt.close(fig)


def routing(data: dict) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.0, 4.4), constrained_layout=True)
    benches = list(data)
    y = np.arange(len(benches))[::-1]
    two_share, two_tok, four_tok = [], [], []
    for bench in benches:
        rows = rows_for(data[bench], "sempointer_v2")
        two = [r for r in rows if r["router_route"] == "lexical"]
        four = [r for r in rows if r["router_route"] == "expanded_lexical"]
        if len(two) + len(four) != len(rows):
            raise ValueError(f"unexpected route in {bench}")
        two_share.append(len(two) / len(rows) * 100)
        two_tok.append(mean(r["active_tokens"] for r in two))
        four_tok.append(mean(r["active_tokens"] for r in four))
    top.barh(y, two_share, color=BLUE, label="Two-block route")
    top.barh(y, [100 - x for x in two_share], left=two_share, color=ORANGE,
             label="Expanded route")
    top.set_yticks(y, [LABELS[b] for b in benches])
    top.set_xlim(0, 100)
    top.set_xlabel("Share of questions (%)")
    top.legend(frameon=False, ncol=2, fontsize=7, loc="lower center",
               bbox_to_anchor=(.5, 1.01))
    for pos, share in zip(y, two_share):
        top.text(share / 2, pos, f"{share:.1f}%", ha="center", va="center",
                 color="white", fontsize=7, fontweight="bold")
    offset = .18
    bottom.barh(y + offset, two_tok, height=.34, color=BLUE, label="Two-block route")
    bottom.barh(y - offset, four_tok, height=.34, color=ORANGE, label="Expanded route")
    bottom.set_yticks(y, [LABELS[b] for b in benches])
    bottom.set_xlabel("Mean active prompt tokens within route")
    bottom.grid(axis="x", alpha=.18)
    bottom.set_axisbelow(True)
    fig.savefig(OUT / "routing.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = {b: load(f"v2_{b}") for b in BENCHES}
    frontier(data)
    paired_effects(data)
    routing(data)
    for name in ("frontier", "paired_effects", "routing"):
        print(OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
