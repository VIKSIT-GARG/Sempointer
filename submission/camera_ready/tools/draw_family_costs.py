"""Plot paired cost reductions from retained generations, without model calls.

Pass --database for the retained 4K results.sqlite. The archived CSV also
permits figure regeneration without the experiment database.
"""
import argparse
import csv
import json
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = [("ruler", "RULER"), ("longbench", "LongBench"), ("locomo", "LoCoMo"),
          ("longmem", "LongMemEval"), ("babilong", "BABILong"),
          ("longbench_v2", "LongBench v2"), ("infinite", "InfiniteBench"),
          ("loogle", "LooGLE"), ("leval", "L-Eval"), ("scrolls", "SCROLLS")]
parser = argparse.ArgumentParser()
parser.add_argument("--database", type=Path)
args = parser.parse_args()
csv_path = ROOT / "family_costs.csv"
if args.database:
    with sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True) as db:
        rows = {(r[0],r[1]): json.loads(r[2]) for r in db.execute(
            "select example_id,system,payload from latest where status='scored'")}
    output = []
    for family, label in LABELS:
        pairs = [(v, rows[(k[0],"bm25_budget")]) for k,v in rows.items()
                 if k[1]=="bounded_pointer" and v["family"]==family]
        assert len(pairs)==100, (family, len(pairs))
        values = {"family": label, "pairs": len(pairs)}
        for field, key in [("active_tokens","token_saving_percent"),
                           ("generation_ms","time_saving_percent")]:
            values[key] = 100*(1-sum(a[field] for a,b in pairs)/sum(b[field] for a,b in pairs))
        output.append(values)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
with csv_path.open() as f:
    rows = list(csv.DictReader(f))
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9.5,
                     "pdf.fonttype":42, "ps.fonttype":42,
                     "axes.spines.top":False, "axes.spines.right":False})
fig, axes = plt.subplots(1,2, figsize=(372/72, 248/72), sharey=True)
fig.subplots_adjust(left=.225, right=.98, top=.88, bottom=.16, wspace=.22)
y = np.arange(len(rows))
for ax, field, title, color in zip(axes,
        ["token_saving_percent","time_saving_percent"],
        ["Input tokens", "Generation time"], ["#344d63", "#8b9ca8"]):
    values = [float(r[field]) for r in rows]
    ax.barh(y, values, height=.62, color=color)
    ax.set_xlim(0, 80)
    ax.set_xticks([0,25,50,75])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.grid(axis="x", color=".88", linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    for position,value in zip(y,values):
        ax.text(value+1.8, position, f"{value:.1f}", va="center", fontsize=9)
axes[0].set_yticks(y, [r["family"] for r in rows], fontsize=9.5)
axes[0].invert_yaxis()
fig.supxlabel("Reduction versus budget-filling BM25 (%)", fontsize=9.5, y=.01)
fig.savefig(ROOT / "family_costs.pdf", metadata={"Title":"SemPointer cost reductions at 4K"})
plt.close(fig)
