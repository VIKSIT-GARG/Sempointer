"""Fig C2 K* crossover (C12/20): analytic closed-form only, CPU-only, no weights/data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sempointer.cost_model import SemPointerCostModel
except ImportError:  # fallback when run from repo root
    from experiments.sempointer.cost_model import SemPointerCostModel

# Params (closed-form only)
D, B, K, LQ = 4096, 512, 8, 64
GEN = "pooling"
CONFIGS = [(1, "M_kv"), (1, "M_text"), (4, "M_kv"), (4, "M_text")]
STYLES = {(1, "M_kv"): ("-", "C0"), (1, "M_text"): ("--", "C0"),
          (4, "M_kv"): ("-", "C1"), (4, "M_text"): ("--", "C1")}

N = np.unique(np.round(np.logspace(0, 5, 600)).astype(int))
N = N[(N >= 1) & (N <= 100000)]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
dom_union = np.zeros_like(N, dtype=bool)
for m, sub in CONFIGS:
    model = SemPointerCostModel(d=D, B=B, k=K, m=m, B_res=B, L_Q=LQ,
                                generator_class=GEN, substrate=sub)
    Ks = np.array([model.k_star(int(n)) for n in N], dtype=float)
    Dl = np.array([model.delta_c_step(int(n)) for n in N], dtype=float)
    dom_union |= (Dl <= 0)
    y = np.where(np.isinf(Ks), np.nan, Ks)
    ls, c = STYLES[(m, sub)]
    label = f"m={m}, {sub}" + (" ($C_{res}=0$)" if sub == "M_kv" else "")
    ax.plot(N, y, ls=ls, color=c, lw=1.8, label=label)

# Shade dominated region (Delta_C_step <= 0 for any curve)
if dom_union.any():
    idx = np.where(dom_union)[0]
    # group contiguous; here single block at small N
    ax.axvspan(float(N[idx[0]]), float(N[idx[-1]]) + 0.6, color="0.85", alpha=0.7,
               label=r"dominated ($\Delta C_{\rm step}\leq 0$)")

ax.axvline(2, color="k", lw=1.0, dashes=(3, 2), label="N=2 viability")
ax.set_xscale("log")
ax.set_xlim(1, 1e5)
ax.set_ylim(0, 20)
ax.set_xlabel("N (blocks, log scale)")
ax.set_ylabel(r"$K^*$ (queries to amortize, linear scale)")
ax.set_title(r"$K^*$ vs $N$ ($L_Q$=64, d=4096, B=512, k=8, pooling)")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "figures" / "C2_kstar.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("Caption: Analytic, not measured: closed-form K* vs N shows pooling indirection amortizes within few queries for N>=2 except the shaded small-N dominated region.")
print(f"Saved {out} ({out.stat().st_size} bytes)")
