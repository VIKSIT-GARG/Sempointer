"""C13/20 Fig C3 L_active analytic illustration (CPU-only, pure math, no data files)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

B = 512
M = 1
B_RES = 512
KS = (4, 8)

x = np.logspace(np.log10(512), np.log10(5e7), 300)  # history tokens N*B
N = x / B
y_full = x
curves = {k: N * k + M * B_RES for k in KS}

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.loglog(x, y_full, label="full: $NB$", linewidth=2)
for k in KS:
    ax.loglog(x, curves[k], label=f"pointer $Nk+mB_{{res}}$: k={k}, m=1", linewidth=2)
ax.set_xlabel("history tokens $NB$ (log scale)")
ax.set_ylabel("active tokens $L_{active}$")
ax.set_title("Active tokens vs history ($B=512$)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

# Annotate measured-regime coefficient cut (~13k history -> 10-15x)
x_ref = 13000.0
n_ref = x_ref / B
for k in KS:
    r = x_ref / (n_ref * k + M * B_RES)
    ax.annotate(f"k={k}: {r:.1f}x fewer", xy=(x_ref, n_ref * k + M * B_RES),
                xytext=(8, -14 if k == 4 else 12), textcoords="offset points",
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.98, 0.08, "10-15x coefficient cut (measured regime);\nattention still $O(L_{active}^2)$: degree unchanged",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

out = Path(__file__).resolve().parent.parent / "figures" / "C3_lactive.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print("Figure C3 (analytic illustration): active tokens vs history tokens for full (NB) vs pointer (Nk+mB_res, k=4,8; m=1, B=512) on log history axis, showing a 10-15x coefficient cut with quadratic degree unchanged.")
