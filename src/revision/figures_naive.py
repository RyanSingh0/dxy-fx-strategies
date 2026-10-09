"""Figures 1 to 5 of the paper (the naive decomposition on the free-provider series).

They were first drawn in the original notebooks; this rebuilds them from the same grid
as naive_stats.py so every figure in the paper has a script behind it.
Usage: python3 revision/figures_naive.py <figure_dir>"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import paths
import naive_stats as N

FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(str(paths.RESULTS), "revision")
os.makedirs(FIG, exist_ok=True)
yh = N.load_free()
dates = yh.Date
R0, bnh, co, oc = N.grid(yh)
grids = {c: N.grid(yh, c)[0] for c in (1.0, 2.0)}
grids[0.0] = R0
eq = lambda r: 100 * np.cumprod(1 + np.asarray(r, float))
BH = eq(bnh)

# Figure 1: where the returns accrue
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(dates, eq(np.nan_to_num(co)), color="#7b2fbe", lw=1.2, label="Overnight (close→open) only")
ax.plot(dates, eq(np.nan_to_num(oc)), color="#0f9d76", lw=1.2, label="Daytime (open→close) only")
ax.plot(dates, BH, "k--", lw=1.2, label="Full close-to-close")
ax.axhline(100, color="k", lw=0.8)
ax.set(title="Where DXY returns actually accrue, 2010–2025", xlabel="Date", ylabel="Value of $100")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig0_decomposition.pdf")); plt.close(fig)

# Figure 2: the four headline strategies against buy and hold, log scale
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(dates, BH, "k--", lw=1.4, label="DXY Buy & Hold")
for k, c, lab in [(7, "#d62728", "S7  Short overnight / Long daytime"),
                  (3, "#1f77b4", "S3  Cash / Long daytime"),
                  (2, "#2ca02c", "S2  Short overnight / Cash"),
                  (1, "#7f7f7f", "S1  Long overnight / Cash")]:
    ax.plot(dates, eq(R0[k]), color=c, lw=1.3, label=lab)
ax.set_yscale("log")
ax.set(title="DXY: overnight/daytime decomposition strategies vs. buy & hold (zero cost)",
       xlabel="Date", ylabel="Portfolio value (start = $100, log scale)")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_dxy_equity.pdf")); plt.close(fig)

# Figure 3: all 24 strategies at three cost levels
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(1, 25); w = 0.27
for i, (c, col) in enumerate([(0.0, "#2b3a4a"), (1.0, "#ef7d1a"), (2.0, "#a6a6a6")]):
    ax.bar(x + (i - 1) * w, [100 * np.prod(1 + grids[c][k]) for k in x], w, color=col,
           label="%d bp per trade" % c)
ax.axhline(BH[-1], color="red", ls="--", lw=1.4, label="DXY buy & hold")
ax.axhline(100, color="k", lw=0.8)
ax.set_xticks(x)
ax.set(title="All 24 DXY strategies under three transaction-cost regimes", xlabel="Strategy number",
       ylabel="Final balance (start = $100)")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_dxy_costs.pdf")); plt.close(fig)

# Figure 4: drawdowns of Strategy 7 and buy and hold
dd = lambda v: 100 * (v / np.maximum.accumulate(v) - 1)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(dates, dd(eq(R0[7])), 0, color="#d62728", alpha=0.5, label="Strategy 7")
ax.fill_between(dates, dd(BH), 0, color="#7f7f7f", alpha=0.6, label="DXY buy & hold")
ax.set(title="Drawdown profile, zero cost", xlabel="Date", ylabel="Drawdown (%)")
ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_dxy_drawdown.pdf")); plt.close(fig)

# Figure 5: Strategy 7 terminal value as the cost per trade rises
cs = np.linspace(0, 2.5, 51)
s7 = lambda c: 100 * np.prod(1 + N.grid(yh, c)[0][7])
tv = np.array([s7(c) for c in cs])
cross = lambda lvl: brentq(lambda c: s7(c) - lvl, 0.0, 2.5, xtol=1e-4)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(cs, tv, color="#d62728", lw=1.5, label="Strategy 7 terminal value")
ax.axhline(BH[-1], color="navy", ls="--", lw=1.2, label="DXY buy & hold (%.2f)" % BH[-1])
ax.axhline(100, color="k", lw=1.0, label="Break-even (100)")
for lvl in (BH[-1], 100.0):
    c = cross(lvl)
    ax.axvline(c, color="k", ls=":", lw=0.8)
    ax.text(c, tv.max() * 0.9, "%.2f bp" % c, rotation=90, fontsize=7, ha="right")
ax.set(title="Strategy 7: decay of the edge with execution cost",
       xlabel="Transaction cost per trade (basis points)", ylabel="Final value of $100")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_s7_sweep.pdf")); plt.close(fig)
print("break-even vs buy and hold %.3f bp, vs $100 %.3f bp" % (cross(BH[-1]), cross(100.0)))
