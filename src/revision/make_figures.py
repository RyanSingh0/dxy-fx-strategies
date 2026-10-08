"""Figures added in the revision."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths
OUT = os.path.join(str(paths.RESULTS), "revision")
FIG = sys.argv[1] if len(sys.argv) > 1 else OUT
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})
S = json.load(open(os.path.join(OUT, "ceiling_summary.json")))
obs = S["observed_best"]

# ---------------- noise ceiling
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
mx = np.load(os.path.join(OUT, "maxdist_boot_L21.npy"))
ax[0].hist(mx, bins=50, color=C[0], alpha=0.85, edgecolor="white", linewidth=0.4)
for v, lab, ls in [(obs, "best on real data %.3f" % obs, "-"), (mx.mean(), "ceiling (mean) %.3f" % mx.mean(), "--"),
                   (np.quantile(mx, .95), "95th percentile %.3f" % np.quantile(mx, .95), ":")]:
    ax[0].axvline(v, color=INK if ls == "-" else MUTED, ls=ls, lw=1.3)
    ax[0].text(v + 0.012, ax[0].get_ylim()[1] * (0.97 if ls == "-" else 0.86 if ls == "--" else 0.75), lab, fontsize=7, color=INK,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax[0].set_xlabel("Best net Sharpe ratio of the 1,312-configuration search")
ax[0].set_ylabel("Replications")
ax[0].set_title("(a) Null distribution, stationary bootstrap, B = %d" % len(mx), fontsize=9, loc="left")
for i, (k, lab) in enumerate([("boot_L5", "bootstrap, mean block 5"), ("boot_L21", "bootstrap, mean block 21"),
                               ("boot_L63", "bootstrap, mean block 63"), ("boot_L126", "bootstrap, mean block 126"),
                               ("gauss_L21", "Gaussian, correlated")]):
    f = os.path.join(OUT, "maxdist_%s.npy" % k)
    if not os.path.exists(f): continue
    x = np.sort(np.load(f)); ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=C[i], lw=1.6, label="%s (B=%d)" % (lab, len(x)))
ax[1].axvline(obs, color=INK, lw=1.3); ax[1].text(obs - 0.015, 0.25, "real best", rotation=90, ha="right", fontsize=7)
ax[1].set_xlabel("Best net Sharpe ratio"); ax[1].set_ylabel("Cumulative share of replications")
ax[1].set_title("(b) Sensitivity to the resampling scheme", fontsize=9, loc="left")
ax[1].legend(fontsize=6.3, loc="upper left")
ax[1].set_xlim(0.4, 1.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figE_noise_ceiling.pdf")); plt.close(fig)

# ---------------- power of the screen
D = json.load(open(os.path.join(OUT, "screen_study.json")))["D"]
kinds = [("atlow", "Session low written into open"), ("outside", "Open outside range (71% below)"),
         ("down", "Open displaced down, within range"), ("stale", "Stale open = previous close"),
         ("noise", "Symmetric noise, 20% of range")]
rates = [0.005, 0.01, 0.02, 0.035, 0.05]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
for i, (k, lab) in enumerate(kinds):
    ax[0].plot([100 * r for r in rates], [D["%s_%g" % (k, r)]["rej5"] for r in rates], marker="o", ms=4, lw=1.6, color=C[i], label=lab)
    ax[1].plot([100 * r for r in rates], [D["%s_%g" % (k, r)]["on_sig"] for r in rates], marker="o", ms=4, lw=1.6, color=C[i], label=lab)
ax[0].axhline(0.05, color=MUTED, ls=":", lw=1)
ax[0].set_ylabel("Share of injections flagged (p < 0.05)"); ax[1].set_ylabel("Share with |t| > 1.96 on overnight leg")
for a in ax: a.set_xlabel("Share of sessions corrupted (%)"); a.set_ylim(-0.03, 1.03)
ax[0].set_title("(a) Power of the binomial screen", fontsize=9, loc="left")
ax[1].set_title("(b) Overnight leg becomes 'significant'", fontsize=9, loc="left")
ax[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figF_screen_power.pdf")); plt.close(fig)

# ---------------- decomposition
a = pd.read_csv(os.path.join(OUT, "decomp_path.csv"), index_col=0, parse_dates=True)
b = pd.read_csv(os.path.join(OUT, "decomp_path_exch.csv"), index_col=0, parse_dates=True)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
ax[0].plot(a.index, a["index"], color=C[0], lw=1.6, label="Free index, overnight leg")
ax[0].plot(a.index, a["constituents"], color=C[1], lw=1.6, label="Weighted free constituent spot bars")
ax[0].set_title("(a) Free provider", fontsize=9, loc="left")
ax[1].plot(b.index, b["index"], color=C[0], lw=1.6, label="Index futures (DX), overnight leg")
ax[1].plot(b.index, b["constituents"], color=C[1], lw=1.6, label="Weighted five currency futures")
ax[1].set_title("(b) Exchange-sourced futures", fontsize=9, loc="left")
ax[0].set_ylabel("Cumulative overnight log return")
for x in ax: x.legend(fontsize=6.5, loc="lower left"); x.axhline(0, color=MUTED, lw=0.8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figG_decomposition.pdf")); plt.close(fig)

# ---------------- yearly screen counts
F = json.load(open(os.path.join(OUT, "screen_study.json")))["F"]["free"]
yrs = sorted(int(y) for y in F)
fig, ax = plt.subplots(figsize=(7.2, 2.3))
w = 0.38
ax.bar([y - w / 2 for y in yrs], [F[str(y)][0] for y in yrs], width=w - 0.04, color=C[0], label="open at session low")
ax.bar([y + w / 2 for y in yrs], [F[str(y)][1] for y in yrs], width=w - 0.04, color=C[1], label="open at session high")
ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs], fontsize=7); ax.set_ylabel("Sessions")
ax.legend(fontsize=7); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figH_screen_by_year.pdf")); plt.close(fig)

# ---------------- carry rolling
r = pd.read_csv(os.path.join(OUT, "carry_rolling.csv"), index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.plot(r.index, r["spot6_long"], color=C[0], lw=1.6, label="Spot plus interbank differential, six currencies")
ax.plot(r.index, r["futures_roll_signal"], color=C[1], lw=1.6, label="Currency futures, roll-implied carry signal")
ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel("Rolling 3-year Sharpe ratio")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figI_carry_rolling.pdf")); plt.close(fig)
print("figures written to", FIG)
