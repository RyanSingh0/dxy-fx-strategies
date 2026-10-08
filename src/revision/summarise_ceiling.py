"""Summarise the noise-ceiling replications: mean, Monte Carlo standard error, quantiles with
order-statistic 95% confidence intervals, Monte Carlo p-value of the observed best Sharpe,
and a per-configuration comparison against the replicate distribution."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd
from scipy import stats as st
import paths, data as D
import fast_search as F
OUT = os.path.join(str(paths.RESULTS), "revision")
P = D.panel(); keys, M = F.evaluate_all(P); real = F.sharpes(M)
obs = float(np.nanmax(real))
def qci(x, q, a=0.05):
    x = np.sort(x); n = len(x)
    lo = int(st.binom.ppf(a / 2, n, q)); hi = int(st.binom.ppf(1 - a / 2, n, q))
    return float(np.quantile(x, q)), float(x[max(lo - 1, 0)]), float(x[min(hi, n - 1)])
out = {"observed_best": obs}
for fn in sorted(os.listdir(OUT)):
    if not (fn.startswith("ceiling_") and fn.endswith(".npy") and "_L" in fn): continue
    A = np.load(os.path.join(OUT, fn)); A = A[~np.isnan(A).all(axis=1)]
    mx = np.nanmax(A, axis=1); B = len(mx)
    r = dict(B=B, mean=float(mx.mean()), sd=float(mx.std(ddof=1)), mcse=float(mx.std(ddof=1) / np.sqrt(B)),
             first6=[float(v) for v in mx[:6]], first6_mean=float(mx[:6].mean()),
             p_mc=float((1 + np.sum(mx >= obs)) / (1 + B)),
             n_exceed_mean=int(np.sum(real > mx.mean())))
    for q in (0.5, 0.9, 0.95, 0.99):
        r["q%02d" % int(q * 100)] = qci(mx, q)
    # running mean, to show convergence
    r["running"] = [float(mx[:k].mean()) for k in (6, 25, 50, 100, 250, 500, 1000, 2000) if k <= B]
    # per-configuration: rank of each real Sharpe within its own replicate distribution
    pc = (1 + np.sum(A >= real[None, :], axis=0)) / (1 + B)
    r["min_config_p"] = float(np.nanmin(pc)); r["n_config_p05"] = int(np.sum(pc < 0.05)); r["n_config_p01"] = int(np.sum(pc < 0.01))
    # expected number of configs with p<0.05 under the null is 0.05 * 1312
    r["exceed_q95"] = int(np.sum(real > r["q95"][0]))
    # 99th / 95th percentile of the replicate's whole cross-section (to compare with the real 99th pct)
    r["xs_q99_mean"] = float(np.nanmean(np.nanquantile(A, 0.99, axis=1)))
    r["xs_median_mean"] = float(np.nanmean(np.nanmedian(A, axis=1)))
    out[fn[8:-4]] = r
    np.save(os.path.join(OUT, "maxdist_" + fn[8:-4] + ".npy"), mx)
    print("%-12s B=%4d mean %.3f (MC s.e. %.4f, sd %.3f) | q50 %.3f q90 %.3f [%.3f, %.3f] q95 %.3f [%.3f, %.3f] q99 %.3f | P(max>=%.3f) = %.3f | first-6 mean %.3f"
          % (fn[8:-4], B, r["mean"], r["mcse"], r["sd"], r["q50"][0], *r["q90"], *r["q95"], r["q99"][0], obs, r["p_mc"], r["first6_mean"]))
    print("             running mean", np.round(r["running"], 3), "| per-config p<0.05: %d (expected %.0f), p<0.01: %d, min p %.4f"
          % (r["n_config_p05"], 0.05 * 1312, r["n_config_p01"], r["min_config_p"]))
print("real cross-section: 99th pct %.3f, median %.3f" % (np.nanquantile(real, .99), np.nanmedian(real)))
out["real_q99"] = float(np.nanquantile(real, .99)); out["real_median"] = float(np.nanmedian(real))
json.dump(out, open(os.path.join(OUT, "ceiling_summary.json"), "w"), indent=1)
