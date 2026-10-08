"""Revision: (a) nonlinear and conditional predictability checks, (b) OHLC-based
effective-spread estimates for the six futures.

(a) BDS test (Brock, Dechert, Scheinkman and LeBaron 1996) on raw and on EWMA-volatility
    standardised returns; threshold / state-dependent autoregressions; out-of-sample
    directional accuracy of a nonparametric (k-nearest-neighbour) sign forecast.
(b) Roll (1984), Corwin and Schultz (2012) and Abdi and Ranaldo (2017) estimators from
    daily bars, monthly averages with negative estimates set to zero."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd, json
import statsmodels.api as sm
from statsmodels.tsa.stattools import bds
from scipy import stats as st
import paths, data as D

OUT = os.path.join(str(paths.RESULTS), "revision"); res = {}
P = D.panel(); ret = P["ret"].copy()
dxr = P["dx_ret"]; ret["DXY"] = dxr
ret = ret.iloc[1:]
print("(a) NONLINEAR AND CONDITIONAL PREDICTABILITY")
print("%-8s %9s %9s | %9s %9s | %8s %8s %8s | %7s %7s" % ("series", "BDS2 raw", "p", "BDS2 std", "p", "b_lag", "b_big", "b_hivol", "Wald p", "kNN hit"))
for c in ret.columns:
    r = ret[c].values
    lam = 0.94; v = np.empty_like(r); v[0] = r[:250].var()
    for t in range(1, len(r)): v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    z = r / np.sqrt(v)
    b_raw = bds(r, max_dim=3, epsilon=None); b_std = bds(z[250:], max_dim=3)
    # state-dependent AR(1): lag, lag x 1{|lag| > 90th pct}, lag x 1{EWMA vol above median}
    y = r[1:]; x = r[:-1]; big = (np.abs(x) > np.quantile(np.abs(x), .9)).astype(float)
    hv = (v[:-1] > np.median(v)).astype(float)
    X = sm.add_constant(np.column_stack([x, x * big, x * hv]))
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    w = m.wald_test(np.eye(4)[1:], scalar=True)
    # kNN sign forecast, expanding window, features = last 5 standardised returns, k = 50
    F = np.column_stack([z[i:len(z) - 5 + i] for i in range(5)]); tgt = np.sign(r[5:])
    hits = []; start = 1500
    for t in range(start, len(F), 5):
        d = np.sum((F[:t - 1] - F[t]) ** 2, axis=1); nn = np.argsort(d)[:50]
        pred = np.sign(np.sum(tgt[nn])) or 1.0
        hits.append(pred == tgt[t])
    hit = float(np.mean(hits)); pb = float(st.binomtest(int(np.sum(hits)), len(hits), 0.5).pvalue)
    res[c] = dict(bds_raw=float(b_raw[0][0]), bds_raw_p=float(b_raw[1][0]), bds_std=float(b_std[0][0]), bds_std_p=float(b_std[1][0]),
                  b=m.params[1:].tolist(), t=m.tvalues[1:].tolist(), wald_p=float(w.pvalue), knn_hit=hit, knn_p=pb, knn_n=len(hits))
    print("%-8s %9.2f %9.2e | %9.2f %9.3f | %+8.3f %+8.3f %+8.3f | %7.3f %7.3f (p %.2f, n %d)" %
          (c, b_raw[0][0], b_raw[1][0], b_std[0][0], b_std[1][0], m.params[1], m.params[2], m.params[3], w.pvalue, hit, pb, len(hits)))

print("\n(b) EFFECTIVE SPREAD ESTIMATES FROM DAILY BARS (bp, median of monthly estimates)")
def ests(o, h, l, c):
    lc, lh, ll = np.log(c), np.log(h), np.log(l)
    eta = (lh + ll) / 2
    ar = 4 * (lc - eta) * (lc - eta.shift(-1))          # Abdi-Ranaldo two-day version
    dc = lc.diff(); roll = -dc * dc.shift(1)              # Roll covariance
    # Corwin-Schultz
    beta = (np.log(h / l)) ** 2 + (np.log(h.shift(-1) / l.shift(-1))) ** 2
    H2 = pd.concat([h, h.shift(-1)], axis=1).max(axis=1); L2 = pd.concat([l, l.shift(-1)], axis=1).min(axis=1)
    gam = np.log(H2 / L2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gam / k)
    cs = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    df = pd.DataFrame({"ar": ar, "roll": roll, "cs": cs})
    mo = df.groupby(df.index.to_period("M")).mean()
    out = {"AR": np.sqrt(mo.ar.clip(lower=0)), "Roll": 2 * np.sqrt(mo.roll.clip(lower=0)), "CS": mo.cs.clip(lower=0)}
    return {k_: float(1e4 * v.median()) for k_, v in out.items()}
SP = {}
for sym, nm in list(D.SYM.items()) + [("DX", "DXY")]:
    q = pd.read_csv(str(paths.FUTURES / (sym + ".csv")), skipfooter=1, engine="python")
    q = pd.DataFrame({"O": pd.to_numeric(q.Open).values, "H": pd.to_numeric(q.High).values, "L": pd.to_numeric(q.Low).values,
                      "C": pd.to_numeric(q.Latest).values, "Sym": q.Symbol.astype(str).values}, index=pd.to_datetime(q.Time)).sort_index()
    rollday = q.Sym != q.Sym.shift(1)
    q.loc[rollday | rollday.shift(-1, fill_value=False), ["H", "L", "C"]] = np.nan
    SP[nm] = ests(q.O, q.H, q.L, q.C)
    print("   %-7s Abdi-Ranaldo %5.1f | Roll %5.1f | Corwin-Schultz %5.1f" % (nm, SP[nm]["AR"], SP[nm]["Roll"], SP[nm]["CS"]))
res["spreads"] = SP
json.dump(res, open(os.path.join(OUT, "nonlinear_spreads.json"), "w"), indent=1, default=str)
