"""Revision 2: does the binomial screen survive dependence between sessions?

The exact binomial test treats the sessions on which the open sits at an extreme as
independent draws with P(low) = 1/2. The size study in screen_study.py simulates
independent sessions, so it says nothing about dependence. Here I check both sides.

  A. Empirical: is the sequence of extreme types serially dependent? Wald-Wolfowitz runs
     test on the ordered low/high sequence, and the first-order autocorrelation of the
     session indicator d_t (+1 open at low, -1 open at high, 0 otherwise).
  B. Dependence-robust versions of the screen: a Newey-West test that the mean of d_t is
     zero (8, 21 and 63 lags), and a sign test over calendar years (each year one vote,
     ties dropped), which allows any dependence within a year.
  C. Size when sessions are dependent: the screen_study simulation with (i) volatility
     that persists across sessions (AR(1) log volatility, phi = 0.98) and (ii) a zero-mean
     intraday drift that persists across sessions (AR(1), phi = 0.9), at two sizes."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import json
import numpy as np, pandas as pd
from scipy import stats as st
import statsmodels.api as sm
import paths

OUT = os.path.join(str(paths.RESULTS), "revision"); res = {}

def load_fut(sym):
    d = pd.read_csv(str(paths.FUTURES / (sym + ".csv")), skipfooter=1, engine="python")
    return pd.DataFrame({"Date": pd.to_datetime(d["Time"]), "O": pd.to_numeric(d["Open"]),
        "H": pd.to_numeric(d["High"]), "L": pd.to_numeric(d["Low"]),
        "C": pd.to_numeric(d["Latest"])}).sort_values("Date").reset_index(drop=True)

def load_free():
    d = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"])
    x = d[["Date", "DXY_Open", "DXY_High", "DXY_Low", "DXY_Close"]].copy()
    x.columns = ["Date", "O", "H", "L", "C"]
    return x.dropna().reset_index(drop=True)

def indicator(d):
    # same definition as screen_study.counts at delta = 0
    O, H, L = (d[c].values.astype(float) for c in "OHL")
    k = np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & (H > L)
    lo = np.isclose(O, L, rtol=0, atol=1e-9) & k; hi = np.isclose(O, H, rtol=0, atol=1e-9) & k
    return pd.Series(lo.astype(int) - hi.astype(int), index=d.Date)[k]

def runs_test(seq):
    seq = np.asarray(seq); n1 = int((seq == 1).sum()); n2 = int((seq == -1).sum()); n = n1 + n2
    runs = 1 + int((seq[1:] != seq[:-1]).sum())
    mu = 2 * n1 * n2 / n + 1; var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return runs, mu, z, 2 * st.norm.sf(abs(z))

def hac_mean(x, L):
    m = sm.OLS(np.asarray(x, float), np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    return float(m.tvalues[0]), float(m.pvalues[0])

SER = [("Dollar index, free provider", load_free())] + \
      [(nm, load_fut(s)) for s, nm in [("DX", "Dollar index futures"), ("E6", "Euro futures"),
       ("B6", "Sterling futures"), ("D6", "Canadian dollar futures"), ("J6", "Yen futures"),
       ("S6", "Swiss franc futures")]]

print("A/B. DEPENDENCE IN THE EXTREME-LOCATED OPENS, AND DEPENDENCE-ROBUST SCREENS")
print("%-26s %5s %5s %10s | %6s %7s %6s %7s | %7s | %9s %9s %9s | %s"
      % ("series", "low", "high", "binom p", "runs", "E[runs]", "z", "p", "acf1 d", "NW8 p", "NW21 p", "NW63 p", "years low>high / decided, sign p"))
A = {}
for lab, d in SER:
    dt = indicator(d)
    nl, nh = int((dt == 1).sum()), int((dt == -1).sum())
    pb = float(st.binomtest(nl, nl + nh, 0.5).pvalue)
    seq = dt[dt != 0].values
    runs, mu, z, pr = runs_test(seq)
    acf1 = float(np.corrcoef(dt.values[1:], dt.values[:-1])[0, 1])
    nws = [hac_mean(dt.values, L)[1] for L in (8, 21, 63)]
    yr = dt.groupby(dt.index.year).sum()
    up, dn = int((yr > 0).sum()), int((yr < 0).sum())
    ps = float(st.binomtest(up, up + dn, 0.5).pvalue) if up + dn else np.nan
    A[lab] = dict(low=nl, high=nh, binom=pb, runs=runs, runs_exp=mu, runs_z=z, runs_p=pr, acf1=acf1,
                  nw=nws, years_low=up, years_decided=up + dn, sign_p=ps)
    print("%-26s %5d %5d %10.2e | %6d %7.1f %6.2f %7.3f | %+7.3f | %9.2e %9.2e %9.2e | %d / %d, p %.3g"
          % (lab, nl, nh, pb, runs, mu, z, pr, acf1, *nws, up, up + dn, ps))
res["A"] = A
d = SER[0][1]; dt = indicator(d)
ex = dt[dt.index.year != 2013]
nl, nh = int((ex == 1).sum()), int((ex == -1).sum())
print("   free provider excluding 2013: %d/%d binomial p %.2e, NW21 p %.2e"
      % (nl, nh, st.binomtest(nl, nl + nh, 0.5).pvalue, hac_mean(ex.values, 21)[1]))
res["free_ex2013"] = (nl, nh, float(st.binomtest(nl, nl + nh, 0.5).pvalue), hac_mean(ex.values, 21)[1])

# ------------------------------------------------------------------ C
print("\nC. MONTE CARLO SIZE WITH DEPENDENCE ACROSS SESSIONS (3,800 sessions, 300 steps, DX tick, 400 samples)")
dx = load_fut("DX")
sd_day = float(np.std(np.log(dx.C).diff().dropna()))
mu_day = float(np.log(dx.C).diff().dropna().mean())
tick = 0.005; px = round(float(dx.C.median()) / tick) * tick
rng = np.random.default_rng(20261015)

def ar1(b, n, phi, sd, rng):
    e = rng.standard_normal((b, n)) * sd * np.sqrt(1 - phi ** 2)
    x = np.empty((b, n)); x[:, 0] = rng.standard_normal(b) * sd
    for t in range(1, n): x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x

def simulate(n_sess=3800, K=300, reps=400, vol_phi=None, drift_phi=None, drift_sd=0.0, chunk=50):
    rej5 = rej1 = 0; rej5_nw = 0; acfs = []; share = []; plow = []
    for c0 in range(0, reps, chunk):
        b = min(chunk, reps - c0)
        vol = np.full((b, n_sess), sd_day / np.sqrt(K))
        if vol_phi is not None:   # log vol AR(1), same unconditional spread as screen_study (0.4)
            lv = ar1(b, n_sess, vol_phi, 0.4, rng); vol = vol * np.exp(lv - 0.08)
        drift = np.zeros((b, n_sess))
        if drift_phi is not None:
            drift = ar1(b, n_sess, drift_phi, drift_sd, rng)
        z = rng.standard_normal((b, n_sess, K))
        inc = z * vol[:, :, None] + (drift / K)[:, :, None]
        path = np.concatenate([np.zeros((b, n_sess, 1)), np.cumsum(inc, axis=2)], axis=2)
        p = (px + tick * rng.random((b, n_sess, 1))) * np.exp(path)
        p = np.round(p / tick) * tick
        O = p[:, :, 0]; H = p.max(axis=2); L = p.min(axis=2)
        valid = H > L
        lo = np.isclose(O, L, atol=1e-9) & valid; hi = np.isclose(O, H, atol=1e-9) & valid
        share.append(((lo.sum(axis=1) + hi.sum(axis=1)) / valid.sum(axis=1)).mean())
        plow.append((lo.sum(axis=1) / np.maximum(lo.sum(axis=1) + hi.sum(axis=1), 1)).mean())
        for i in range(b):
            nl, nh = int(lo[i].sum()), int(hi[i].sum())
            pv = st.binomtest(nl, nl + nh, 0.5).pvalue if nl + nh else 1.0
            rej5 += pv < 0.05; rej1 += pv < 0.01
            dd = lo[i].astype(int) - hi[i].astype(int)
            acfs.append(np.corrcoef(dd[1:], dd[:-1])[0, 1])
            rej5_nw += hac_mean(dd, 21)[1] < 0.05
    return dict(rej5=rej5 / reps, rej1=rej1 / reps, rej5_nw21=rej5_nw / reps, acf1=float(np.nanmean(acfs)),
                share_ext=float(np.mean(share)), p_low=float(np.mean(plow)))

scen = [("independent sessions (reference)", dict()),
        ("persistent volatility, log-vol AR(1) phi 0.98", dict(vol_phi=0.98)),
        ("persistent zero-mean drift, AR(1) phi 0.9, sd = 10% of daily vol", dict(drift_phi=0.9, drift_sd=0.10 * sd_day)),
        ("persistent zero-mean drift, AR(1) phi 0.9, sd = 25% of daily vol", dict(drift_phi=0.9, drift_sd=0.25 * sd_day))]
C = {}
for lab, kw in scen:
    r = simulate(**kw); C[lab] = r
    print("  %-66s size5 %.3f size1 %.3f | NW21 size5 %.3f | mean acf1 of d %.4f | share-at-extreme %.3f P(low|ext) %.3f"
          % (lab, r["rej5"], r["rej1"], r["rej5_nw21"], r["acf1"], r["share_ext"], r["p_low"]))
res["C"] = C
res["meta"] = dict(sd_day=sd_day, mu_day=mu_day)
json.dump(res, open(os.path.join(OUT, "screen_dependence.json"), "w"), indent=1, default=float)
