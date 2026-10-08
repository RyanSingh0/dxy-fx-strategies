"""Revision: Section 3-6 statistics recomputed in one place with documented settings.

  * the 24-strategy grid on the free-provider series (reproduces Table 1)
  * Newey-West t at the automatic bandwidth (8 lags) and at 9 and 12 lags
  * PSR against buy-and-hold, DSR at N = 24 and at formally estimated effective N,
    including Strategy 3 (previously n/a)
  * stationary-bootstrap Sharpe confidence intervals
  * White RC / Hansen SPA / Romano-Wolf with every setting stated, against zero and
    against buy-and-hold, reconciling the two Reality Check values in the first submission
  * effective number of independent trials: eigenvalue estimators (Nyholt 2004, Li & Ji
    2005, Galwey 2009) and clustering (Lopez de Prado & Lewis 2019 style)
  * robustness: the overnight leg excluding 2013"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd, json
from scipy import stats as st
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import statsmodels.api as sm
import paths, evaluate as E

T = 252
MAP = {1:("Long","Cash"),2:("Short","Cash"),3:("Cash","Long"),4:("Cash","Short"),5:("Long","Long"),
6:("Short","Short"),7:("Short","Long"),8:("Long","Short"),9:("Cash","Inertia"),10:("Cash","Reversal"),
11:("Inertia","Cash"),12:("Reversal","Cash"),13:("Inertia","Inertia"),14:("Inertia","Reversal"),
15:("Reversal","Inertia"),16:("Reversal","Reversal"),17:("Long","Inertia"),18:("Long","Reversal"),
19:("Short","Inertia"),20:("Short","Reversal"),21:("Inertia","Long"),22:("Reversal","Long"),
23:("Inertia","Short"),24:("Reversal","Short")}

def pos(r, s):
    if r == "Cash": return 0
    if r == "Long": return 1
    if r == "Short": return -1
    if np.isnan(s): return 0
    b = 1 if s >= 0 else -1
    return b if r == "Inertia" else -b

def load_free():
    d = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"])
    x = d[["Date","DXY_Open","DXY_High","DXY_Low","DXY_Close"]].copy()
    x.columns = ["Date","O","H","L","C"]
    return x.sort_values("Date").reset_index(drop=True)

def load_exch():
    d = pd.read_csv(str(paths.DX_FILE), skipfooter=1, engine="python")
    return pd.DataFrame({"Date": pd.to_datetime(d["Time"]), "O": pd.to_numeric(d["Open"]),
        "H": pd.to_numeric(d["High"]), "L": pd.to_numeric(d["Low"]),
        "C": pd.to_numeric(d["Latest"])}).sort_values("Date").reset_index(drop=True)

def grid(d, cb=0.0):
    co = (d.O / d.C.shift(1) - 1).values; oc = (d.C / d.O - 1).values; N = len(d)
    out = {}
    for k, (a, b) in MAP.items():
        c = cb / 1e4; prev = np.concatenate(([np.nan], oc[:-1])); held = 0
        ro = np.empty(N); rd = np.empty(N)
        for i in range(N):
            p = pos(a, prev[i]); ch = c if (p != 0 and p != held) else 0.0
            ro[i] = 0.0 if np.isnan(co[i]) else p * co[i] - ch; held = p
            q = pos(b, co[i]); ch = c if (q != 0 and q != held) else 0.0
            rd[i] = 0.0 if np.isnan(oc[i]) else q * oc[i] - ch; held = q
        out[k] = (1 + ro) * (1 + rd) - 1
    bnh = d.C.ffill().pct_change().fillna(0).values
    return pd.DataFrame(out), bnh, co, oc

sr = lambda x: np.asarray(x, float).mean() / np.asarray(x, float).std(ddof=1) * np.sqrt(T)

def nw(x, l):
    x = np.asarray(x, float)
    r = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": l})
    return float(r.tvalues[0]), float(r.pvalues[0])

def psr(x, bench_ann):
    x = np.asarray(x, float); s = x.mean() / x.std(ddof=1); n = len(x)
    g3 = st.skew(x); g4 = st.kurtosis(x, fisher=False)
    den = np.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2)
    return float(st.norm.cdf((s - bench_ann / np.sqrt(T)) * np.sqrt(n - 1) / den))

def sr0(all_sr, N):
    v = np.var(np.asarray(all_sr) / np.sqrt(T), ddof=1); g = np.euler_gamma
    return float(np.sqrt(v) * ((1 - g) * st.norm.ppf(1 - 1 / N) + g * st.norm.ppf(1 - 1 / (N * np.e))) * np.sqrt(T))

def neff(R):
    C = np.corrcoef(R.T); M = C.shape[0]
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    nyholt = 1 + (M - 1) * (1 - np.var(lam, ddof=1) / M)
    liji = float(sum((l >= 1) + (l - np.floor(l)) for l in lam))
    galwey = float(np.sum(np.sqrt(lam)) ** 2 / np.sum(lam))
    # share of variance and 95% / 99% rank
    cs = np.cumsum(np.sort(lam)[::-1]) / lam.sum()
    k95 = int(np.searchsorted(cs, 0.95) + 1); k99 = int(np.searchsorted(cs, 0.99) + 1)
    # clustering on |rho|: mirror pairs (rho ~ -1) are one trial; cut at |rho| = 0.5
    D_ = 1 - np.abs(C); np.fill_diagonal(D_, 0); D_ = np.clip((D_ + D_.T) / 2, 0, None)
    Z = linkage(squareform(D_, checks=False), "average")
    ncl = {t: int(fcluster(Z, t=1 - t, criterion="distance").max()) for t in (0.3, 0.5, 0.7)}
    # average-correlation rule rho + (1-rho) M (Lopez de Prado 2018), on |rho|
    rbar = (np.abs(C).sum() - M) / (M * (M - 1))
    avgc = rbar + (1 - rbar) * M
    return dict(nyholt=float(nyholt), liji=liji, galwey=galwey, k95=k95, k99=k99,
                clusters=ncl, avgcorr=float(avgc), rbar=float(rbar),
                top_eig_share=[float(v) for v in (np.sort(lam)[::-1][:6] / lam.sum())])

def mt(R, bench, B=5000, seed=20260828, L=21):
    f = R - (bench[:, None] if bench is not None else 0.0)
    n, k = f.shape; fbar = f.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = E.stationary_bootstrap_idx(n, B, rng, L)
    boot = np.empty((B, k))
    for b in range(B): boot[b] = f[idx[b]].mean(axis=0)
    om = boot.std(axis=0, ddof=1) * np.sqrt(n); om[om == 0] = 1e-12
    w = np.sqrt(n) * (boot - fbar)
    rc = float((w.max(axis=1) >= np.sqrt(n) * fbar.max()).mean())
    thr = -(om / np.sqrt(n)) * np.sqrt(2 * np.log(np.log(n))); g = np.where(fbar >= thr, fbar, 0.0)
    Tobs = max(0.0, float(np.max(np.sqrt(n) * fbar / om)))
    Tb = np.maximum(0.0, (np.sqrt(n) * (boot - g) / om).max(axis=1))
    spa = float((Tb >= Tobs).mean())
    stud = np.sqrt(n) * fbar / om; surv = []; act = list(range(k))
    while act:
        cr = np.percentile((w[:, act] / om[act]).max(axis=1), 95)
        rej = [x for x in act if stud[x] > cr]
        if not rej: break
        surv += rej; act = [x for x in act if x not in rej]
    return rc, spa, sorted(int(j) + 1 for j in surv)

def boot_ci(x, B=5000, seed=1, L=21):
    rng = np.random.default_rng(seed); n = len(x)
    idx = E.stationary_bootstrap_idx(n, B, rng, L)
    s = np.array([sr(x[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main():
    out = {}
    yh = load_free(); bc = load_exch()
    R, bnh, co, oc = grid(yh)
    s = R.apply(sr)
    print("TABLE 1 CHECK  S7 final %.2f SR %.3f | S2 SR %.3f | S5 final %.2f | B&H %.2f"
          % (100 * np.prod(1 + R[7]), s[7], s[2], 100 * np.prod(1 + R[5]), 100 * np.prod(1 + bnh)))
    lag_auto = int(np.floor(4 * (len(R) / 100.0) ** (2.0 / 9.0)))
    print("automatic NW bandwidth for n=%d: %d lags" % (len(R), lag_auto))
    bsr = sr(bnh)
    ne = neff(R.values)
    out["neff"] = ne
    print("effective number of trials:", json.dumps(ne))
    Ns = {"24 (nominal)": 24, "12 (mirror pairs)": 12, "Nyholt": ne["nyholt"], "Li-Ji": ne["liji"],
          "Galwey": ne["galwey"], "clusters |rho|>0.5": ne["clusters"][0.5]}
    rows = []
    print("\n%-4s %7s %7s %9s %7s %9s %7s %8s | DSR at N = %s" % ("#", "SR", "t8", "p8", "t9", "p9", "t12", "PSRb",
          ", ".join(Ns)))
    for k in [7, 2, 22, 20, 3]:
        x = R[k].values
        t8, p8 = nw(x, 8); t9, p9 = nw(x, 9); t12, p12 = nw(x, 12)
        pb = psr(x, bsr)
        dsrs = {nm: psr(x, sr0(s.values, max(N, 1.0001))) for nm, N in Ns.items()}
        lo, hi = boot_ci(x)
        rows.append(dict(k=k, sr=s[k], t8=t8, p8=p8, t9=t9, p9=p9, t12=t12, p12=p12, psrb=pb,
                         dsr=dsrs, ci=(lo, hi)))
        print("S%-3d %7.3f %7.3f %9.2e %7.3f %9.2e %7.3f %8.4f | %s | CI [%.3f, %.3f]" % (k, s[k], t8, p8, t9, p9, t12, pb,
              " ".join("%.3f" % v for v in dsrs.values()), lo, hi))
    t8, p8 = nw(bnh, 8)
    print("B&H  SR %.3f t8 %.3f p8 %.4f" % (bsr, t8, p8))
    out["bnh"] = dict(sr=bsr, t8=t8, p8=p8)
    out["rows"] = rows
    out["sr0"] = {nm: sr0(s.values, max(N, 1.0001)) for nm, N in Ns.items()}
    print("expected max under null:", {k: round(v, 3) for k, v in out["sr0"].items()})

    print("\nMULTIPLE TESTING, all 24 strategies, stationary bootstrap B=5000, mean block 21, seed 20260828")
    mtres = {}
    for lab, bench in [("zero", None), ("buy-and-hold", bnh)]:
        rc, spa, rw = mt(R.values, bench)
        mtres[lab] = (rc, spa, rw)
        print("   vs %-13s RC p %.4f | SPA p %.4f | RW survivors %s" % (lab, rc, spa, rw))
    out["mt"] = mtres
    # seed sensitivity of the RC p-value vs buy-and-hold
    seeds = [mt(R.values, bnh, B=5000, seed=sd)[0] for sd in range(1, 11)]
    print("   RC p vs B&H across 10 further seeds: min %.4f max %.4f mean %.4f" % (min(seeds), max(seeds), np.mean(seeds)))
    seeds0 = [mt(R.values, None, B=5000, seed=sd)[0] for sd in range(1, 11)]
    print("   RC p vs zero across 10 further seeds: min %.4f max %.4f mean %.4f" % (min(seeds0), max(seeds0), np.mean(seeds0)))
    out["rc_seed"] = dict(bnh=seeds, zero=seeds0)
    for L in (5, 10, 63):
        a = mt(R.values, bnh, B=5000, L=L); b = mt(R.values, None, B=5000, L=L)
        print("   block %d: RC vs B&H %.4f, SPA %.4f | RC vs zero %.4f, SPA %.4f" % (L, a[0], a[1], b[0], b[1]))
        out["mt_L%d" % L] = (a[:2], b[:2])

    print("\nROBUSTNESS: free-series overnight leg excluding calendar 2013")
    yr = yh.Date.dt.year.values
    c0 = np.nan_to_num(co)
    for lab, m in [("all", np.ones(len(c0), bool)), ("ex-2013", yr != 2013), ("2013 only", yr == 2013),
                   ("2014-2025", yr >= 2014)]:
        t, p = nw(c0[m], 8)
        print("   %-10s n %4d  SR %+.3f  t %+.2f  p %.2e  100->%.2f" % (lab, m.sum(), sr(c0[m]), t, p, 100 * np.prod(1 + c0[m])))
        out["on_" + lab] = (int(m.sum()), sr(c0[m]), t, p)
    # exchange overnight for the same windows
    yb = bc.Date.dt.year.values
    cb_ = np.nan_to_num((bc.O / bc.C.shift(1) - 1).values)
    for lab, m in [("all", np.ones(len(cb_), bool)), ("ex-2013", yb != 2013), ("2013 only", yb == 2013)]:
        t, p = nw(cb_[m], 8)
        print("   exch %-10s n %4d SR %+.3f t %+.2f p %.3f" % (lab, m.sum(), sr(cb_[m]), t, p))
        out["on_exch_" + lab] = (int(m.sum()), sr(cb_[m]), t, p)
    # strategy 7 ex-2013
    m = yr != 2013
    print("   S7 ex-2013 SR %.3f, S2 ex-2013 SR %.3f" % (sr(R[7].values[m]), sr(R[2].values[m])))
    out["S7_ex2013"] = sr(R[7].values[m]); out["S2_ex2013"] = sr(R[2].values[m])
    # terminal value reconciliation
    tv5 = 100 * np.prod(1 + R[5]); tvb = 100 * yh.C.ffill().iloc[-1] / yh.C.iloc[0]
    cc = (yh.C / yh.C.shift(1) - 1).fillna(0); tvs = 100 * np.prod(1 + cc)
    print("\nTERMINAL VALUES: S5 %.2f | B&H ffill %.2f | close-to-close skipped %.2f" % (tv5, tvb, tvs))
    out["tv"] = (tv5, tvb, tvs)
    # Sharpe of S5 vs bnh
    print("S5 SR %.3f vs B&H SR %.3f" % (s[5], bsr))
    json.dump(out, open(os.path.join(str(paths.RESULTS), "revision", "naive_stats.json"), "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
