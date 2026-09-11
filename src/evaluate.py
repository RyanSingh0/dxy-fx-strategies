"""Evaluation harness.

The one-period shift lives here and nowhere else. Weights formed at the close of day t
earn the return of day t+1. Everything goes through apply_positions, so no strategy can
look ahead by accident.

Statistics implemented from source:
  Newey and West (1987)                 HAC t-statistic
  Politis and Romano (1994)             stationary bootstrap
  White (2000)                          Reality Check
  Hansen (2005)                         Superior Predictive Ability, with recentring
  Romano and Wolf (2005)                stepwise familywise error control
  Bailey and Lopez de Prado (2014)      probabilistic and deflated Sharpe ratio
  Bailey, Borwein, Lopez de Prado and Zhu (2017)  CSCV probability of backtest overfitting
  Harvey and Liu (2015)                 haircut Sharpe ratio (Bonferroni, Holm, BHY)"""
import numpy as np, pandas as pd, itertools
from scipy import stats as st
import statsmodels.api as sm

TD = 252


# ------------------------------------------------------------------ core
def apply_positions(W, P, cost_bp=1.0):
    """THE shift. Returns (gross, net, turnover, gross_exposure)."""
    Wl = W.shift(1).fillna(0.0)
    gross = (Wl * P["ret"]).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = gross - turn * (cost_bp / 1e4)
    return gross, net, turn, W.abs().sum(axis=1)


def sharpe(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 2: return np.nan
    s = x.std(ddof=0)
    return float(x.mean() / s * np.sqrt(TD)) if s > 0 else np.nan


def summary(r, turn=None, gexp=None):
    r = pd.Series(r).dropna()
    if len(r) < 2: return {}
    eq = (1 + r).cumprod()
    dn = r[r < 0].std(ddof=0)
    t, p = nw_t(r)
    out = dict(TotRet=float(eq.iloc[-1] - 1), CAGR=float(eq.iloc[-1] ** (TD / len(r)) - 1),
               Sharpe=sharpe(r), Vol=float(r.std(ddof=0) * np.sqrt(TD)),
               Sortino=float(r.mean() / dn * np.sqrt(TD)) if dn and dn > 0 else np.nan,
               MaxDD=float((eq / eq.cummax() - 1).min()), Skew=float(st.skew(r)),
               Kurt=float(st.kurtosis(r, fisher=False)), t=t, p=p, n=len(r),
               HitRate=float((r > 0).mean()))
    if turn is not None:
        out["TurnoverPA"] = float(pd.Series(turn).reindex(r.index).fillna(0).mean() * TD)
    if gexp is not None:
        out["GrossExp"] = float(pd.Series(gexp).reindex(r.index).fillna(0).mean())
    return out


def nw_t(r, lags=None):
    r = np.asarray(pd.Series(r).dropna(), float)
    if len(r) < 30 or r.std() == 0: return np.nan, np.nan
    if lags is None:
        lags = int(np.floor(4 * (len(r) / 100.0) ** (2.0 / 9.0)))   # Newey-West rule of thumb
    m = sm.OLS(r, np.ones((len(r), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.tvalues[0]), float(m.pvalues[0])


# ------------------------------------------------------------------ bootstrap
def stationary_bootstrap_idx(n, B, rng, mean_block=21):
    """Politis & Romano (1994), vectorised across replicates."""
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=np.int64)
    cur = rng.integers(0, n, size=B)
    jump = rng.random((B, n)) < p
    fresh = rng.integers(0, n, size=(B, n))
    for t in range(n):
        idx[:, t] = cur
        cur = np.where(jump[:, t], fresh[:, t], (cur + 1) % n)
    return idx


def multiple_testing(R, bench=None, B=2000, seed=7, mean_block=21, chunk=400):
    """White RC, Hansen SPA, Romano-Wolf over the columns of R (a DataFrame of returns)."""
    f = (R.sub(bench, axis=0) if bench is not None else R).values.astype(float)
    n, k = f.shape
    fbar = f.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, B, rng, mean_block)
    boot = np.empty((B, k))
    for a in range(0, B, chunk):
        b = min(a + chunk, B)
        C = np.zeros((b - a, n))
        for i, row in enumerate(idx[a:b]):
            C[i] = np.bincount(row, minlength=n)
        boot[a:b] = (C @ f) / n
    om = boot.std(axis=0, ddof=1) * np.sqrt(n)
    om[om <= 0] = 1e-12
    w = np.sqrt(n) * (boot - fbar)
    p_rc = float((w.max(axis=1) >= np.sqrt(n) * fbar.max()).mean())
    thr = -(om / np.sqrt(n)) * np.sqrt(2 * np.log(np.log(n)))
    g = np.where(fbar >= thr, fbar, 0.0)
    Tobs = max(0.0, float(np.max(np.sqrt(n) * fbar / om)))
    Tb = np.maximum(0.0, (np.sqrt(n) * (boot - g) / om).max(axis=1))
    p_spa = float((Tb >= Tobs).mean())
    stud = np.sqrt(n) * fbar / om
    surv, act = [], list(range(k))
    while act:
        crit = np.percentile((w[:, act] / om[act]).max(axis=1), 95)
        rej = [j for j in act if stud[j] > crit]
        if not rej: break
        surv += rej
        act = [j for j in act if j not in rej]
    order = sorted(surv, key=lambda j: -fbar[j])
    return dict(rc=p_rc, spa=p_spa, rw=[R.columns[j] for j in order], n_rw=len(surv))


# ------------------------------------------------------------------ deflation
def psr(r, sr_bench_ann=0.0):
    r = pd.Series(r).dropna()
    n = len(r)
    s = sharpe(r) / np.sqrt(TD)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * s + (g4 - 1) / 4.0 * s ** 2))
    return float(st.norm.cdf((s - sr_bench_ann / np.sqrt(TD)) * np.sqrt(n - 1) / den))


def expected_max_sharpe(all_sr_ann, N):
    v = np.nanvar(np.asarray(all_sr_ann, float) / np.sqrt(TD), ddof=1)
    e = np.euler_gamma
    return float(np.sqrt(v) * ((1 - e) * st.norm.ppf(1 - 1.0 / N)
                               + e * st.norm.ppf(1 - 1.0 / (N * np.e))) * np.sqrt(TD))


def deflated_sharpe(r, all_sr_ann, N):
    sr0 = expected_max_sharpe(all_sr_ann, N)
    return psr(r, sr0), sr0


def harvey_liu_haircut(t_obs, N, rho=0.2):
    """Harvey & Liu (2015). Returns adjusted p-values under Bonferroni, Holm and BHY."""
    p_single = 2 * (1 - st.norm.cdf(abs(t_obs)))
    p_bonf = min(1.0, p_single * N)
    p_holm = min(1.0, p_single * N)          # for the single most significant test these coincide
    c = np.sum(1.0 / np.arange(1, N + 1))
    p_bhy = min(1.0, p_single * N * c / 1.0)
    hc = lambda p: (abs(t_obs) - st.norm.ppf(1 - p / 2)) / abs(t_obs) if abs(t_obs) > 0 else np.nan
    return dict(p_single=p_single, p_bonf=p_bonf, p_holm=p_holm, p_bhy=p_bhy,
                haircut_bonf=hc(p_bonf), haircut_bhy=hc(p_bhy))


# ------------------------------------------------------------------ overfitting
def cscv_pbo(R, S=16):
    M = R.values.astype(float)
    k = M.shape[1]
    parts = np.array_split(np.arange(len(M)), S)
    cnt = np.array([len(b) for b in parts], float)
    bs = np.array([M[b].sum(axis=0) for b in parts])
    bss = np.array([(M[b] ** 2).sum(axis=0) for b in parts])
    lam, rk = [], []
    allb = set(range(S))
    for cmb in itertools.combinations(range(S), S // 2):
        c = list(cmb); o = list(allb - set(cmb))
        out = []
        for sel in (c, o):
            n_ = cnt[sel].sum(); su = bs[sel].sum(axis=0); sq = bss[sel].sum(axis=0)
            mu = su / n_
            var = np.maximum(sq / n_ - mu ** 2, 1e-300)
            out.append(mu / np.sqrt(var))
        a, b = out
        j = int(np.nanargmax(a))
        r_ = 1 + int((b < b[j]).sum())
        ww = r_ / (k + 1.0)
        lam.append(np.log(ww / (1 - ww))); rk.append(r_)
    return dict(pbo=float((np.array(lam) <= 0).mean()), med_rank=float(np.median(rk)), k=k)
