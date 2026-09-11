"""Engine for the regime-filtered cross-sectional momentum rule.

Four defects in the original implementation of the rule are corrected here:

  1. to_period('2Q') is silently treated as 'Q', so the semiannual row was a
     byte-identical duplicate of the quarterly one. Semiannual is grouped explicitly.
  2. the pair price files are one trading day late relative to the DXY file, so the
     primary results run on exchange futures, which are correctly dated.
  3. White RC, SPA and DSR were computed against DXY buy and hold. For a self-financing
     long/short book the null is zero. Both are reported.
  4. np.where(mom < 0, 1, -1) maps NaN to -1, so the first LB sessions, where no momentum
     can be computed yet, were traded as though the dollar were rising.

The free parameters are enumerated rather than assumed: rebalance frequency, rank
cardinality, momentum lookback, which slice of the cross-section and regime on or off.
The multiple-testing correction is then applied to the search that was actually run.
Factor attribution against dollar beta, FX carry and unconditional FX momentum follows,
then walk-forward with reselection over the full grid, PSR and DSR against zero,
break-even cost and downside risk."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, itertools, warnings
from scipy import stats as st
import statsmodels.api as sm
warnings.filterwarnings("ignore")

TD = 252; START = 100.0
FUTDIR = str(paths.FUTURES)
NAME = {"E6": "EURUSD", "B6": "GBPUSD", "D6": "CADUSD", "J6": "JPYUSD", "S6": "CHFUSD"}
DXP = str(paths.DX_FILE)

# ---------------------------------------------------------------- data
def load_bc(path):
    d = pd.read_csv(path, skipfooter=1, engine="python")
    return pd.DataFrame({"Date": pd.to_datetime(d["Time"]), "Sym": d["Symbol"].astype(str),
                         "C": pd.to_numeric(d["Latest"], errors="coerce")}
                        ).sort_values("Date").set_index("Date")

def within_contract_returns(d):
    r = d.C.pct_change(); r[d.Sym != d.Sym.shift(1)] = np.nan
    return r.iloc[1:]

FUT = {NAME[k]: load_bc(str(paths.FUTURES / (k + ".csv"))) for k in NAME}
DX = load_bc(DXP)
_fr = pd.read_csv(str(paths.FRED_DAILY), parse_dates=["Date"]).set_index("Date")
SPOT = pd.DataFrame({"EURUSD": _fr.DEXUSEU, "GBPUSD": _fr.DEXUSUK, "JPYUSD": 1 / _fr.DEXJPUS,
                     "CADUSD": 1 / _fr.DEXCAUS, "CHFUSD": 1 / _fr.DEXSZUS, "SEKUSD": 1 / _fr.DEXSDUS})
FRED_DXY = (50.14348112 * _fr.DEXUSEU ** -0.576 * _fr.DEXJPUS ** 0.136 * _fr.DEXUSUK ** -0.119
            * _fr.DEXCAUS ** 0.091 * _fr.DEXSDUS ** 0.042 * _fr.DEXSZUS ** 0.036)
RATES = pd.read_csv(str(paths.FRED_RATES), parse_dates=["Date"]).set_index("Date")
RCOL = {"EURUSD": "IR3TIB01EZM156N", "GBPUSD": "IR3TIB01GBM156N", "JPYUSD": "IR3TIB01JPM156N",
        "CADUSD": "IR3TIB01CAM156N", "CHFUSD": "IR3TIB01CHM156N", "SEKUSD": "IR3TIB01SEM156N"}

def universe(kind="futures"):
    """returns (pair daily returns, DXY continuous level, DXY daily return)"""
    if kind == "futures":
        R = pd.DataFrame({nm: within_contract_returns(d).fillna(0.0) for nm, d in FUT.items()}).dropna(how="any")
        dxr = within_contract_returns(DX).fillna(0.0)
        lvl = START * (1 + dxr).cumprod()
        return R, lvl, dxr.reindex(R.index).fillna(0.0)
    if kind == "futures6":
        R = pd.DataFrame({nm: within_contract_returns(d).fillna(0.0) for nm, d in FUT.items()})
        R["SEKUSD"] = SPOT["SEKUSD"].pct_change().reindex(R.index)
        R = R.dropna(how="any")
        dxr = within_contract_returns(DX).fillna(0.0)
        lvl = START * (1 + dxr).cumprod()
        return R, lvl, dxr.reindex(R.index).fillna(0.0)
    R = SPOT.pct_change().dropna(how="any")
    return R, FRED_DXY, FRED_DXY.pct_change().reindex(R.index).fillna(0.0)

def period_key(idx, freq):
    """FIX: '2Q'/semiannual is built explicitly. pandas to_period('2Q') silently returns 'Q'."""
    if freq == "S":
        return pd.Index([f"{d.year}-S{1 if d.quarter <= 2 else 2}" for d in idx])
    return idx.to_period(freq).astype(str)

# ---------------------------------------------------------------- engine
def strat(R, lvl, freq="Q", slice_="bottom", card=2, signed=True, LB=126, cost_bp=0.0,
          mom_calendar="universe"):
    """mom_calendar: 'universe' computes the LB-session lookback on the traded universe's
    calendar, which is what the original implementation did, with DXY sitting inside the
    merged frame. 'native' computes it on the DXY series' own calendar before alignment.
    The two differ because the DXY futures file has about 111 more sessions than the pair
    panel."""
    if slice_ != "median" and 2 * card > R.shape[1]:
        return pd.Series(dtype=float)
    if mom_calendar == "native":
        mom = (lvl / lvl.shift(LB) - 1.0)
    else:
        mom = (lvl.reindex(R.index).ffill())
        mom = mom / mom.shift(LB) - 1.0
    regime = pd.Series(np.where(mom < 0, 1.0, -1.0), index=mom.index)
    # np.where maps NaN to -1, which would trade the first LB sessions as though the
    # dollar were rising. No position until the lookback window actually exists.
    regime[mom.isna()] = np.nan
    keys = period_key(R.index, freq)
    grouped = {k: v for k, v in R.groupby(keys)}
    per = sorted(grouped.keys())
    out = []
    for i in range(1, len(per)):
        pp, cp = per[i - 1], per[i]
        o = ((1 + grouped[pp]).prod(axis=0) - 1).dropna().sort_values(ascending=False)
        if len(o) < 2 * card: continue
        if slice_ == "top":      picks = o.index[:card].tolist()
        elif slice_ == "bottom": picks = o.index[-card:].tolist()
        else:
            m = len(o) // 2; lo = max(0, m - card // 2); picks = o.index[lo:lo + card].tolist()
        if signed:
            ld = grouped[pp].index[-1]
            if ld not in regime.index or not np.isfinite(regime.loc[ld]): continue
            sign = -regime.loc[ld] if slice_ == "bottom" else regime.loc[ld]
        else:
            sign = 1.0
        blk = grouped[cp][picks]
        if blk.empty: continue
        s = sign * blk.mean(axis=1)
        if cost_bp: s.iloc[0] = s.iloc[0] - cost_bp / 1e4
        out.append(s)
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)

FREQS = ["D", "W", "M", "Q", "S", "Y"]   # "Y"=annual; older pandas spelled it "A" and
                                         # "S"=semiannual, which pandas cannot express as a period alias
SLICES = ["top", "median", "bottom"]
CARDS = [1, 2, 3]
LBS = [63, 126, 252]
def all_configs():
    c = [(f, s, k, True, lb) for f in FREQS for s in SLICES for k in CARDS for lb in LBS]
    c += [(f, s, k, False, 126) for f in FREQS for s in SLICES for k in CARDS]
    return c

def build_grid(R, lvl, cost_bp=0.0, minlen=500, verbose=False):
    """No silent failures. Every configuration is either included or reported with a reason.
    The original implementation wrapped this loop in a bare `except: pass`, which on
    current pandas discarded every annual configuration without saying so, because the
    alias 'A' was renamed 'Y'."""
    cols = {}; dropped = []
    for f, s, k, sg, lb in all_configs():
        key = (f, s, k, "reg%d" % lb if sg else "plain")
        try:
            r = strat(R, lvl, f, s, k, sg, lb, cost_bp)
        except Exception as e:
            dropped.append((key, "error: %s" % e)); continue
        if len(r) == 0:
            dropped.append((key, "universe too small for cardinality %d" % k)); continue
        if len(r) <= minlen:
            dropped.append((key, "only %d sessions" % len(r))); continue
        cols[key] = r
    if verbose and dropped:
        reasons = {}
        for kk, why in dropped: reasons.setdefault(why.split(":")[0], []).append(kk)
        for why, ks in reasons.items():
            print("      dropped %3d configurations - %s" % (len(ks), why))
    return pd.DataFrame(cols).dropna(), dropped

# ---------------------------------------------------------------- statistics
def sr(x):
    x = pd.Series(x).dropna()
    if len(x) < 2: return np.nan
    s = x.std(ddof=0)
    return float(x.mean() / s * np.sqrt(TD)) if s > 0 else np.nan

def perf(r):
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return dict(End=np.nan,Sharpe=np.nan,Vol=np.nan,Sortino=np.nan,MaxDD=np.nan,Skew=np.nan,Kurt=np.nan,n=len(r))
    r = r; eq = START * (1 + r).cumprod(); sd = r.std(ddof=0)
    dn = r[r < 0].std(ddof=0)
    return dict(End=float(eq.iloc[-1]), Sharpe=sr(r), Vol=float(sd * np.sqrt(TD)),
                Sortino=float(r.mean() / dn * np.sqrt(TD)) if dn > 0 else np.nan,
                MaxDD=float((eq / eq.cummax() - 1).min()),
                Skew=float(st.skew(r)), Kurt=float(st.kurtosis(r, fisher=False)), n=len(r))

def nw(r, l=8):
    r = np.asarray(pd.Series(r).dropna(), float)
    if len(r) < 30: return np.nan, np.nan
    m = sm.OLS(r, np.ones((len(r), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': l})
    return float(m.tvalues[0]), float(m.pvalues[0])

def psr(r, sr_bench=0.0):
    r = pd.Series(r).dropna(); n = len(r); s = sr(r) / np.sqrt(TD)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    den = np.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2)
    return float(st.norm.cdf((s - sr_bench / np.sqrt(TD)) * np.sqrt(n - 1) / den))

def dsr(r, all_sr_ann, N):
    """Bailey & Lopez de Prado deflated Sharpe: a CONFIDENCE LEVEL, needs > 0.95."""
    v = np.var(np.asarray(all_sr_ann) / np.sqrt(TD), ddof=1); e = np.euler_gamma
    sr0 = np.sqrt(v) * ((1 - e) * st.norm.ppf(1 - 1 / N) + e * st.norm.ppf(1 - 1 / (N * np.e)))
    return float(psr(r, sr0 * np.sqrt(TD))), float(sr0 * np.sqrt(TD))

def stationary_bootstrap_idx(n, B, rng, mean_block=21):
    """Politis & Romano (1994). Vectorised across the B replicates: step through the n
    positions once, advancing all B chains together."""
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=np.int64)
    cur = rng.integers(0, n, size=B)
    jump = rng.random((B, n)) < p
    fresh = rng.integers(0, n, size=(B, n))
    for t in range(n):
        idx[:, t] = cur
        cur = np.where(jump[:, t], fresh[:, t], (cur + 1) % n)
    return idx

def multiple_testing(G, bench=None, B=5000, seed=20260828, mean_block=21):
    f_ = (G.sub(bench, axis=0) if bench is not None else G).values.astype(float)
    n, k = f_.shape; fbar = f_.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, B, rng, mean_block)
    # counts @ data is far faster than gathering B times
    C = np.zeros((B, n))
    for b in range(B):
        C[b] = np.bincount(idx[b], minlength=n)
    boot = (C @ f_) / n
    om = boot.std(axis=0, ddof=1) * np.sqrt(n); om[om == 0] = 1e-12
    w = np.sqrt(n) * (boot - fbar)
    p_rc = float((w.max(axis=1) >= np.sqrt(n) * fbar.max()).mean())
    thr = -(om / np.sqrt(n)) * np.sqrt(2 * np.log(np.log(n))); g = np.where(fbar >= thr, fbar, 0.0)
    Tobs = max(0.0, float(np.max(np.sqrt(n) * fbar / om)))
    Tb = np.maximum(0.0, (np.sqrt(n) * (boot - g) / om).max(axis=1))
    p_spa = float((Tb >= Tobs).mean())
    stud = np.sqrt(n) * fbar / om; surv = []; act = list(range(k))
    while act:
        cr = np.percentile((w[:, act] / om[act]).max(axis=1), 95)
        rej = [x for x in act if stud[x] > cr]
        if not rej: break
        surv += rej; act = [x for x in act if x not in rej]
    return p_rc, p_spa, [G.columns[j] for j in sorted(surv, key=lambda z: -fbar[z])]

def pbo(G, S=16):
    """CSCV (Bailey, Borwein, Lopez de Prado, Zhu). Sharpe over any union of blocks is a
    function of that union's count, sum and sum-of-squares, so the blocks are reduced once
    and every split is then pure arithmetic instead of a re-slice of the return matrix."""
    M = G.values.astype(float); k = M.shape[1]
    parts = np.array_split(np.arange(len(M)), S)
    cnt = np.array([len(b) for b in parts], float)
    bs = np.array([M[b].sum(axis=0) for b in parts])
    bss = np.array([(M[b] ** 2).sum(axis=0) for b in parts])
    lam = []; rk = []
    allb = set(range(S))
    for cmb in itertools.combinations(range(S), S // 2):
        c = list(cmb); o = list(allb - set(cmb))
        out = []
        for sel_ in (c, o):
            n_ = cnt[sel_].sum(); su = bs[sel_].sum(axis=0); sq = bss[sel_].sum(axis=0)
            mu = su / n_; var = np.maximum(sq / n_ - mu ** 2, 1e-300)
            out.append(mu / np.sqrt(var) * np.sqrt(TD))
        a, b = out
        j = int(np.nanargmax(a)); r_ = 1 + int((b < b[j]).sum()); ww = r_ / (k + 1.0)
        lam.append(np.log(ww / (1 - ww))); rk.append(r_)
    return float((np.array(lam) <= 0).mean()), float(np.median(rk)), k

# ---------------------------------------------------------------- factors
def factors(R, dxr):
    """dollar, FX carry (high-minus-low 3M interbank), unconditional XS momentum"""
    rt = RATES[[RCOL[c] for c in R.columns]].rename(columns={v: k for k, v in RCOL.items()})
    rt = rt.reindex(R.index).ffill()
    car = []
    keys = period_key(R.index, "M"); grouped = {k: v for k, v in R.groupby(keys)}; per = sorted(grouped)
    for i in range(1, len(per)):
        pp, cp = per[i - 1], per[i]
        lastr = rt.loc[grouped[pp].index[-1]].dropna()
        if len(lastr) < 4: continue
        o = lastr.sort_values(ascending=False)
        lo_, hi_ = o.index[-2:].tolist(), o.index[:2].tolist()
        car.append(grouped[cp][hi_].mean(axis=1) - grouped[cp][lo_].mean(axis=1))
    CARRY = pd.concat(car).sort_index() if car else pd.Series(dtype=float)
    mo = []
    keys = period_key(R.index, "Q"); grouped = {k: v for k, v in R.groupby(keys)}; per = sorted(grouped)
    for i in range(1, len(per)):
        pp, cp = per[i - 1], per[i]
        o = ((1 + grouped[pp]).prod(axis=0) - 1).sort_values(ascending=False)
        mo.append(grouped[cp][o.index[:2].tolist()].mean(axis=1) - grouped[cp][o.index[-2:].tolist()].mean(axis=1))
    XSMOM = pd.concat(mo).sort_index() if mo else pd.Series(dtype=float)
    return pd.DataFrame({"DOLLAR": dxr, "CARRY": CARRY, "XSMOM": XSMOM}).dropna()

def attribution(r, F):
    j = pd.concat([r.rename("y"), F], axis=1).dropna()
    X = sm.add_constant(j[["DOLLAR", "CARRY", "XSMOM"]])
    m = sm.OLS(j.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    return m
