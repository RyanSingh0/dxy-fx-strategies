"""Fast, numerically equivalent re-implementation of the 1,312-configuration search.

The original harness (signals.enumerate_configs + evaluate.apply_positions) rebuilds every
base signal for every one of its 32 overlay/frequency/volatility-target variants. Here each
of the 41 base signals is built once with the *same* functions in signals.py, and the
regime overlay, rebalancing, volatility targeting and the one-period shift are applied in
NumPy. check_equivalence() verifies, on the real data, that every one of the 1,312 net
return series matches the original harness."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd
import data as D, evaluate as E, signals as S

TD = 252
OVERLAYS = [("none", None), ("reg63", 63), ("reg126", 126), ("reg252", 252)]


def _keep_index(index, freq):
    if freq == "D":
        return np.arange(len(index))
    if freq == "S":
        key = pd.Index([f"{d.year}-S{1 if d.quarter <= 2 else 2}" for d in index])
    else:
        key = index.to_period(freq).astype(str)
    mark = pd.Series(key, index=index)
    keep = (mark != mark.shift(1)).values
    pos = np.where(keep, np.arange(len(index)), 0)
    return np.maximum.accumulate(pos)


def _regime_sign(P, L):
    d = P["dx_close"]
    mom = d / d.shift(L) - 1.0
    sgn = np.where(mom < 0, 1.0, -1.0)
    sgn[mom.isna().values] = 0.0
    return sgn


def _rolling_std_ddof1(x, L):
    s = pd.Series(x).rolling(L).std().values
    return s


def evaluate_all(P, cost_bp=1.0):
    """Returns (keys, net matrix n x K) in the same order as signals.enumerate_configs()."""
    ret = P["ret"].values
    n, m = ret.shape
    idx = P["ret"].index
    keepix = {f: _keep_index(idx, f) for f in S.FREQS}
    sg = {L: _regime_sign(P, L) for _, L in OVERLAYS if L is not None}
    keys, nets = [], []
    c = cost_bp / 1e4
    for fam, lab, fn in S.base_families():
        W0 = fn(P).reindex(idx).values.astype(float)
        for freq in S.FREQS:
            for oname, oL in OVERLAYS:
                Wo = W0 if oL is None else W0 * sg[oL][:, None]
                Wr = Wo[keepix[freq]]
                for vt in (False, True):
                    if vt:
                        Wl = np.vstack([np.zeros((1, m)), Wr[:-1]])
                        pr = (Wl * ret).sum(axis=1)
                        rv = _rolling_std_ddof1(pr, 63) * np.sqrt(252)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            lev = 0.10 / np.where(rv == 0, np.nan, rv)
                        lev = np.minimum(lev, 3.0)
                        lev = np.concatenate([[np.nan], lev[:-1]])
                        W = Wr * lev[:, None]
                        W = np.where(np.isnan(W), 0.0, W)
                    else:
                        W = Wr
                    Wl = np.vstack([np.zeros((1, m)), W[:-1]])
                    gross = (Wl * ret).sum(axis=1)
                    turn = np.abs(W - Wl).sum(axis=1)
                    turn[0] = 0.0
                    net = gross - turn * c
                    keys.append((fam, lab, freq, oname, "vt" if vt else "raw"))
                    nets.append(net)
    return keys, np.column_stack(nets)


def sharpes(M):
    mu = M.mean(axis=0); sd = M.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = mu / sd * np.sqrt(TD)
    s[sd == 0] = np.nan
    return s


def bootstrap_panel(P, seed, mean_block=21, gaussian=False):
    """The resampled panel used for the noise ceiling, exactly as in verify_search.py (V2)
    and run_search.py (Gaussian null)."""
    ret = P["ret"]
    n, m = ret.shape
    if gaussian:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(np.corrcoef(ret.values.T) + 1e-9 * np.eye(m))
        sd = ret.std(ddof=0).values
        z = (rng.standard_normal((n, m)) @ L.T) * sd
    else:
        rng = np.random.default_rng(seed)
        ix = E.stationary_bootstrap_idx(n, 1, rng, mean_block=mean_block)[0]
        z = ret.values[ix]
    Q = dict(P)
    Q["ret"] = pd.DataFrame(z, index=ret.index, columns=ret.columns)
    Q["close"] = 100 * (1 + Q["ret"]).cumprod()
    Q["high"] = Q["close"] * 1.002; Q["low"] = Q["close"] * 0.998
    Q["dx_ret"] = -Q["ret"].mean(axis=1)
    Q["dx_close"] = 100 * (1 + Q["dx_ret"]).cumprod()
    return Q


def check_equivalence():
    import time
    P = D.panel()
    t0 = time.time(); keys, M = evaluate_all(P); t1 = time.time()
    print("fast build: %d configs in %.1fs" % (len(keys), t1 - t0))
    cfgs = S.enumerate_configs()
    worst = 0.0; worst_sr = 0.0
    for j, (key, build) in enumerate(cfgs):
        assert key == keys[j]
        W = build(P)
        g, nt, tu, ge = E.apply_positions(W, P, 1.0)
        d = float(np.abs(nt.values - M[:, j]).max())
        worst = max(worst, d)
        worst_sr = max(worst_sr, abs(E.sharpe(nt) - sharpes(M[:, [j]])[0]))
    print("original build: %.1fs" % (time.time() - t1))
    print("max abs difference in daily net returns over all configs: %.3e" % worst)
    print("max abs difference in Sharpe ratio: %.3e" % worst_sr)
    s = sharpes(M)
    print("best Sharpe %.6f  (paper: 0.653)" % np.nanmax(s))
    # bootstrap replicate 0 must reproduce verify_search V2 bootstrap 0 (+0.753)
    Q = bootstrap_panel(P, 9000)
    _, Mq = evaluate_all(Q)
    print("bootstrap seed 9000 best Sharpe %.3f (paper/V2: +0.753)" % np.nanmax(sharpes(Mq)))


if __name__ == "__main__":
    check_equivalence()
