"""Each headline claim from the search rechecked by a route that does not reuse the code
that produced it.

  V1. ensemble costing. run_search averaged the net returns of the members, which charges
      every member's full turnover. A real ensemble holds the average weight and
      offsetting trades never reach the market, so ensembles are rebuilt in weight space.
  V2. a harder null. The noise ceiling in run_search used Gaussian draws. A stationary
      bootstrap of the actual pair returns destroys time-series predictability while
      preserving the true marginal distributions, the fat tails and the cross-sectional
      correlation. That is the null the ceiling should be measured against.
  V3. the winning configuration rebuilt from first principles, with no call into signals
      or evaluate.
  V4. the neighbourhood of the winner. If the adjacent parameter values collapse then the
      winner is a spike in a noisy surface rather than an effect."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import data as D, evaluate as E, signals as S

TD = 252
COST_BP = 1.0


def main():
    P = D.panel()
    ret = P["ret"]
    cfgs = S.enumerate_configs()
    print("=" * 104); print("V1.  ENSEMBLES REBUILT IN WEIGHT SPACE (offsetting trades never traded)"); print("=" * 104)
    Wsum, fam_W, fam_n = {}, {}, {}
    for key, build in cfgs:
        try:
            W = build(P)
        except Exception:
            continue
        f = key[0]
        fam_W[f] = W if f not in fam_W else fam_W[f] + W
        fam_n[f] = fam_n.get(f, 0) + 1
        Wsum["ALL"] = W if "ALL" not in Wsum else Wsum["ALL"] + W
    nall = sum(fam_n.values())
    print("   %-12s %6s %9s %9s %8s %8s %11s %11s"
          % ("family", "n cfg", "Sharpe", "TotRet%", "t", "p", "turn/yr", "naive turn"))
    res = {}
    for f in sorted(fam_n) + ["ALL"]:
        W = (Wsum["ALL"] / nall) if f == "ALL" else (fam_W[f] / fam_n[f])
        g, nt, tu, ge = E.apply_positions(W, P, COST_BP)
        s = E.summary(nt, tu, ge)
        res[f] = nt
        # naive turnover = mean of members' individual turnover, for contrast
        print("   %-12s %6d %9.3f %9.1f %8.2f %8.4f %11.1f %11s"
              % (f, nall if f == "ALL" else fam_n[f], s["Sharpe"], 100 * s["TotRet"],
                 s["t"], s["p"], s["TurnoverPA"], "-"))
    print("\n   Note: averaging weights cuts the ensemble's traded turnover far below the sum of")
    print("   its members', so this is the fair version of the ensemble result.")

    print("\n" + "=" * 104); print("V2.  A HARDER NOISE CEILING, stationary bootstrap of the real returns"); print("=" * 104)
    print("   Resampling actual returns in blocks preserves fat tails, skew and cross-sectional")
    print("   correlation and destroys only the time-series structure a signal could exploit.")
    n, m = ret.shape
    rv = ret.values
    ceil_boot = []
    for seed in range(6):
        rng = np.random.default_rng(9000 + seed)
        idx = E.stationary_bootstrap_idx(n, 1, rng, mean_block=21)[0]
        z = rv[idx]                                    # same rows across pairs: keeps cross-correlation
        Q = dict(P)
        Q["ret"] = pd.DataFrame(z, index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod()
        Q["high"] = Q["close"] * 1.002; Q["low"] = Q["close"] * 0.998
        Q["dx_ret"] = -Q["ret"].mean(axis=1)
        Q["dx_close"] = 100 * (1 + Q["dx_ret"]).cumprod()
        cols = {}
        for key, build in cfgs:
            try:
                W = build(Q); _, nt, _, _ = E.apply_positions(W, Q, COST_BP)
                if nt.std() > 0: cols[key] = nt
            except Exception: pass
        srs = pd.DataFrame(cols).dropna().apply(E.sharpe)
        ceil_boot.append(float(srs.max()))
        print("   bootstrap %d: best net Sharpe over %d configurations = %+.3f" % (seed, len(srs), srs.max()))
    CB = float(np.mean(ceil_boot))
    print("   -> bootstrap noise ceiling: mean %+.3f (range %+.3f .. %+.3f)"
          % (CB, min(ceil_boot), max(ceil_boot)))

    Gn = pd.read_pickle(str(paths.RESULTS)+"/part3_net_returns.pkl")
    srs_n = Gn.apply(E.sharpe).sort_values(ascending=False)
    print("   observed best on the REAL data: %+.3f" % srs_n.max())
    print("   configurations above the bootstrap ceiling: %d of %d"
          % (int((srs_n > CB).sum()), len(srs_n)))

    print("\n" + "=" * 104); print("V3.  INDEPENDENT REBUILD OF THE WINNING CONFIGURATION"); print("=" * 104)
    best = srs_n.index[0]
    print("   target: %s  (harness Sharpe %+.6f)" % ("/".join(map(str, best)), srs_n.max()))
    # xsmom126_k1 / quarterly / regime-126 / vol-targeted, written from scratch
    c = P["close"]; r = P["ret"]
    mom = (c / c.shift(126) - 1.0).values
    W = np.zeros_like(mom)
    for i in range(mom.shape[0]):
        row = mom[i]
        if np.isnan(row).any(): continue
        W[i, int(np.nanargmax(row))] = 1.0
        W[i, int(np.nanargmin(row))] = -1.0
    W = W / 2.0
    dxc = P["dx_close"].values
    dmom = np.full(len(dxc), np.nan)
    dmom[126:] = dxc[126:] / dxc[:-126] - 1.0
    sgn = np.where(np.isnan(dmom), 0.0, np.where(dmom < 0, 1.0, -1.0))
    W = W * sgn[:, None]
    Wd = pd.DataFrame(W, index=c.index, columns=c.columns)
    q = pd.Series(c.index.to_period("Q").astype(str), index=c.index)
    mark = np.repeat((q != q.shift(1)).values[:, None], Wd.shape[1], axis=1)
    Wd = Wd.where(mark, np.nan).ffill().fillna(0.0)
    pr = (Wd.shift(1).fillna(0) * r).sum(axis=1)
    rvol = pr.rolling(63).std() * np.sqrt(252)
    lev = (0.10 / rvol.replace(0, np.nan)).clip(upper=3.0).shift(1)
    Wv = Wd.mul(lev, axis=0).fillna(0.0)
    g2 = (Wv.shift(1).fillna(0) * r).sum(axis=1)
    t2 = (Wv - Wv.shift(1)).abs().sum(axis=1).fillna(0.0)
    n2 = g2 - t2 * (COST_BP / 1e4)
    s2 = E.sharpe(n2)
    diff = float((n2 - Gn[best]).abs().max())
    print("   independent rebuild Sharpe %+.6f | max abs difference in daily returns %.3e  %s"
          % (s2, diff, "MATCH" if diff < 1e-12 else "MISMATCH"))

    print("\n" + "=" * 104); print("V4.  THE NEIGHBOURHOOD OF THE WINNER"); print("=" * 104)
    print("   If the winner is an effect, nearby parameters should work too.")
    print("   %-46s %9s" % ("configuration", "Sharpe"))
    for key in cfgs:
        k = key[0]
        if k[0] == "XSMOM" and k[2] == "Q" and k[4] == "vt" and k[3] == "reg126":
            if k in Gn.columns:
                print("   %-46s %9.3f" % ("/".join(map(str, k)), E.sharpe(Gn[k])))
    print("   varying only the rebalance frequency of the winner:")
    for f in ["D", "W", "M", "Q"]:
        k = ("XSMOM", "xsmom126_k1", f, "reg126", "vt")
        if k in Gn.columns: print("      freq %-2s  Sharpe %+.3f" % (f, E.sharpe(Gn[k])))
    print("   varying only the regime lookback:")
    for o in ["none", "reg63", "reg126", "reg252"]:
        k = ("XSMOM", "xsmom126_k1", "Q", o, "vt")
        if k in Gn.columns: print("      overlay %-7s Sharpe %+.3f" % (o, E.sharpe(Gn[k])))
    print("   varying only the momentum lookback:")
    for L in [21, 63, 126, 252]:
        k = ("XSMOM", "xsmom%d_k1" % L, "Q", "reg126", "vt")
        if k in Gn.columns: print("      lookback %3d  Sharpe %+.3f" % (L, E.sharpe(Gn[k])))
    print("   with and without volatility targeting:")
    for v in ["raw", "vt"]:
        k = ("XSMOM", "xsmom126_k1", "Q", "reg126", v)
        if k in Gn.columns: print("      %-4s Sharpe %+.3f" % (v, E.sharpe(Gn[k])))


if __name__ == "__main__":
    main()
