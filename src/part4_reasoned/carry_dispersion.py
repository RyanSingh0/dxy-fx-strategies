"""Conditioning carry on the cross-sectional dispersion of carry.

Term-structure carry earned Sharpe 0.19 over 2011 to 2018 and 0.62 to 0.69 over 2019 to
2025, which is not a random split. Carry compensates for taking interest-rate risk, so it
can only pay when there is a differential to harvest. From 2011 to 2021 every G10 policy
rate sat at or below zero and the cross-sectional dispersion of carry was near zero. From
2022 the Fed, ECB, BoE and BoJ diverged sharply.

The hypothesis, stated before testing: carry returns scale with the dispersion of carry,
so the book should be larger when the spread on offer is wide and smaller when it is
compressed. This is the same logic as the value spread predicting value returns (Cohen,
Polk and Vuolteenaho 2003). The conditioning variable is the strategy's own expected
payoff rather than a free parameter fitted to the data.

  1. whether dispersion is related to subsequent carry returns at all, independent of any
     strategy
  2. dispersion-scaled carry against plain carry
  3. linear carry weights against rank-based weights, since with five currencies ranks
     discard a lot
  4. carry combined with the dollar factor at risk parity
  5. all of it on the untouched holdout and against a noise ceiling built for this small
     set of strategies rather than the 1,312-configuration one"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import statsmodels.api as sm
import data as D, evaluate as E
from carry_futures import roll_carry, _norm, xs_rank_weights

TD = 252
PAIRS = D.PAIRS


def carry_linear(P, C=None):
    """cross-sectionally demeaned carry, scaled to gross exposure 1"""
    C = roll_carry(P) if C is None else C
    d = C.sub(C.mean(axis=1), axis=0)
    return _norm(d)


def dispersion(P, C=None):
    C = roll_carry(P) if C is None else C
    return C.std(axis=1)


def main():
    P = D.panel()
    C = roll_carry(P)
    disp = dispersion(P, C)
    ann_disp = 400 * disp

    print("=" * 108)
    print("1.  DOES CARRY DISPERSION ACTUALLY PREDICT CARRY RETURNS?")
    print("=" * 108)
    print("   cross-sectional dispersion of implied carry, annualised:")
    for a, b in [(2011, 2014), (2015, 2018), (2019, 2021), (2022, 2025)]:
        m = (ann_disp.index.year >= a) & (ann_disp.index.year <= b)
        print("      %d-%d  mean %.2f%%  (min %.2f%%, max %.2f%%)"
              % (a, b, ann_disp[m].mean(), ann_disp[m].min(), ann_disp[m].max()))

    Wl = carry_linear(P, C)
    _, base, turn, _ = E.apply_positions(Wl, P, 1.0)
    q = ann_disp.shift(1)
    terc = pd.qcut(q.rank(method="first"), 3, labels=["low", "mid", "high"])
    print("\n   carry-strategy return, sorted by the dispersion available that day:")
    print("      %-8s %10s %10s %10s %8s" % ("tercile", "mean disp", "return/yr", "Sharpe", "n"))
    for t_ in ["low", "mid", "high"]:
        m = (terc == t_).values
        r = base[m]
        print("      %-8s %9.2f%% %9.2f%% %10.3f %8d"
              % (t_, ann_disp[m].mean(), 100 * r.mean() * TD, E.sharpe(r), m.sum()))
    j = pd.concat([base.rename("r"), q.rename("d")], axis=1).dropna()
    m = sm.OLS(j.r.values, sm.add_constant(j.d.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
    print("   regression of daily carry return on lagged dispersion:")
    print("      slope %+.6f  t = %+.2f  p = %.4f   (positive slope = the hypothesis holds)"
          % (m.params[1], m.tvalues[1], m.pvalues[1]))

    print("\n" + "=" * 108)
    print("2.  STRATEGIES")
    print("=" * 108)
    print("   %-46s %8s %9s %8s %8s %8s %8s"
          % ("strategy", "Sharpe", "TotRet%", "t", "p", "MaxDD%", "turn/yr"))
    out = {}

    def add(nm, W, cost=1.0):
        g, n, t, ge = E.apply_positions(W, P, cost)
        s = E.summary(n, t)
        out[nm] = n
        print("   %-46s %8.3f %9.1f %8.2f %8.4f %8.1f %8.1f"
              % (nm, s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], 100 * s["MaxDD"], s["TurnoverPA"]))
        return n

    add("E1 carry, rank top/bottom-2", xs_rank_weights(C, 2))
    add("E2 carry, linear demeaned weights", Wl)
    # dispersion scaling: normalise by the expanding median so it is causal
    med = ann_disp.expanding(min_periods=500).median()
    scale = (ann_disp / med.replace(0, np.nan)).clip(0.0, 3.0).shift(1).fillna(0.0)
    add("E3 carry linear x dispersion scaling", Wl.mul(scale, axis=0))
    # smoothed carry: average of the last four rolls, i.e. one year of implied differentials
    Cs = C.rolling(252).mean()
    add("E4 carry linear, one-year smoothed signal", carry_linear(P, Cs))
    add("E5 carry smoothed x dispersion scaling", carry_linear(P, Cs).mul(scale, axis=0))

    # dollar factor and the risk-parity combination
    dxr = P["dx_ret"]
    fvd = dxr.rolling(21).std() * np.sqrt(TD)
    lev = (0.06 / fvd.replace(0, np.nan)).clip(upper=2.0).shift(1)
    dol = (lev.shift(1) * dxr).fillna(0.0)
    out["E6 vol-managed dollar"] = dol
    s = E.summary(dol)
    print("   %-46s %8.3f %9.1f %8.2f %8.4f %8.1f %8s"
          % ("E6 vol-managed dollar", s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], 100 * s["MaxDD"], "-"))

    for nm, cr in [("E7 carry + dollar, equal risk", out["E2 carry, linear demeaned weights"]),
                   ("E8 carry(disp) + dollar, equal risk", out["E3 carry linear x dispersion scaling"])]:
        a = cr / cr.std()
        b = dol / dol.std()
        comb = 0.5 * (a + b)
        comb = comb / comb.std() * cr.std()
        out[nm] = comb
        s = E.summary(comb)
        print("   %-46s %8.3f %9.1f %8.2f %8.4f %8.1f %8s"
              % (nm, s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], 100 * s["MaxDD"], "-"))

    print("\n" + "=" * 108)
    print("3.  HOLDOUT, 2019-2025 was never used to choose anything")
    print("=" * 108)
    print("   %-46s %10s %10s %9s %10s" % ("strategy", "IS Sharpe", "OOS Sharpe", "OOS p", "OOS ret%"))
    for nm, r in out.items():
        yr = r.index.year
        a, b = r[yr <= 2018], r[yr > 2018]
        sb = E.summary(b)
        print("   %-46s %10.3f %10.3f %9.4f %10.1f"
              % (nm, E.sharpe(a), E.sharpe(b), sb["p"], 100 * sb["TotRet"]))
    bm = dxr[dxr.index.year > 2018]
    print("   %-46s %10s %10.3f %9s %10.1f" % ("benchmark: long dollar", "-", E.sharpe(bm), "-",
                                               100 * ((1 + bm).prod() - 1)))

    print("\n" + "=" * 108)
    print("4.  YEAR BY YEAR, the best carry variant against the dollar")
    print("=" * 108)
    bestnm = max(out, key=lambda k: E.sharpe(out[k]))
    r = out[bestnm]
    print("   %s" % bestnm)
    yb = pd.DataFrame({"strat": r, "dollar": dxr.reindex(r.index).fillna(0),
                       "disp": ann_disp.reindex(r.index)}).groupby(r.index.year).agg(
        strat=("strat", lambda x: 100 * ((1 + x).prod() - 1)),
        dollar=("dollar", lambda x: 100 * ((1 + x).prod() - 1)),
        disp=("disp", "mean"))
    print(yb.round(2).to_string())

    print("\n" + "=" * 108)
    print("5.  NOISE CEILING FOR THIS SMALL SET (8 strategies, not 1,312)")
    print("=" * 108)
    ret = P["ret"]
    n, m_ = ret.shape
    ceil = []
    for seed in range(8):
        rng = np.random.default_rng(700 + seed)
        idx = E.stationary_bootstrap_idx(n, 1, rng, mean_block=21)[0]
        Q = dict(P)
        Q["ret"] = pd.DataFrame(ret.values[idx], index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod()
        Q["roll"] = P["roll"]
        Q["dx_ret"] = -Q["ret"].mean(axis=1)
        Cq = roll_carry(Q)
        Wq = carry_linear(Q, Cq)
        dq = 400 * Cq.std(axis=1)
        sq = (dq / dq.expanding(min_periods=500).median().replace(0, np.nan)).clip(0, 3).shift(1).fillna(0)
        cands = [xs_rank_weights(Cq, 2), Wq, Wq.mul(sq, axis=0),
                 carry_linear(Q, Cq.rolling(252).mean())]
        srs = []
        for Wk in cands:
            _, nn, _, _ = E.apply_positions(Wk, Q, 1.0)
            srs.append(E.sharpe(nn))
        ceil.append(max(srs))
    print("   best Sharpe over the carry variants on block-bootstrapped returns:")
    print("      " + "  ".join("%+.3f" % c for c in ceil))
    print("   -> noise ceiling for this family: mean %+.3f (max %+.3f)" % (np.mean(ceil), max(ceil)))
    print("   -> best observed on real data: %+.3f (%s)" % (E.sharpe(out[bestnm]), bestnm))

    pd.DataFrame(out).to_pickle(str(paths.RESULTS)+"/part4_carry_dispersion.pkl")


if __name__ == "__main__":
    main()
