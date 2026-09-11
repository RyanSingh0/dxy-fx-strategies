"""Mechanism check on carry, after the dispersion hypothesis failed.

The dispersion hypothesis does not hold. Regressing carry returns on lagged dispersion
gives a slope of -0.000007 (t = -0.19, p = 0.85). Sorting by dispersion puts the high
tercile last at Sharpe 0.016 against 0.471 and 0.541. Dispersion also barely moved across
the sample, 1.63% to 2.28% annualised, so the account in which ZIRP compressed the spread
does not survive either. The small edge of E3 over E2 is not that mechanism and is not
claimed as one.

That leaves carry itself. The mechanism has to be verified rather than assumed. A
cross-sectional carry return in futures decomposes as

      realised return = interest differential earned + spot move against the position

The premise of the carry trade is that the second term does not fully offset the first,
which is the forward premium puzzle. If it does offset in this sample then any positive
number is luck and there is nothing to trade.

Also tested is dollar carry (Lustig, Roussanov and Verdelhan 2014), the average foreign
minus US rate traded as a single long/short of the whole basket. It is a documented factor
distinct from the cross-sectional carry trade and one this small universe can support."""
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
from carry_dispersion import carry_linear

TD = 252
PAIRS = D.PAIRS


def main():
    P = D.panel()
    C = roll_carry(P)
    ret = P["ret"]

    print("=" * 108)
    print("1.  MECHANISM, is the carry actually being EARNED, or given back in spot?")
    print("=" * 108)
    print("   For each currency: the implied carry it offers, the spot return it delivered,")
    print("   and the sum. Under uncovered interest parity the two would cancel exactly.")
    print("   %-9s %14s %14s %14s %10s" %
          ("pair", "carry %/yr", "spot %/yr", "total %/yr", "UIP holds?"))
    tot = {}
    for p in PAIRS:
        cy = 400 * C[p].mean()
        sp = TD * 100 * ret[p].mean()
        tot[p] = cy + sp
        print("   %-9s %+13.2f%% %+13.2f%% %+13.2f%% %10s"
              % (p, cy, sp, cy + sp, "no" if abs(cy + sp) > 0.5 else "roughly"))
    print("\n   Cross-sectionally: does a HIGHER offered carry come with a WORSE spot move")
    print("   that cancels it (UIP), or not (the forward premium puzzle)?")
    cys = np.array([400 * C[p].mean() for p in PAIRS])
    sps = np.array([TD * 100 * ret[p].mean() for p in PAIRS])
    b, a, r_, pv, se = st.linregress(cys, sps)
    print("      regression of spot return on offered carry across the five currencies:")
    print("         slope %+.3f (UIP predicts -1.00, the puzzle predicts around 0)  R2 %.3f  p %.3f"
          % (b, r_ ** 2, pv))
    print("      -> a slope near zero means the carry is kept, not competed away.")

    print("\n   Panel version, using time variation rather than five points:")
    Cl = C.sub(C.mean(axis=1), axis=0)
    fwd = ret.rolling(63).sum().shift(-63)
    rows = []
    for p in PAIRS:
        j = pd.concat([Cl[p].rename("c"), fwd[p].rename("f")], axis=1).dropna()
        rows.append(j.assign(pair=p))
    pan = pd.concat(rows)
    m = sm.OLS(pan.f.values, sm.add_constant(pan.c.values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 63})
    print("      next-quarter spot return on relative carry: slope %+.3f (t %+.2f, p %.4f)"
          % (m.params[1], m.tvalues[1], m.pvalues[1]))
    print("      -> UIP would give -1.0 here. A slope above -1 is the tradeable premium.")

    print("\n" + "=" * 108)
    print("2.  DOLLAR CARRY, Lustig, Roussanov & Verdelhan (2014)")
    print("=" * 108)
    avg = C.mean(axis=1)
    print("   average foreign-minus-US implied rate: mean %+.2f%%/yr, sd %.2f%%, "
          "%% of days foreign > US: %.0f%%"
          % (400 * avg.mean(), 400 * avg.std(), 100 * (avg > 0).mean()))
    out = {}

    def add(nm, W, cost=1.0, series=None):
        if series is None:
            g, n, t, ge = E.apply_positions(W, P, cost)
            s = E.summary(n, t); tv = s["TurnoverPA"]
        else:
            n = series; s = E.summary(n); tv = np.nan
        out[nm] = n
        print("   %-46s %8.3f %9.1f %8.2f %8.4f %8.1f %8s"
              % (nm, s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], 100 * s["MaxDD"],
                 "%.1f" % tv if np.isfinite(tv) else "-"))
        return n

    print("\n   %-46s %8s %9s %8s %8s %8s %8s"
          % ("strategy", "Sharpe", "TotRet%", "t", "p", "MaxDD%", "turn/yr"))
    sgn = np.sign(avg).shift(1).fillna(0.0)
    Wdc = pd.DataFrame({p: sgn / len(PAIRS) for p in PAIRS})
    add("F1 dollar carry (sign of average carry)", Wdc)
    Wdcs = pd.DataFrame({p: (avg / avg.abs().rolling(252).mean()).clip(-2, 2).shift(1).fillna(0) / len(PAIRS)
                         for p in PAIRS})
    add("F2 dollar carry, proportional sizing", Wdcs)
    add("F3 cross-sectional carry, smoothed", carry_linear(P, C.rolling(252).mean()))
    add("F4 cross-sectional carry, raw", carry_linear(P, C))

    # the combination of the two documented carry factors, equal risk, no dollar-drift leg
    xs = out["F3 cross-sectional carry, smoothed"]
    dc = out["F1 dollar carry (sign of average carry)"]
    comb = 0.5 * (xs / xs.std() + dc / dc.std())
    comb = comb / comb.std() * xs.std()
    add("F5 cross-sectional + dollar carry, equal risk", None, series=comb)

    print("\n" + "=" * 108)
    print("3.  HOLDOUT")
    print("=" * 108)
    print("   %-46s %10s %10s %9s %10s" % ("strategy", "IS Sharpe", "OOS Sharpe", "OOS p", "OOS ret%"))
    for nm, r in out.items():
        yr = r.index.year
        a_, b_ = r[yr <= 2018], r[yr > 2018]
        sb = E.summary(b_)
        print("   %-46s %10.3f %10.3f %9.4f %10.1f"
              % (nm, E.sharpe(a_), E.sharpe(b_), sb["p"], 100 * sb["TotRet"]))

    print("\n" + "=" * 108)
    print("4.  COST TOLERANCE OF THE SURVIVORS")
    print("=" * 108)
    for nm, W in [("F3 cross-sectional carry, smoothed", carry_linear(P, C.rolling(252).mean())),
                  ("F1 dollar carry", Wdc)]:
        print("   %s" % nm)
        for c in (0, 1, 2, 5, 10, 25, 50):
            g, n, t, _ = E.apply_positions(W, P, c)
            print("      %4.0f bp/unit turnover: Sharpe %+6.3f  TotRet %+7.1f%%" % (c, E.sharpe(n), 100 * ((1 + n).prod() - 1)))

    print("\n" + "=" * 108)
    print("5.  NOISE CEILING, matched to this six-strategy family")
    print("=" * 108)
    n_, m_ = ret.shape
    ceil = []
    for seed in range(10):
        rng = np.random.default_rng(400 + seed)
        idx = E.stationary_bootstrap_idx(n_, 1, rng, mean_block=21)[0]
        Q = dict(P)
        Q["ret"] = pd.DataFrame(ret.values[idx], index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod()
        Q["roll"] = P["roll"]
        Q["dx_ret"] = -Q["ret"].mean(axis=1)
        Cq = roll_carry(Q)
        aq = Cq.mean(axis=1); sq = np.sign(aq).shift(1).fillna(0.0)
        cands = [xs_rank_weights(Cq, 2), carry_linear(Q, Cq), carry_linear(Q, Cq.rolling(252).mean()),
                 pd.DataFrame({p: sq / len(PAIRS) for p in PAIRS}),
                 pd.DataFrame({p: (aq / aq.abs().rolling(252).mean()).clip(-2, 2).shift(1).fillna(0) / len(PAIRS) for p in PAIRS})]
        srs = []
        for Wk in cands:
            _, nn, _, _ = E.apply_positions(Wk, Q, 1.0)
            srs.append(E.sharpe(nn))
        ceil.append(max(srs))
    print("   best Sharpe on block-bootstrapped returns, %d simulations:" % len(ceil))
    print("      " + "  ".join("%+.2f" % c for c in ceil))
    print("   ceiling: mean %+.3f | 90th percentile %+.3f | max %+.3f"
          % (np.mean(ceil), np.percentile(ceil, 90), max(ceil)))
    best = max(out, key=lambda k: E.sharpe(out[k]))
    print("   best observed on real data: %+.3f (%s)" % (E.sharpe(out[best]), best))
    print("   -> %s" % ("ABOVE the ceiling" if E.sharpe(out[best]) > np.percentile(ceil, 90)
                        else "within the range noise produces"))

    pd.DataFrame(out).to_pickle(str(paths.RESULTS)+"/part4_carry_mechanism.pkl")


if __name__ == "__main__":
    main()
