"""Strategies built from what the structural analysis showed rather than from a grid.

From analyse_structure.py:
  1. direction is unpredictable. AR(1) R^2 on returns runs 0.0000 to 0.0011, variance
     ratios 0.83 to 1.02 and Ljung-Box on the mean pair return gives p = 0.24. That is
     why 1,312 configurations found nothing.
  2. variance is highly predictable. AR(1) R^2 on log realised volatility runs 0.966 to
     0.972, three orders of magnitude above direction.
  3. the dollar factor is 54.8% of the cross-section and the residual has no daily
     reversion, with half-lives of 158 to 726 days. Nothing to trade there at this
     frequency.
  4. DX and the five currency futures roll on the same IMM cycle 98.2% of the time, so
     the matched synthetic basket is an exact no-arbitrage relationship. The krona, of
     weight 0.042 and with no futures contract in this panel, explains 98% of the raw
     tracking error.

If direction is unpredictable then no amount of signal engineering helps. What is left is
a genuine risk premium, which does not require prediction, or a convergence relationship
that holds by construction.

  A. term-structure carry. The roll gap between the expiring and the deferred contract is
     the market's own quoted interest differential, so the futures curve is the carry and
     no external rate data is needed. Koijen, Moskowitz, Pedersen and Vrugt (2018) define
     carry this way. Holding a currency whose curve is in contango earns negative carry.
  B. the same book, volatility managed. Menkhoff, Sarno, Schmeling and Schrimpf (2012)
     show that carry returns compensate for global FX volatility risk and that carry
     crashes coincide with volatility spikes. Volatility is the one forecastable quantity
     in this data, so the carry book is scaled by forecast volatility.
  C. the basket convergence trade, measured against its true transaction cost.

Every strategy is held through the evaluate shift, costed on turnover and taken to the
same holdout and noise ceiling as everything else."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import statsmodels.api as sm
import data as D, evaluate as E

TD = 252
PAIRS = D.PAIRS
WLOG = {"EURUSD": -0.576, "JPYUSD": -0.136, "GBPUSD": -0.119,
        "CADUSD": -0.091, "CHFUSD": -0.036}


# ---------------------------------------------------------------- building blocks
def roll_carry(P):
    """Market-implied carry from the futures term structure.

    On a roll session the printed price jumps from the expiring contract to the deferred
    one. That gap is the quoted interest differential over one quarter. A positive gap
    (deferred richer, i.e. contango) means being long the currency loses on the roll, so
    carry = -gap. Held constant until the next roll. Known at the roll close, so it is
    available the following session; apply_positions supplies that lag.
    """
    close, roll = P["close"], P["roll"]
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for p in PAIRS:
        c, rl = close[p], roll[p]
        gap = np.log(c) - np.log(c.shift(1))
        out.loc[rl & (out.index > out.index[0]), p] = -gap[rl & (out.index > out.index[0])]
    return out.ffill()


def realised_vol(P, L=21):
    """forecast of tomorrow's vol from information up to today"""
    return P["ret"].rolling(L).std() * np.sqrt(TD)


def global_fx_vol(P, L=21):
    return P["ret"].mean(axis=1).rolling(L).std() * np.sqrt(TD)


def _norm(W):
    g = W.abs().sum(axis=1)
    return W.div(g.where(g > 0, np.nan), axis=0).fillna(0.0)


def xs_rank_weights(score, k):
    """long the top k, short the bottom k; gross exposure 1"""
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    n = score.notna().sum(axis=1)
    rk = score.rank(axis=1, ascending=False, method="first")
    W[rk.le(k)] = 1.0
    W[rk.gt(n - k, axis=0)] = -1.0
    return _norm(W.where(n >= 2 * k, 0.0))


# ---------------------------------------------------------------- strategies
def A_carry(P, k=2):
    """plain term-structure carry, equal risk, no conditioning"""
    return xs_rank_weights(roll_carry(P), k)


def A_carry_riskparity(P, k=2, L=63):
    """same ranking, but each leg sized by inverse volatility"""
    W = xs_rank_weights(roll_carry(P), k)
    iv = 1.0 / realised_vol(P, L).replace(0, np.nan)
    W = W * iv
    return _norm(W.fillna(0.0))


def B_carry_volmanaged(P, k=2, L=63, target=0.08, cap=2.0, vol_L=21):
    """
    Carry, scaled by forecast volatility (Moreira & Muir 2017 applied to the carry factor,
    motivated by Menkhoff et al. 2012's finding that carry is compensation for global FX
    volatility risk). Exposure falls exactly when carry crashes historically occur.
    """
    W = A_carry_riskparity(P, k, L)
    pr = (W.shift(1) * P["ret"]).sum(axis=1)
    fv = pr.rolling(vol_L).std() * np.sqrt(TD)
    lev = (target / fv.replace(0, np.nan)).clip(upper=cap).shift(1)
    return W.mul(lev, axis=0).fillna(0.0)


def B_carry_volfilter(P, k=2, L=63, q=0.80, vol_L=21):
    """binary version: stand down when global FX volatility is in its top quintile"""
    W = A_carry_riskparity(P, k, L)
    gv = global_fx_vol(P, vol_L)
    thr = gv.expanding(min_periods=500).quantile(q)
    on = (gv <= thr).astype(float).shift(1).fillna(0.0)
    return W.mul(on, axis=0)


def C_basket_convergence(P, entry=1.5, exit_=0.25, L=63, use_sek=False):
    """
    DX futures against the matched synthetic basket. Trade the spread when it is far from
    its own recent mean and flatten when it returns. The five currency legs plus DX are
    six instruments, so a round trip costs roughly six times a single trade, which is the
    thing this test exists to measure.
    """
    lc = np.log(P["close"])
    synth = sum(WLOG[p] * lc[p] for p in WLOG)
    spread = np.log(P["dx_raw"]) - synth
    if use_sek:
        fr = pd.read_csv(str(paths.FRED_DAILY), parse_dates=["Date"]).set_index("Date")
        spread = spread - 0.042 * np.log(fr["DEXSDUS"]).reindex(spread.index).ffill()
    z = (spread - spread.rolling(L).mean()) / spread.rolling(L).std()
    pos = pd.Series(np.nan, index=z.index)
    pos[z >= entry] = -1.0          # DX rich: short DX, long the basket
    pos[z <= -entry] = +1.0
    pos[z.abs() <= exit_] = 0.0
    pos = pos.ffill().fillna(0.0)
    # express as weights on the five currency futures; the DX leg is accounted separately
    W = pd.DataFrame({p: -pos * WLOG[p] for p in PAIRS}, index=z.index)
    return W, pos, z


def D_volmanaged_dollar(P, target=0.06, cap=2.0, vol_L=21):
    """the dollar itself, exposure scaled by forecast volatility"""
    dxr = P["dx_ret"]
    fv = dxr.rolling(vol_L).std() * np.sqrt(TD)
    lev = (target / fv.replace(0, np.nan)).clip(upper=cap).shift(1)
    return (lev.shift(1) * dxr).fillna(0.0), lev


# ---------------------------------------------------------------- reporting
def report(name, r, turn=None, extra=""):
    s = E.summary(r, turn)
    if not s:
        print("   %-44s (degenerate)" % name); return None
    print("   %-44s %8.3f %9.1f %8.2f %8.4f %8.1f %8.1f %s"
          % (name, s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], 100 * s["MaxDD"],
             s.get("TurnoverPA", np.nan), extra))
    return s


def main():
    P = D.panel()
    print("=" * 112)
    print("STRATEGY 5, built from the structure, not from a grid")
    print("=" * 112)

    rc = roll_carry(P)
    print("\nA.  IS THE TERM-STRUCTURE CARRY SIGNAL SANE?")
    print("   mean implied quarterly carry by currency (positive = earns carry when long):")
    for p in PAIRS:
        print("      %-9s %+7.3f%% per quarter  (%+.2f%% annualised)"
              % (p, 100 * rc[p].mean(), 400 * rc[p].mean()))
    fr = pd.read_csv(str(paths.FRED_RATES), parse_dates=["Date"]).set_index("Date")
    print("   cross-check against Federal Reserve 3-month interbank differentials:")
    for p in PAIRS:
        emp = 400 * rc[p].mean()
        act = (fr[D.RATE_COL[p]] - fr[D.US_RATE]).reindex(P["index"]).ffill().mean()
        print("      %-9s futures-implied %+6.2f%%   interbank differential %+6.2f%%   diff %+.2f"
              % (p, emp, act, emp - act))

    print("\n" + "=" * 112)
    print("RESULTS  (costs 1 bp per unit of turnover)")
    print("=" * 112)
    print("   %-44s %8s %9s %8s %8s %8s %8s"
          % ("strategy", "Sharpe", "TotRet%", "t", "p", "MaxDD%", "turn/yr"))

    out = {}
    for nm, W in [("A1 carry, equal weight, k=2", A_carry(P, 2)),
                  ("A2 carry, equal weight, k=1", A_carry(P, 1)),
                  ("A3 carry, inverse-vol legs, k=2", A_carry_riskparity(P, 2)),
                  ("B1 carry, vol-managed (target 8%)", B_carry_volmanaged(P, 2)),
                  ("B2 carry, stand down in top-quintile vol", B_carry_volfilter(P, 2))]:
        g, n, t, ge = E.apply_positions(W, P, 1.0)
        out[nm] = n
        report(nm, n, t)

    Wc, pos, z = C_basket_convergence(P)
    gc, nc, tc, _ = E.apply_positions(Wc, P, 1.0)
    dxr = P["dx_ret"]
    dx_leg = pos.shift(1).fillna(0.0) * dxr                  # the DX side of the spread
    dx_turn = (pos - pos.shift(1)).abs().fillna(0.0)
    tot_gross = gc + dx_leg
    tot_turn = tc + dx_turn
    for cbp in (0.0, 0.5, 1.0):
        net = tot_gross - tot_turn * (cbp / 1e4)
        report("C basket convergence @ %.1f bp/leg" % cbp, net, tot_turn)
    out["C basket convergence"] = tot_gross - tot_turn * 1e-4

    dvm, lev = D_volmanaged_dollar(P)
    report("D vol-managed dollar", dvm, (lev.diff().abs()).fillna(0))
    out["D vol-managed dollar"] = dvm
    report("   benchmark: long dollar, unmanaged", P["dx_ret"])

    print("\n" + "=" * 112)
    print("HOLDOUT, 2011-2018 in sample, 2019-2025 never used to choose anything")
    print("=" * 112)
    print("   %-44s %9s %9s %9s %9s" % ("strategy", "IS Sharpe", "OOS Sharpe", "OOS p", "OOS TotRet%"))
    for nm, r in out.items():
        yr = r.index.year
        a, b = r[yr <= 2018], r[yr > 2018]
        sb = E.summary(b)
        print("   %-44s %9.3f %9.3f %9.4f %9.1f"
              % (nm, E.sharpe(a), E.sharpe(b), sb["p"] if sb else np.nan,
                 100 * sb["TotRet"] if sb else np.nan))
    b = P["dx_ret"][P["dx_ret"].index.year > 2018]
    print("   %-44s %9s %9.3f %9s %9.1f" % ("   benchmark: long dollar", "-", E.sharpe(b), "-",
                                            100 * ((1 + b).prod() - 1)))

    pd.DataFrame(out).to_pickle(str(paths.RESULTS)+"/part4_carry_futures.pkl")
    print("\nsaved results/part4_carry_futures.pkl")


if __name__ == "__main__":
    main()
