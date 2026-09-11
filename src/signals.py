"""Signal library.

Each family is a published specification rather than something invented here, so the
search space is fixed in advance instead of being discovered while looking at results.

  TSMOM               Moskowitz, Ooi and Pedersen (2012), JFE 104:228
  MA crossover        Brock, Lakonishok and LeBaron (1992); Han, Yang and Zhou (2013)
  Donchian breakout   the classical futures channel rule
  Short-term reversal Jegadeesh (1990); Lehmann (1990)
  XS FX momentum      Menkhoff, Sarno, Schmeling and Schrimpf (2012), JFE 106:660
  FX carry            Lustig, Roussanov and Verdelhan (2011); Koijen et al. (2018)
  FX value            Asness, Moskowitz and Pedersen (2013), JF 68:929
  Open interest       Hong and Yogo (2012), JFE 105:473
  Volatility scaling  Moreira and Muir (2017), JF 72:1611
  Regime conditioning the rule examined in part 2

A signal takes the panel and returns target weights indexed by date with pairs as
columns, using only information available at the close of that date. evaluate applies the
shift. The lookahead test in validate_harness.py is what proves that no function here
references a future row.

Gross exposure is normalised to 1.0 for every family so that returns, turnover and costs
are comparable across families. Volatility scaling deliberately departs from this and is
reported separately."""
import numpy as np, pandas as pd
from data import PAIRS


def _norm(W):
    """scale each row to gross exposure 1; all-zero rows stay zero"""
    g = W.abs().sum(axis=1)
    return W.div(g.where(g > 0, np.nan), axis=0).fillna(0.0)


def _xs_pick(score, k, direction=1):
    """long the top k, short the bottom k, of a cross-sectional score"""
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    valid = score.notna().sum(axis=1) >= 2 * k
    rk = score.rank(axis=1, ascending=False, method="first")
    n = score.notna().sum(axis=1)
    W[rk.le(k)] = 1.0
    W[rk.gt(n - k, axis=0)] = -1.0
    W = W.where(valid, 0.0)
    return _norm(W * direction)


# ------------------------------------------------------------------ time series
def tsmom(P, L):
    c = P["close"]
    s = np.sign(c / c.shift(L) - 1.0)
    return _norm(s.where(c.shift(L).notna(), 0.0))


def ma_cross(P, fast, slow):
    c = P["close"]
    s = np.sign(c.rolling(fast).mean() - c.rolling(slow).mean())
    return _norm(s.fillna(0.0))


def donchian(P, L):
    c, h, l = P["close"], P["high"], P["low"]
    up = h.rolling(L).max().shift(1)
    dn = l.rolling(L).min().shift(1)
    s = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    s[c >= up] = 1.0
    s[c <= dn] = -1.0
    return _norm(s.ffill().fillna(0.0))


def ts_reversal(P, L):
    c = P["close"]
    s = -np.sign(c / c.shift(L) - 1.0)
    return _norm(s.where(c.shift(L).notna(), 0.0))


# ------------------------------------------------------------------ cross section
def xs_momentum(P, L, k):
    c = P["close"]
    return _xs_pick(c / c.shift(L) - 1.0, k, +1)


def xs_reversal(P, L, k):
    c = P["close"]
    return _xs_pick(c / c.shift(L) - 1.0, k, -1)


def carry(P, k):
    """long the high interest-rate currencies, short the low ones"""
    rt = P["rates"]
    diff = pd.DataFrame({p: rt[p] - rt["USD"] for p in PAIRS})
    return _xs_pick(diff, k, +1)


def value(P, k, L=1260):
    """five-year real-exchange-rate reversal, the AMP (2013) FX value proxy"""
    c = P["close"]
    return _xs_pick(c / c.shift(L) - 1.0, k, -1)


def oi_growth(P, L, k):
    """Hong & Yogo (2012): open-interest growth as a predictor of futures returns"""
    oi = P["oi"].replace(0, np.nan)
    return _xs_pick(oi / oi.shift(L) - 1.0, k, +1)


def volume_trend(P, L, k):
    v = P["vol"].replace(0, np.nan)
    return _xs_pick(v.rolling(L).mean() / v.rolling(4 * L).mean() - 1.0, k, +1)


# ------------------------------------------------------------------ overlays
def regime_overlay(W, P, L):
    """the authors' specification: flip the book on the sign of DXY momentum"""
    d = P["dx_close"]
    mom = d / d.shift(L) - 1.0
    sgn = pd.Series(np.where(mom < 0, 1.0, -1.0), index=W.index).where(mom.notna(), 0.0)
    return W.mul(sgn, axis=0)


def vol_target(W, P, target=0.10, L=63, cap=3.0):
    """Moreira & Muir (2017) scaling. Deliberately leaves gross exposure floating."""
    pr = (W.shift(1) * P["ret"]).sum(axis=1)
    rv = pr.rolling(L).std() * np.sqrt(252)
    lev = (target / rv.replace(0, np.nan)).clip(upper=cap).shift(1)
    return W.mul(lev, axis=0).fillna(0.0)


def rebalance(W, freq):
    """hold the weights between rebalance dates; 'D' means refresh every session"""
    if freq == "D":
        return W
    if freq == "S":
        key = pd.Index([f"{d.year}-S{1 if d.quarter <= 2 else 2}" for d in W.index])
    else:
        key = W.index.to_period(freq).astype(str)
    mark = pd.Series(key, index=W.index)
    keep = mark != mark.shift(1)
    return W.where(keep, np.nan).ffill().fillna(0.0)


# ------------------------------------------------------------------ registry
LOOKBACKS = [21, 63, 126, 252]
CARDS = [1, 2]
FREQS = ["D", "W", "M", "Q"]


def base_families():
    """(family, label, callable), the pre-specified base signals, before overlays"""
    F = []
    for L in LOOKBACKS:
        F.append(("TSMOM", "tsmom%d" % L, lambda P, L=L: tsmom(P, L)))
        F.append(("REVERSAL", "rev%d" % L, lambda P, L=L: ts_reversal(P, L)))
        for k in CARDS:
            F.append(("XSMOM", "xsmom%d_k%d" % (L, k), lambda P, L=L, k=k: xs_momentum(P, L, k)))
            F.append(("XSREV", "xsrev%d_k%d" % (L, k), lambda P, L=L, k=k: xs_reversal(P, L, k)))
    for f, s in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        F.append(("MACROSS", "ma%d_%d" % (f, s), lambda P, f=f, s=s: ma_cross(P, f, s)))
    for L in [20, 50, 100]:
        F.append(("BREAKOUT", "don%d" % L, lambda P, L=L: donchian(P, L)))
    for k in CARDS:
        F.append(("CARRY", "carry_k%d" % k, lambda P, k=k: carry(P, k)))
        F.append(("VALUE", "value_k%d" % k, lambda P, k=k: value(P, k)))
        for L in [21, 63]:
            F.append(("OI", "oi%d_k%d" % (L, k), lambda P, L=L, k=k: oi_growth(P, L, k)))
        F.append(("VOLUME", "volume_k%d" % k, lambda P, k=k: volume_trend(P, 21, k)))
    return F


def enumerate_configs(with_regime=True, with_voltarget=True):
    """The complete, declared search space. Every element is (key, builder)."""
    cfgs = []
    for fam, lab, fn in base_families():
        for freq in FREQS:
            overlays = [("none", None)]
            if with_regime:
                overlays += [("reg%d" % L, L) for L in [63, 126, 252]]
            for oname, oL in overlays:
                for vt in ([False, True] if with_voltarget else [False]):
                    key = (fam, lab, freq, oname, "vt" if vt else "raw")

                    def build(P, fn=fn, freq=freq, oL=oL, vt=vt):
                        W = fn(P)
                        if oL is not None:
                            W = regime_overlay(W, P, oL)
                        W = rebalance(W, freq)
                        if vt:
                            W = vol_target(W, P)
                        return W
                    cfgs.append((key, build))
    return cfgs
