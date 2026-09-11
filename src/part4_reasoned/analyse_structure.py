"""What is in this data, established before anything is built on it.

The grid search asked whether the direction of a currency can be predicted. In FX at
daily frequency the answer is essentially no. This sets out what structure does exist.

  A. is the level of returns predictable at all (autocorrelation, variance ratios)
  B. is the variance predictable, the standard contrast where returns are a martingale
     and volatility is not
  C. how much of the cross-section is one dollar factor and whether the residual reverts
  D. the index identity. DXY is an exact deterministic function of its six currencies:
       DXY = 50.14348112 * EUR^-0.576 * JPY^0.136 * GBP^-0.119 * CAD^0.091 * SEK^0.042 * CHF^0.036
     DX and the five currency futures roll on the same IMM cycle, so a matched synthetic
     basket should track DX closely and any deviation is a spread that has to converge.
     This is the only genuinely model-free relationship available here.
  E. calendar structure. Month-end hedging flows are a documented FX microstructure
     effect."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import statsmodels.api as sm
import data as D

TD = 252
W = {"EURUSD": -0.576, "JPYUSD": -0.136, "GBPUSD": -0.119,
     "CADUSD": -0.091, "CHFUSD": -0.036}          # log-DXY loading on log(XXXUSD)
SEK_W = 0.042                                      # log(USDSEK) loading; no futures contract


def main():
    P = D.panel()
    r = P["ret"]
    lr = np.log1p(r)
    print("=" * 100)
    print("A.  IS THE DIRECTION PREDICTABLE AT ALL?")
    print("=" * 100)
    print("   %-9s %8s %8s %8s %8s | %10s %10s" %
          ("pair", "ac(1)", "ac(2)", "ac(5)", "ac(21)", "VR(5)", "VR(21)"))
    for p in D.PAIRS:
        x = lr[p].dropna()
        ac = [x.autocorr(k) for k in (1, 2, 5, 21)]
        vr = []
        for q in (5, 21):
            v1 = x.var(ddof=1)
            vq = x.rolling(q).sum().dropna().var(ddof=1) / q
            vr.append(vq / v1)
        print("   %-9s %8.4f %8.4f %8.4f %8.4f | %10.3f %10.3f" % (p, *ac, *vr))
    lb = sm.stats.acorr_ljungbox(lr.mean(axis=1).dropna(), lags=[10], return_df=True)
    print("   Ljung-Box(10) on the average pair return: p = %.4f" % lb["lb_pvalue"].iloc[0])

    print("\n" + "=" * 100)
    print("B.  IS THE VARIANCE PREDICTABLE?")
    print("=" * 100)
    print("   %-9s %10s %10s %10s | %s" % ("pair", "ac(1) |r|", "ac(5) |r|", "ac(21)|r|", "Ljung-Box(10) on r^2"))
    for p in D.PAIRS:
        a = lr[p].abs().dropna()
        lb2 = sm.stats.acorr_ljungbox((lr[p].dropna()) ** 2, lags=[10], return_df=True)
        print("   %-9s %10.4f %10.4f %10.4f | p = %.3g" %
              (p, a.autocorr(1), a.autocorr(5), a.autocorr(21), lb2["lb_pvalue"].iloc[0]))
    rv = lr.std(axis=1)
    print("\n   predictability of tomorrow's realised vol from today's (R^2 of a simple AR(1) on log vol):")
    for p in D.PAIRS:
        v = np.log(lr[p].abs().rolling(21).mean().dropna() + 1e-9)
        m = sm.OLS(v.iloc[1:].values, sm.add_constant(v.iloc[:-1].values)).fit()
        print("      %-9s R2 = %.3f   (direction R2 for comparison: %.4f)" %
              (p, m.rsquared,
               sm.OLS(lr[p].dropna().iloc[1:].values,
                      sm.add_constant(lr[p].dropna().iloc[:-1].values)).fit().rsquared))

    print("\n" + "=" * 100)
    print("C.  THE DOLLAR FACTOR AND WHAT IS LEFT AFTER IT")
    print("=" * 100)
    X = lr.dropna()
    Xs = (X - X.mean()) / X.std()
    ev, evec = np.linalg.eigh(np.cov(Xs.values.T))
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    print("   variance explained: " + "  ".join("PC%d %.1f%%" % (i + 1, 100 * e / ev.sum())
                                                for i, e in enumerate(ev)))
    print("   PC1 loadings: " + "  ".join("%s %+.2f" % (c, v) for c, v in zip(X.columns, evec[:, 0])))
    dollar = Xs.values @ evec[:, 0]
    resid = pd.DataFrame(Xs.values - np.outer(dollar, evec[:, 0]), index=X.index, columns=X.columns)
    print("\n   after removing PC1, does the residual mean-revert?")
    print("   %-9s %10s %10s %10s" % ("pair", "resid ac(1)", "ac(5)", "ac(21)"))
    for p in X.columns:
        print("   %-9s %10.4f %10.4f %10.4f" % (p, resid[p].autocorr(1), resid[p].autocorr(5), resid[p].autocorr(21)))
    cum = resid.cumsum()
    print("\n   half-life of the cumulative residual (Ornstein-Uhlenbeck fit, trading days):")
    for p in X.columns:
        y = cum[p].diff().dropna(); x = cum[p].shift(1).dropna().loc[y.index]
        b = sm.OLS(y.values, sm.add_constant(x.values)).fit().params[1]
        hl = -np.log(2) / np.log(1 + b) if b < 0 else np.inf
        print("      %-9s beta %+.5f  half-life %s" % (p, b, "%.1f" % hl if np.isfinite(hl) else "no reversion"))

    print("\n" + "=" * 100)
    print("D.  THE IDENTITY, DX FUTURES vs A MATCHED SYNTHETIC BASKET")
    print("=" * 100)
    fut, dx = D.load_raw()
    idx = P["index"]
    print("   contract alignment (do DX and the currency futures roll on the same cycle?)")
    sym = pd.DataFrame({nm: d["Sym"].reindex(idx) for nm, d in fut.items()})
    sym["DX"] = dx["Sym"].reindex(idx)
    suff = sym.apply(lambda c: c.str[-3:])
    same = (suff.eq(suff["DX"], axis=0)).drop(columns="DX").all(axis=1)
    print("      sessions where all five currency contracts share DX's expiry code: %.1f%%" % (100 * same.mean()))
    for c in suff.columns:
        print("         %-8s example codes: %s" % (c, ", ".join(suff[c].dropna().unique()[:4])))

    # synthetic log-DXY from the five futures (SEK omitted, weight 0.042)
    lc = np.log(P["close"])
    synth = sum(W[p] * lc[p] for p in W)
    ldx = np.log(P["dx_raw"])
    spread = ldx - synth
    spread = spread.dropna()
    print("\n   log(DX futures) - weighted log(basket of five currency futures):")
    print("      mean %+.5f | sd %.5f | min %+.5f | max %+.5f" %
          (spread.mean(), spread.std(), spread.min(), spread.max()))
    print("      in index points at DXY=100: sd %.4f pts, range %.4f pts" %
          (100 * spread.std(), 100 * (spread.max() - spread.min())))
    dsp = spread.diff().dropna()
    print("      daily CHANGE in the spread: sd %.6f (%.2f bp), ac(1) %+.3f" %
          (dsp.std(), 1e4 * dsp.std(), dsp.autocorr(1)))
    adf = sm.tsa.stattools.adfuller(spread.values, maxlag=10, regression="c")
    print("      ADF on the spread: stat %.3f, p = %.4g  -> %s" %
          (adf[0], adf[1], "stationary" if adf[1] < 0.05 else "NOT stationary"))
    y = spread.diff().dropna(); x = spread.shift(1).dropna().loc[y.index]
    b = sm.OLS(y.values, sm.add_constant(x.values)).fit().params[1]
    print("      OU half-life of the spread: %.1f trading days" %
          (-np.log(2) / np.log(1 + b) if b < 0 else np.inf))
    print("\n      NOTE: SEK carries weight 0.042 and has no futures contract, so part of this")
    print("      spread is unhedged Swedish krona. Quantifying that next.")
    fr = pd.read_csv(str(paths.FRED_DAILY), parse_dates=["Date"]).set_index("Date")
    sek = np.log(fr["DEXSDUS"]).reindex(idx).ffill()          # log(USDSEK)
    spread_adj = (ldx - (synth + SEK_W * sek)).dropna()
    print("      spread WITH the FRED krona leg included: sd %.5f (was %.5f)" %
          (spread_adj.std(), spread.std()))
    print("      -> krona omission explains %.0f%% of the spread variance" %
          (100 * (1 - spread_adj.var() / spread.var())))
    dsa = spread_adj.diff().dropna()
    print("      daily change of the krona-adjusted spread: sd %.2f bp, ac(1) %+.3f" %
          (1e4 * dsa.std(), dsa.autocorr(1)))

    print("\n" + "=" * 100)
    print("E.  CALENDAR STRUCTURE")
    print("=" * 100)
    dxr = P["dx_ret"]
    print("   dollar return by day of week (bp/day):")
    for i, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        s = dxr[dxr.index.dayofweek == i]
        t, p = st.ttest_1samp(s.dropna(), 0)
        print("      %-4s %+7.2f bp  (n %4d, t %+5.2f, p %.3f)" % (nm, 1e4 * s.mean(), len(s), t, p))
    eom = dxr.index.to_period("M")
    is_last = pd.Series(eom, index=dxr.index) != pd.Series(eom, index=dxr.index).shift(-1)
    rank_in_month = dxr.groupby(eom).cumcount()
    n_in_month = dxr.groupby(eom).transform("size")
    from_end = n_in_month - 1 - rank_in_month
    print("\n   dollar return by position relative to month end (bp/day):")
    for k in range(0, 6):
        s = dxr[from_end == k]
        t, p = st.ttest_1samp(s.dropna(), 0)
        print("      %d session(s) before month end: %+7.2f bp (n %4d, t %+5.2f, p %.3f)"
              % (k, 1e4 * s.mean(), len(s), t, p))
    print("\n   pair returns on the last session of the month (bp):")
    for p_ in D.PAIRS:
        s = r[p_][is_last.values]
        t, pv = st.ttest_1samp(s.dropna(), 0)
        print("      %-9s %+7.2f bp (n %3d, t %+5.2f, p %.3f)" % (p_, 1e4 * s.mean(), len(s), t, pv))


if __name__ == "__main__":
    main()
