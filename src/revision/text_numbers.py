"""Numbers quoted in the running text that no other script prints.

I kept these in one place so every figure in the paper has a line of output behind it:
  1. the regime strategy before and after the np.where fix (the 0.79, t = 3.09 in the
     original account) and with no regime overlay
  2. overnight legs of both sources at 8, 9 and 12 Newey-West lags
  3. buy-and-hold at the automatic bandwidth
  4. the Section 8.1 tests across the 24 strategies (ANOVA and its assumption checks)
  5. how the smoothed carry strategy's return splits into differential held and spot"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "part2_declared_search"))
sys.path.insert(0, os.path.join(_HERE, "..", "part4_reasoned"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox
import data as D, evaluate as E
import engine as G
import naive_stats as N
from carry_futures import roll_carry
from carry_dispersion import carry_linear

print("1. REGIME STRATEGY, quarterly bottom-2, 126-day DXY regime")
for kind, lab in [("spot", "Federal Reserve spot, 6 pairs"), ("futures", "exchange futures, 5 pairs")]:
    R, lvl, _ = G.universe(kind)
    for bug in (True, False):
        r = G.strat(R, lvl, "Q", "bottom", 2, True, 126, nan_as_rising=bug)
        t, p = G.nw(r)
        print("   %-30s %-28s Sharpe %+.3f  t %.2f  p %.4f"
              % (lab, "np.where as committed" if bug else "fixed (no position on NaN)", G.sr(r), t, p))
R, lvl, _ = G.universe("futures")
r = G.strat(R, lvl, "Q", "bottom", 2, False, 126)
print("   %-30s %-28s Sharpe %+.3f" % ("exchange futures, 5 pairs", "no regime overlay", G.sr(r)))

print("\n2. OVERNIGHT LEG AT 8, 9 AND 12 LAGS (sessions present in both sources)")
yh, bc = N.load_free(), N.load_exch()
j = bc.merge(yh, on="Date", suffixes=("_b", "_y")).dropna().reset_index(drop=True)
for lab, o, c in [("free", j.O_y, j.C_y), ("exchange", j.O_b, j.C_b)]:
    x = (o / c.shift(1) - 1).fillna(0).values
    out = []
    for l in (8, 9, 12):
        out.append("%d lags t %+.2f p %.4g" % (l, *N.nw(x, l)))
    print("   %-9s n %d | %s" % (lab, len(x), " | ".join(out)))

print("\n3. BUY-AND-HOLD ON THE FREE SERIES")
R24, bnh, co, oc = N.grid(yh)
lag = int(np.floor(4 * (len(bnh) / 100.0) ** (2.0 / 9.0)))
t, p = N.nw(bnh, lag)
print("   n %d, automatic bandwidth %d lags: Sharpe %.3f  t %.3f  p %.3f" % (len(bnh), lag, N.sr(bnh), t, p))

print("\n4. ARE THE 24 STRATEGIES DISTINGUISHABLE?")
cols = [R24[k].values for k in R24.columns]
F, pF = st.f_oneway(*cols)
print("   one-way ANOVA        F(%d, %d) = %.2f  p %.2g" % (len(cols) - 1, sum(map(len, cols)) - len(cols), F, pF))
print("   Levene               W = %.2f  p %.2g" % st.levene(*cols))
jb = st.jarque_bera(R24[7].values)
print("   Jarque-Bera, S7      JB = %.1f  p %.2g" % (jb.statistic, jb.pvalue))
lb = acorr_ljungbox(R24[7].values, lags=[10])
print("   Ljung-Box, S7, 10    Q = %.2f  p %.2g" % (lb.lb_stat.iloc[0], lb.lb_pvalue.iloc[0]))
print("   Kruskal-Wallis       H = %.2f  p %.2g" % st.kruskal(*cols))

print("\n5. SMOOTHED CROSS-SECTIONAL CARRY ON FUTURES, where the return comes from")
P = D.panel()
C = roll_carry(P)
W = carry_linear(P, C.rolling(252).mean())
g, n, tu, _ = E.apply_positions(W, P, 1.0)
held = (W.shift(1).fillna(0.0) * C / 63.0).sum(axis=1)
print("   Sharpe %.3f net | gross return %.2f%%/yr = differential held %.2f%% + spot %.2f%%"
      % (E.sharpe(n), 100 * 252 * g.mean(), 100 * 252 * held.mean(), 100 * 252 * (g.mean() - held.mean())))
