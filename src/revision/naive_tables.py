"""Revision 2: Tables 2 to 11 of the paper from the released code.

These tables were first produced in the original notebooks. I rebuild them here from the
same grid as naive_stats.py so that every number in Sections 4 to 7 has a line of output
behind it: the worked example (Tables 2 to 5), the 24-strategy grid at zero cost
(Table 6) and the cost tables (Tables 8 to 11), plus the break-even costs of Figure 4."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd
from scipy.optimize import brentq
import naive_stats as N

T = 252
yh = N.load_free()

def trades(d, a, b):
    # same rule as the cost charge in naive_stats.grid: a trade whenever the target
    # position differs from the one carried in and is non-zero
    co = (d.O / d.C.shift(1) - 1).values; oc = (d.C / d.O - 1).values
    prev = np.concatenate(([np.nan], oc[:-1])); held = 0; n = 0
    for i in range(len(d)):
        p = N.pos(a, prev[i]); n += (p != 0 and p != held); held = p
        q = N.pos(b, co[i]); n += (q != 0 and q != held); held = q
    return n

def stats(r):
    r = np.asarray(r, float); eq = 100 * np.cumprod(1 + r); yrs = len(r) / T
    return dict(final=eq[-1], tot=eq[-1] - 100, cagr=100 * ((eq[-1] / 100) ** (1 / yrs) - 1), sr=N.sr(r),
                dd=100 * np.min(eq / np.maximum.accumulate(eq) - 1), vol=100 * r.std(ddof=1) * np.sqrt(T))

print("TABLES 2-5. WORKED EXAMPLE, week of 27 June 2016")
w = yh[(yh.Date >= "2016-06-24") & (yh.Date <= "2016-07-01")].reset_index(drop=True)
for _, x in w.iterrows():
    print("   %s  O %.2f  C %.2f" % (x.Date.date(), x.O, x.C))
co = w.O / w.C.shift(1) - 1; oc = w.C / w.O - 1; cc = (1 + co) * (1 + oc) - 1
for i in range(1, len(w)):
    print("   %s  night %+.3f%%  day %+.3f%%  24h %+.3f%%" % (w.Date[i].date(), 100 * co[i], 100 * oc[i], 100 * cc[i]))
print("   week: night %+.3f%%  day %+.3f%%  24h %+.3f%%" % (100 * (np.prod(1 + co[1:]) - 1), 100 * (np.prod(1 + oc[1:]) - 1), 100 * (np.prod(1 + cc[1:]) - 1)))
R0, bnh, _, _ = N.grid(yh)
m = ((yh.Date >= "2016-06-27") & (yh.Date <= "2016-07-01")).values
for k in (5, 2, 3, 7, 20):
    eq = 100 * np.cumprod(1 + R0[k].values[m])
    print("   S%-2d %-22s %s" % (k, "(%s, %s)" % N.MAP[k], " ".join("%.2f" % v for v in eq)))

G = {c: N.grid(yh, c)[0] for c in (0.0, 1.0, 2.0)}
print("\nTABLE 6. ALL 24 STRATEGIES AT ZERO COST")
print("   %-3s %-9s %-9s %8s %7s %7s %7s %7s %6s %6s" % ("#", "night", "day", "final", "tot%", "cagr%", "SR", "maxdd%", "vol%", "trades"))
for k, (a, b) in N.MAP.items():
    s = stats(G[0.0][k])
    print("   %-3d %-9s %-9s %8.2f %7.1f %7.2f %7.3f %7.1f %6.2f %6d" % (k, a, b, s["final"], s["tot"], s["cagr"], s["sr"], s["dd"], s["vol"], trades(yh, a, b)))
bh = 100 * yh.C.ffill().iloc[-1] / yh.C.iloc[0]
s = stats(bnh)
print("   buy-and-hold (forward-filled close) final %.2f tot %.1f cagr %.2f SR %.3f maxdd %.1f vol %.2f"
      % (bh, bh - 100, 100 * ((bh / 100) ** (T / len(bnh)) - 1), s["sr"], s["dd"], s["vol"]))

print("\nTABLES 8-11. TERMINAL VALUE, SHARPE, VOLATILITY AND MAXIMUM DRAWDOWN AT 0, 1 AND 2 BP PER TRADE")
print("   %-3s %26s | %23s | %20s | %20s" % ("#", "final 0/1/2", "Sharpe 0/1/2", "vol% 0/1/2", "maxdd% 0/1/2"))
for k in N.MAP:
    ss = [stats(G[c][k]) for c in (0.0, 1.0, 2.0)]
    print("   %-3d %8.2f %8.2f %8.2f | %7.3f %7.3f %7.3f | %6.2f %6.2f %6.2f | %6.1f %6.1f %6.1f"
          % (k, *[x["final"] for x in ss], *[x["sr"] for x in ss], *[x["vol"] for x in ss], *[x["dd"] for x in ss]))
for c in (0.0, 1.0, 2.0):
    s5 = stats(G[c][5])
    beat = [k for k in N.MAP if k != 5 and stats(G[c][k])["final"] > s5["final"]]
    beat_sr = [k for k in N.MAP if k != 5 and stats(G[c][k])["sr"] > s5["sr"]]
    low_vol = [k for k in N.MAP if k != 5 and stats(G[c][k])["vol"] < s5["vol"]]
    shallow = [k for k in N.MAP if k != 5 and stats(G[c][k])["dd"] > s5["dd"]]
    print("   %.0f bp: beat S5 on final %d, on Sharpe %d, lower vol %d, shallower drawdown %d"
          % (c, len(beat), len(beat_sr), len(low_vol), len(shallow)))

print("\nFIGURE 4. BREAK-EVEN COSTS OF STRATEGY 7")
s7 = lambda c: 100 * np.prod(1 + N.grid(yh, c)[0][7].values)
for lab, lvl in [("buy-and-hold %.2f" % bh, bh), ("break-even 100", 100.0)]:
    print("   S7 falls below %s at %.3f bp" % (lab, brentq(lambda c: s7(c) - lvl, 0.0, 2.5, xtol=1e-6)))
