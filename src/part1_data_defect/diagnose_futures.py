"""What the five FX futures files contain, checked before any strategy runs on them.

  A. coverage, symbols, roll structure, liquidity
  B. quote convention (USD per foreign unit) against FRED
  C. correlation of futures returns with FRED spot returns, plus the lag test
  D. the open-location binomial screen on each futures series
  E. size of the roll gaps and how many sessions they touch"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, warnings
from scipy import stats as st
warnings.filterwarnings("ignore")

F = str(paths.FUTURES)
NAME = {"E6": "EURUSD", "B6": "GBPUSD", "D6": "CADUSD", "J6": "JPYUSD", "S6": "CHFUSD"}
DXP = str(paths.DX_FILE)


def load_bc(path):
    d = pd.read_csv(path, skipfooter=1, engine="python")
    out = pd.DataFrame({
        "Date": pd.to_datetime(d["Time"]), "Sym": d["Symbol"].astype(str),
        "O": pd.to_numeric(d["Open"], errors="coerce"),
        "H": pd.to_numeric(d["High"], errors="coerce"),
        "L": pd.to_numeric(d["Low"], errors="coerce"),
        "C": pd.to_numeric(d["Latest"], errors="coerce"),
        "V": pd.to_numeric(d["Volume"], errors="coerce"),
        "OI": pd.to_numeric(d["Open Int"], errors="coerce")})
    return out.sort_values("Date").reset_index(drop=True)


fut = {k: load_bc(str(paths.FUTURES / (k + ".csv"))) for k in NAME}
dx = load_bc(DXP)

print("=" * 96)
print("A.  COVERAGE, ROLL STRUCTURE, LIQUIDITY")
print("=" * 96)
print("%-6s %-8s %6s %-12s %-12s %8s %10s %10s" %
      ("sym", "pair", "rows", "first", "last", "contracts", "med vol", "med OI"))
for k, d in list(fut.items()) + [("DX", dx)]:
    nm = NAME.get(k, "DXY")
    print("%-6s %-8s %6d %-12s %-12s %8d %10.0f %10.0f" %
          (k, nm, len(d), d.Date.iloc[0].date(), d.Date.iloc[-1].date(),
           d.Sym.nunique(), d.V.median(), d.OI.median()))

print("\nroll days (session where the front symbol changes):")
for k, d in list(fut.items()) + [("DX", dx)]:
    rolls = d.index[d.Sym != d.Sym.shift(1)][1:]
    gaps = (d.C.iloc[rolls].values / d.C.iloc[rolls - 1].values - 1) * 100
    print("   %-4s %3d rolls | roll-day price gap: mean %+.3f%%  sd %.3f%%  |max| %.3f%%"
          % (k, len(rolls), gaps.mean(), gaps.std(), np.abs(gaps).max()))

print("\n" + "=" * 96)
print("B.  QUOTE CONVENTION AND ALIGNMENT vs FEDERAL RESERVE H.10")
print("=" * 96)
fr = pd.read_csv(str(paths.FRED_DAILY), parse_dates=["Date"]).set_index("Date")
spot = pd.DataFrame({"EURUSD": fr.DEXUSEU, "GBPUSD": fr.DEXUSUK, "JPYUSD": 1 / fr.DEXJPUS,
                     "CADUSD": 1 / fr.DEXCAUS, "CHFUSD": 1 / fr.DEXSZUS})
print("%-8s %10s %10s %9s %9s | %s" %
      ("pair", "fut med", "spot med", "ratio", "meandiff%", "return corr at lag -1 / 0 / +1"))
for k, nm in NAME.items():
    d = fut[k].set_index("Date")
    j = pd.concat([d.C.rename("f"), spot[nm].rename("s")], axis=1).dropna()
    rf = j.f.pct_change()
    rs = j.s.pct_change()
    cs = [rf.corr(rs.shift(L)) for L in (-1, 0, 1)]
    print("%-8s %10.5f %10.5f %9.4f %+9.3f | %+.3f  %+.3f  %+.3f" %
          (nm, j.f.median(), j.s.median(), (j.f / j.s).median(),
           100 * (j.f / j.s - 1).mean(), cs[0], cs[1], cs[2]))

print("\n" + "=" * 96)
print("C.  OPEN-LOCATION BINOMIAL SCREEN ON EACH EXCHANGE FUTURES SERIES")
print("=" * 96)
print("Under the null that a recorded open is an unbiased draw from the session,")
print("P(open = daily low) = P(open = daily high).")
print("%-10s %8s %8s %12s %12s" % ("series", "at low", "at high", "binomial p", "open loc"))
for k, d in list(fut.items()) + [("DX", dx)]:
    nm = NAME.get(k, "DXY")
    ok = (d.H > d.L) & d.O.notna()
    lo = int(np.isclose(d.O[ok], d.L[ok], rtol=0, atol=1e-12).sum())
    hi = int(np.isclose(d.O[ok], d.H[ok], rtol=0, atol=1e-12).sum())
    p = st.binomtest(lo, lo + hi, 0.5).pvalue if (lo + hi) else np.nan
    loc = ((d.O[ok] - d.L[ok]) / (d.H[ok] - d.L[ok])).mean()
    print("%-10s %8d %8d %12.4g %12.3f" % (nm + " fut", lo, hi, p, loc))

print("\n" + "=" * 96)
print("D.  INTERNAL CONSISTENCY")
print("=" * 96)
for k, d in list(fut.items()) + [("DX", dx)]:
    nm = NAME.get(k, "DXY")
    viol = ((d.O < d.L) | (d.O > d.H) | (d.C < d.L) | (d.C > d.H)).mean()
    print("   %-8s OHLC violations %5.2f%% | open==prev close %5.2f%% | zero-range days %5.2f%% | missing %d"
          % (nm, 100 * viol,
             100 * np.isclose(d.O, d.C.shift(1), rtol=0, atol=1e-12).mean(),
             100 * np.isclose(d.H, d.L, rtol=0, atol=1e-12).mean(), int(d.C.isna().sum())))
