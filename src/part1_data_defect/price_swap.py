"""Overnight and daytime decomposition with the two sources reconciled properly.

The two sources carry a persistent level offset, the exchange series sitting about 0.04
index points below the free one. A return formed from one source's open and the other's
close inherits that offset as a mechanical bias, which is enough to manufacture a result
on its own. Two corrections:

  * transfer the close-to-open return rather than the price level, which is scale free
  * remove the same-day level offset before locating one source's open inside the other's
    range"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, statsmodels.api as sm
T = 252
U = str(paths.DATA)
bc = pd.read_csv(str(paths.DX_FILE),
                 skipfooter=1, engine="python")
bc = pd.DataFrame({"Date": pd.to_datetime(bc["Time"]),
                   "O": pd.to_numeric(bc["Open"],   errors="coerce"),
                   "H": pd.to_numeric(bc["High"],   errors="coerce"),
                   "L": pd.to_numeric(bc["Low"],    errors="coerce"),
                   "C": pd.to_numeric(bc["Latest"], errors="coerce")}).sort_values("Date").reset_index(drop=True)
yh = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"])
yh = yh[["Date","DXY_Open","DXY_High","DXY_Low","DXY_Close"]].rename(
     columns={"DXY_Open":"O","DXY_High":"H","DXY_Low":"L","DXY_Close":"C"}).sort_values("Date").reset_index(drop=True)
j = bc.merge(yh, on="Date", suffixes=("_b","_y")).dropna(
    subset=["O_b","H_b","L_b","C_b","O_y","H_y","L_y","C_y"]).reset_index(drop=True)

print("="*92)
print("0.  THE LEVEL OFFSET THAT INVALIDATED THE NAIVE SWAP")
print("="*92)
off = j.C_y - j.C_b
print("close(free) - close(exchange): mean %+.4f  sd %.4f  (index points)" % (off.mean(), off.std()))
print("typical daily range, exchange : mean %.4f  median %.4f" % ((j.H_b-j.L_b).mean(), (j.H_b-j.L_b).median()))
print("offset as a share of one day's range: %.1f%%" % (100*off.mean()/(j.H_b-j.L_b).mean()))
print("-> large enough that a cross-vendor price ratio is dominated by the offset.")

print("\n" + "="*92)
print("1.  OPEN LOCATION, WITH THE SAME-DAY OFFSET REMOVED")
print("="*92)
rng = j.H_b - j.L_b; k = rng > 0
# put the free open on the exchange's price level using the same-day close offset
Oy_adj = j.O_y - off
loc_raw = ((j.O_y  - j.L_b)/rng)[k]
loc_adj = ((Oy_adj - j.L_b)/rng)[k]
loc_bch = ((j.O_b  - j.L_b)/rng)[k]
print("position of the open inside the exchange daily range (0 = low, 1 = high):")
print("   exchange open                       mean %.3f   median %.3f" % (loc_bch.mean(), loc_bch.median()))
print("   free open, offset NOT removed       mean %.3f   median %.3f   <- contaminated" % (loc_raw.mean(), loc_raw.median()))
print("   free open, offset removed           mean %.3f   median %.3f" % (loc_adj.mean(), loc_adj.median()))
print("\nshare of sessions where the open lies OUTSIDE the exchange high-low range:")
print("   exchange open              %5.2f%%" % (100*(((j.O_b<j.L_b)|(j.O_b>j.H_b))[k]).mean()))
print("   free open, offset removed  %5.2f%%" % (100*(((Oy_adj<j.L_b)|(Oy_adj>j.H_b))[k]).mean()))
print("   (an open outside the true session range cannot be a traded price)")

print("\n" + "="*92)
print("2.  THE SWAP, DONE IN RETURN SPACE (scale free)")
print("="*92)
print("Each row uses ONE vendor's price level throughout. Only the close-to-open")
print("return is transferred. Daytime is then the residual, so the 24h return is")
print("identical across rows by construction.")
sr = lambda x: np.asarray(x,float).mean()/np.asarray(x,float).std(ddof=1)*np.sqrt(T)
tv = lambda x: 100*np.prod(1+np.asarray(x,float))
def nw(x):
    x=np.asarray(x,float)
    r=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':int(np.floor(4*(len(x)/100.0)**(2.0/9.0)))})
    return float(r.tvalues[0]), float(r.pvalues[0])

co_y = (j.O_y/j.C_y.shift(1)-1)          # free overnight return
co_b = (j.O_b/j.C_b.shift(1)-1)          # exchange overnight return
cc_y = (j.C_y/j.C_y.shift(1)-1)          # free 24h return
cc_b = (j.C_b/j.C_b.shift(1)-1)          # exchange 24h return

rows = [("free 24h,     free overnight",     cc_y, co_y),
        ("free 24h,     EXCHANGE overnight", cc_y, co_b),
        ("exchange 24h, exchange overnight", cc_b, co_b),
        ("exchange 24h, FREE overnight",     cc_b, co_y)]
print("\n%-36s %9s %8s %8s   %9s %8s %8s" % ("", "ON 100->","ON SR","ON p","DAY 100->","DAY SR","DAY p"))
for lab, cc, co in rows:
    c = co.fillna(0).values
    d = ((1+cc.fillna(0).values)/(1+c) - 1)      # daytime as the exact residual
    t1,p1 = nw(c); t2,p2 = nw(d)
    print("%-36s %9.2f %8.3f %8.4f   %9.2f %8.3f %8.4f"
          % (lab, tv(c), sr(c), p1, tv(d), sr(d), p2))
print("\n-> the overnight leg tracks whichever vendor's OPEN is used, on either")
print("   vendor's price level. The close series is not the source of the effect.")

print("\n" + "="*92)
print("3.  THE SESSIONS THAT CARRY IT, WITH THE OFFSET REMOVED")
print("="*92)
ky = (j.H_y > j.L_y)
isLo = pd.Series(False, index=j.index); isLo[ky] = np.isclose(j.O_y[ky], j.L_y[ky], rtol=0, atol=1e-9)
isHi = pd.Series(False, index=j.index); isHi[ky] = np.isclose(j.O_y[ky], j.H_y[ky], rtol=0, atol=1e-9)
for lab, sel in [("free open at its own LOW", isLo & k), ("free open at its own HIGH", isHi & k),
                 ("all other sessions", (~isLo) & (~isHi) & k)]:
    s = j[sel]; oa = (s.O_y - off[sel])
    out = ((oa < s.L_b) | (oa > s.H_b)).mean()
    print("  %-27s n=%4d | offset-adjusted open sits at %+.3f of the exchange range | outside it %5.1f%%"
          % (lab, len(s), ((oa - s.L_b)/(s.H_b - s.L_b)).mean(), 100*out))
print("\n  free overnight return on those sessions:")
for lab, sel in [("open at own LOW", isLo), ("open at own HIGH", isHi), ("all others", (~isLo)&(~isHi))]:
    print("     %-18s n=%4d   %+.4f%%/day" % (lab, sel.sum(), 100*co_y[sel].mean()))
print("  exchange overnight return on the SAME sessions:")
for lab, sel in [("open at own LOW", isLo), ("open at own HIGH", isHi), ("all others", (~isLo)&(~isHi))]:
    print("     %-18s n=%4d   %+.4f%%/day" % (lab, sel.sum(), 100*co_b[sel].mean()))
