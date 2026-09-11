"""Attempts to break the exchange series before treating it as the trustworthy one.

  A. does the exchange series have stale or degenerate opens of its own?
  B. is "Latest" a settlement price or a last trade? does the difference matter?
  C. is the free series' problem confined to particular years or is it systemic?
  D. what does the PBO contrast actually mean?"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import statsmodels.api as sm
T=252; U = str(paths.DATA)
bc=pd.read_csv(str(paths.DX_FILE),
               skipfooter=1,engine="python")
bc=pd.DataFrame({"Date":pd.to_datetime(bc["Time"]),"Sym":bc["Symbol"],
    "O":pd.to_numeric(bc["Open"],errors="coerce"),"H":pd.to_numeric(bc["High"],errors="coerce"),
    "L":pd.to_numeric(bc["Low"],errors="coerce"),"C":pd.to_numeric(bc["Latest"],errors="coerce"),
    "V":pd.to_numeric(bc["Volume"],errors="coerce")}).sort_values("Date").reset_index(drop=True)
yh=pd.read_csv(str(paths.VENDOR_FILE),parse_dates=["Date"])
yh=yh[["Date","DXY_Open","DXY_High","DXY_Low","DXY_Close"]].rename(
   columns={"DXY_Open":"O","DXY_High":"H","DXY_Low":"L","DXY_Close":"C"}).sort_values("Date").reset_index(drop=True)

print("="*92); print("A.  DOES THE EXCHANGE SERIES HAVE DEGENERATE PRINTS OF ITS OWN?"); print("="*92)
for lab,d in [("exchange",bc),("free",yh)]:
    n=len(d)
    print("\n%s (n=%d)"%(lab,n))
    print("   open == previous close exactly : %5.2f%%"%(100*np.isclose(d.O,d.C.shift(1),rtol=0,atol=1e-9).mean()))
    print("   open == same-day close exactly : %5.2f%%"%(100*np.isclose(d.O,d.C,rtol=0,atol=1e-9).mean()))
    print("   high == low (zero range)       : %5.2f%%"%(100*np.isclose(d.H,d.L,rtol=0,atol=1e-9).mean()))
    print("   any OHLC violation (O or C outside [L,H]) : %5.2f%%"
          %(100*(((d.O<d.L)|(d.O>d.H)|(d.C<d.L)|(d.C>d.H)).mean())))
    print("   missing rows                   : %d"%d.C.isna().sum())
    r=(d.C/d.C.shift(1)-1).dropna()
    print("   daily close return: sd %.4f%%  skew %+.2f  kurt %.1f  |max| %.2f%%"
          %(100*r.std(),st.skew(r),st.kurtosis(r,fisher=False),100*r.abs().max()))
    v=(d.O/d.C.shift(1)-1).dropna()
    print("   overnight return  : sd %.4f%%  |max| %.2f%%"%(100*v.std(),100*v.abs().max()))

print("\n"+"="*92); print("B.  VARIANCE SHARE OF THE TWO SUB-PERIODS"); print("="*92)
print("A plausible 24h split should put a substantial share in each leg.")
for lab,d in [("exchange",bc),("free",yh)]:
    co=(d.O/d.C.shift(1)-1).fillna(0); oc=(d.C/d.O-1).fillna(0)
    tot=co.var()+oc.var()
    print("   %-10s overnight %5.1f%%   daytime %5.1f%%   (ann vol: ON %.2f%%, DAY %.2f%%)"
          %(lab,100*co.var()/tot,100*oc.var()/tot,100*co.std()*np.sqrt(T),100*oc.std()*np.sqrt(T)))

print("\n"+"="*92); print("C.  IS THE FREE SERIES' DEFECT CONCENTRATED IN TIME?"); print("="*92)
k=(yh.H>yh.L); lo=pd.Series(False,index=yh.index); hi=pd.Series(False,index=yh.index)
lo[k]=np.isclose(yh.O[k],yh.L[k],rtol=0,atol=1e-9); hi[k]=np.isclose(yh.O[k],yh.H[k],rtol=0,atol=1e-9)
g=pd.DataFrame({"y":yh.Date.dt.year,"lo":lo,"hi":hi,"k":k}).groupby("y").sum()
co_y=(yh.O/yh.C.shift(1)-1).fillna(0)
g["ON%/yr"]=[100*((1+co_y[yh.Date.dt.year==y]).prod()-1) for y in g.index]
g["binom_p"]=[float(st.binomtest(int(r.lo),int(r.lo+r.hi),0.5).pvalue) if (r.lo+r.hi)>0 else np.nan
              for _,r in g.iterrows()]
print(g[["lo","hi","ON%/yr","binom_p"]].round(4).to_string())
print("\n-> a defect present in most years is systemic, not one bad patch.")

print("\n"+"="*92); print("D.  WHAT THE PBO CONTRAST MEANS"); print("="*92)
print("""  free series      PBO = 0.016-0.028   median OOS rank of the IS winner = 24/24
  exchange series  PBO = 0.765-0.779   median OOS rank of the IS winner =  6-8/24

  The overfitting diagnostic says the FALSE result is highly stable and the TRUE
  data is pure noise-fitting. Both readings are correct on their own terms. PBO
  measures whether an in-sample winner keeps winning out of sample; it cannot
  distinguish a persistent economic effect from a persistent data defect.
  A stationary artefact is, to every overfitting diagnostic, indistinguishable
  from a real anomaly.""")

print("\n"+"="*92); print("E.  THE NULL'S EXPECTED MAXIMUM vs THE OBSERVED MAXIMUM"); print("="*92)
MAPk={1:("Long","Cash"),2:("Short","Cash"),3:("Cash","Long"),4:("Cash","Short"),5:("Long","Long"),
6:("Short","Short"),7:("Short","Long"),8:("Long","Short"),9:("Cash","Inertia"),10:("Cash","Reversal"),
11:("Inertia","Cash"),12:("Reversal","Cash"),13:("Inertia","Inertia"),14:("Inertia","Reversal"),
15:("Reversal","Inertia"),16:("Reversal","Reversal"),17:("Long","Inertia"),18:("Long","Reversal"),
19:("Short","Inertia"),20:("Short","Reversal"),21:("Inertia","Long"),22:("Reversal","Long"),
23:("Inertia","Short"),24:("Reversal","Short")}
def pos(r,s):
    if r=="Cash":return 0
    if r=="Long":return 1
    if r=="Short":return -1
    if np.isnan(s):return 0
    b=1 if s>=0 else -1
    return b if r=="Inertia" else -b
def srs_of(d):
    co=(d.O/d.C.shift(1)-1).values; oc=(d.C/d.O-1).values; N=len(d); out=[]
    for k_ in range(1,25):
        a,b=MAPk[k_]; prev=np.concatenate(([np.nan],oc[:-1])); ro=np.empty(N); rd=np.empty(N)
        for i in range(N):
            p=pos(a,prev[i]); ro[i]=0.0 if np.isnan(co[i]) else p*co[i]
            q=pos(b,co[i]);  rd[i]=0.0 if np.isnan(oc[i]) else q*oc[i]
        r=(1+ro)*(1+rd)-1; out.append(r.mean()/r.std(ddof=1)*np.sqrt(T))
    return np.array(out)
for lab,d in [("exchange",bc),("free",yh)]:
    s=srs_of(d); v=np.var(s/np.sqrt(T),ddof=1); g_=np.euler_gamma
    sstar=np.sqrt(v)*((1-g_)*st.norm.ppf(1-1/24)+g_*st.norm.ppf(1-1/(24*np.e)))*np.sqrt(T)
    print("   %-10s observed max Sharpe %+.3f | null expected max %+.3f | %s"
          %(lab,s.max(),sstar,"observed EXCEEDS null" if s.max()>sstar else "observed BELOW null"))
