"""The carry book on the longest sample the data allows.

The futures panel is 15 years and five currencies, which is part of why nothing reached
significance. FRED gives spot back to 1999 and OECD three-month interbank rates for all
six DXY currencies, with the Japanese series starting in April 2002 as the binding
constraint. That gives 24 years and six currencies instead of 15 and five.

This is spot rather than futures, so it is not directly tradeable. Its job is to answer a
different question, whether the carry premium exists over a long sample. The futures
result then says whether it survives being traded."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import evaluate as E
TD=252
A=pd.read_csv(str(paths.FRED_LONG),parse_dates=["Date"]).set_index("Date").sort_index()
# quote everything as USD per foreign unit
px=pd.DataFrame({"EURUSD":A.DEXUSEU,"GBPUSD":A.DEXUSUK,"JPYUSD":1/A.DEXJPUS,
                 "CADUSD":1/A.DEXCAUS,"CHFUSD":1/A.DEXSZUS,"SEKUSD":1/A.DEXSDUS})
RC={"EURUSD":"IR3TIB01EZM156N","GBPUSD":"IR3TIB01GBM156N","JPYUSD":"IR3TIB01JPM156N",
    "CADUSD":"IR3TIB01CAM156N","CHFUSD":"IR3TIB01CHM156N","SEKUSD":"IR3TIB01SEM156N"}
rates=A[[RC[p] for p in RC]+["IR3TIB01USM156N"]].copy()
rates.columns=list(RC)+["USD"]
start=max(rates[c].first_valid_index() for c in rates.columns)
px=px.loc[start:].dropna(how="any")
rates=rates.reindex(px.index).ffill()
ret=px.pct_change().fillna(0.0)
# carry = foreign 3M minus US 3M, known at month end, lagged into the next session
carry=pd.DataFrame({p:(rates[p]-rates["USD"])/100.0 for p in px.columns}).ffill()

def clin(C):
    d=C.sub(C.mean(axis=1),axis=0); g=d.abs().sum(axis=1)
    return d.div(g.where(g>0,np.nan),axis=0).fillna(0.0)
def rank_w(C,k):
    W=pd.DataFrame(0.0,index=C.index,columns=C.columns)
    n=C.notna().sum(axis=1); rk=C.rank(axis=1,ascending=False,method="first")
    W[rk.le(k)]=1.0; W[rk.gt(n-k,axis=0)]=-1.0
    g=W.abs().sum(axis=1)
    return W.div(g.where(g>0,np.nan),axis=0).fillna(0.0)

P={"ret":ret}
print("="*104); print("LONG-SAMPLE CARRY ,  %s .. %s  |  %d sessions  |  %d currencies"
      %(px.index[0].date(),px.index[-1].date(),len(px),px.shape[1])); print("="*104)
print("   mean carry offered, annualised:")
for p in px.columns: print("      %-9s %+6.2f%%"%(p,100*carry[p].mean()))

print("\n   %-40s %8s %9s %8s %9s %8s %8s"%("strategy","Sharpe","TotRet%","CAGR%","t","p","turn/yr"))
out={}
for nm,W in [("G1 carry, linear weights",clin(carry)),
             ("G2 carry, linear, 1y smoothed",clin(carry.rolling(252).mean())),
             ("G3 carry, top/bottom 2",rank_w(carry,2)),
             ("G4 carry, top/bottom 1",rank_w(carry,1)),
             ("G5 carry, top/bottom 3",rank_w(carry,3))]:
    g,n,t,_=E.apply_positions(W,P,1.0); s=E.summary(n,t); out[nm]=n
    print("   %-40s %8.3f %9.1f %8.2f %9.2f %8.4f %8.1f"
          %(nm,s["Sharpe"],100*s["TotRet"],100*s["CAGR"],s["t"],s["p"],s["TurnoverPA"]))

print("\n"+"="*104); print("MECHANISM: is the differential actually kept?"); print("="*104)
W=clin(carry.rolling(252).mean())
g,_,_,_=E.apply_positions(W,P,0.0)
earned=(W.shift(1).fillna(0)*(carry/TD)).sum(axis=1)
print("   annualised return from the differential held: %+.2f%%"%(100*TD*earned.mean()))
print("   annualised total return                     : %+.2f%%"%(100*TD*g.mean()))
print("   residual (spot)                             : %+.2f%%"%(100*TD*(g.mean()-earned.mean())))

print("\n"+"="*104); print("STABILITY"); print("="*104)
n=out["G2 carry, linear, 1y smoothed"]
for a,b in [(2002,2007),(2008,2013),(2014,2019),(2020,2026)]:
    m=(n.index.year>=a)&(n.index.year<=b); s=E.summary(n[m])
    if s: print("   %d-%d  Sharpe %+6.3f  TotRet %+7.1f%%  t %+5.2f  p %.4f"%(a,b,s["Sharpe"],100*s["TotRet"],s["t"],s["p"]))
print("\n   drop-one-currency:")
for drop in px.columns:
    keep=[q for q in px.columns if q!=drop]
    Q={"ret":ret[keep]}
    Wq=clin(carry[keep].rolling(252).mean())
    _,nq,_,_=E.apply_positions(Wq,Q,1.0); s=E.summary(nq)
    print("      without %-9s Sharpe %+6.3f  t %+5.2f  p %.4f"%(drop,s["Sharpe"],s["t"],s["p"]))

print("\n"+"="*104); print("NOISE CEILING on this long sample (5 carry variants)"); print("="*104)
rv=ret.values; nn=len(ret); ceil=[]
for seed in range(10):
    rng=np.random.default_rng(300+seed)
    idx=E.stationary_bootstrap_idx(nn,1,rng,mean_block=21)[0]
    Q={"ret":pd.DataFrame(rv[idx],index=ret.index,columns=ret.columns)}
    srs=[]
    for Wk in [clin(carry),clin(carry.rolling(252).mean()),rank_w(carry,2),rank_w(carry,1),rank_w(carry,3)]:
        _,q,_,_=E.apply_positions(Wk,Q,1.0); srs.append(E.sharpe(q))
    ceil.append(max(srs))
print("   "+"  ".join("%+.2f"%c for c in ceil))
print("   ceiling mean %+.3f | 90th pct %+.3f | max %+.3f"%(np.mean(ceil),np.percentile(ceil,90),max(ceil)))
best=max(out,key=lambda k:E.sharpe(out[k]))
print("   best observed %+.3f (%s)  -> %s"%(E.sharpe(out[best]),best,
      "ABOVE ceiling" if E.sharpe(out[best])>np.percentile(ceil,90) else "within noise"))
pd.DataFrame(out).to_pickle(str(paths.RESULTS)+"/part4_carry_long.pkl")
