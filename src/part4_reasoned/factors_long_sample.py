"""The canonical FX factor set on the longest sample available.

Asness, Moskowitz and Pedersen (2013) argue that value and momentum work across asset
classes and work best together, because they are negatively correlated. Menkhoff et al.
(2012) document currency momentum and Lustig, Roussanov and Verdelhan document carry. If a
profitable FX strategy exists in G10 at all, this combination over 24 years is where it
should show up."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
import evaluate as E
TD=252
A=pd.read_csv(str(paths.FRED_LONG),parse_dates=["Date"]).set_index("Date").sort_index()
px=pd.DataFrame({"EURUSD":A.DEXUSEU,"GBPUSD":A.DEXUSUK,"JPYUSD":1/A.DEXJPUS,
                 "CADUSD":1/A.DEXCAUS,"CHFUSD":1/A.DEXSZUS,"SEKUSD":1/A.DEXSDUS}).dropna(how="any")
RC={"EURUSD":"IR3TIB01EZM156N","GBPUSD":"IR3TIB01GBM156N","JPYUSD":"IR3TIB01JPM156N",
    "CADUSD":"IR3TIB01CAM156N","CHFUSD":"IR3TIB01CHM156N","SEKUSD":"IR3TIB01SEM156N"}
rates=A[[RC[p] for p in RC]+["IR3TIB01USM156N"]]; rates.columns=list(RC)+["USD"]
start=max(rates[c].first_valid_index() for c in rates.columns)
px=px.loc[start:]; rates=rates.reindex(px.index).ffill()
ret=px.pct_change().fillna(0.0); P={"ret":ret}
carry=pd.DataFrame({p:(rates[p]-rates["USD"])/100.0 for p in px.columns}).ffill()

def clin(C):
    d=C.sub(C.mean(axis=1),axis=0); g=d.abs().sum(axis=1)
    return d.div(g.where(g>0,np.nan),axis=0).fillna(0.0)

# the three canonical signals
SIG={
 "CARRY  (3M interbank differential)"      : carry.rolling(252).mean(),
 "MOMENTUM (12-month, skip last month)"    : (px.shift(21)/px.shift(252)-1.0),
 "VALUE  (5-year reversal, AMP 2013)"      : -(px/px.shift(1260)-1.0),
}
print("="*104)
print("CANONICAL FX FACTORS ,  %s .. %s | %d sessions | %d currencies"
      %(px.index[0].date(),px.index[-1].date(),len(px),px.shape[1])); print("="*104)
print("   %-42s %8s %9s %8s %8s %8s"%("factor","Sharpe","TotRet%","t","p","turn/yr"))
rs={}
for nm,S in SIG.items():
    W=clin(S)
    g,n,t,_=E.apply_positions(W,P,1.0); s=E.summary(n,t); rs[nm]=n
    print("   %-42s %8.3f %9.1f %8.2f %8.4f %8.1f"%(nm,s["Sharpe"],100*s["TotRet"],s["t"],s["p"],s["TurnoverPA"]))

R=pd.DataFrame(rs).dropna()
print("\n   correlation between the factors:")
print(R.corr().round(3).to_string())

z=R.div(R.std())
comb=z.mean(axis=1); comb=comb/comb.std()*R.iloc[:,0].std()
s=E.summary(comb)
print("\n   %-42s %8.3f %9.1f %8.2f %8.4f"%("COMBINED, equal risk",s["Sharpe"],100*s["TotRet"],s["t"],s["p"]))
cv=z[["CARRY  (3M interbank differential)","VALUE  (5-year reversal, AMP 2013)"]].mean(axis=1)
cv=cv/cv.std()*R.iloc[:,0].std(); s=E.summary(cv)
print("   %-42s %8.3f %9.1f %8.2f %8.4f"%("CARRY + VALUE only",s["Sharpe"],100*s["TotRet"],s["t"],s["p"]))

print("\n"+"="*104); print("SUB-PERIODS of the combined portfolio"); print("="*104)
for a,b in [(2002,2007),(2008,2013),(2014,2019),(2020,2026)]:
    m=(comb.index.year>=a)&(comb.index.year<=b); ss=E.summary(comb[m])
    if ss: print("   %d-%d  Sharpe %+6.3f  TotRet %+7.1f%%  t %+5.2f  p %.4f"%(a,b,ss["Sharpe"],100*ss["TotRet"],ss["t"],ss["p"]))

print("\n"+"="*104); print("NOISE CEILING (5 candidates: 3 factors + 2 combinations)"); print("="*104)
rv=ret.values; nn=len(ret); ceil=[]
for seed in range(10):
    rng=np.random.default_rng(880+seed)
    idx=E.stationary_bootstrap_idx(nn,1,rng,mean_block=21)[0]
    pq=pd.DataFrame(rv[idx],index=ret.index,columns=ret.columns)
    Q={"ret":pq}; cx=100*(1+pq).cumprod()
    sigq={"c":carry.rolling(252).mean(),"m":(cx.shift(21)/cx.shift(252)-1.0),"v":-(cx/cx.shift(1260)-1.0)}
    rr={}
    for k,S in sigq.items():
        _,q,_,_=E.apply_positions(clin(S),Q,1.0); rr[k]=q
    Rq=pd.DataFrame(rr).dropna(); zq=Rq.div(Rq.std())
    cands=[E.sharpe(Rq[k]) for k in Rq]+[E.sharpe(zq.mean(axis=1)),E.sharpe(zq[["c","v"]].mean(axis=1))]
    ceil.append(max(cands))
print("   "+"  ".join("%+.2f"%c for c in ceil))
print("   ceiling mean %+.3f | 90th pct %+.3f | max %+.3f"%(np.mean(ceil),np.percentile(ceil,90),max(ceil)))
print("   best observed on real data: %+.3f"%max([E.sharpe(R[c]) for c in R]+[E.sharpe(comb),E.sharpe(cv)]))
