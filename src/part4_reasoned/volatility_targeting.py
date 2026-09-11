"""Volatility targeting, the one claim this data supports without qualification.

Direction R^2 is 0.0000 and volatility R^2 is 0.97, so the only claim that follows is that
volatility targeting improves risk-adjusted outcomes without forecasting direction. Tested
on the long sample, on the dollar and on each pair, against the honest null that
vol-scaling is just leverage."""
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
px=px.loc["1999-01-05":]
ret=px.pct_change().dropna()
DXY=(50.14348112*A.DEXUSEU**-0.576*A.DEXJPUS**0.136*A.DEXUSUK**-0.119
     *A.DEXCAUS**0.091*A.DEXSDUS**0.042*A.DEXSZUS**0.036).reindex(px.index).ffill()
dxr=DXY.pct_change().reindex(ret.index).fillna(0.0)

def volmanage(r,target=0.08,L=21,cap=3.0):
    fv=r.rolling(L).std()*np.sqrt(TD)
    lev=(target/fv.replace(0,np.nan)).clip(upper=cap).shift(1)
    return (lev*r).fillna(0.0),lev

print("="*100)
print("VOLATILITY TARGETING ,  %s .. %s (%d sessions, %d years)"
      %(ret.index[0].date(),ret.index[-1].date(),len(ret),len(ret)//252))
print("="*100)
print("   Does scaling by forecast vol raise the Sharpe ratio? Sharpe is scale-invariant,")
print("   so any improvement must come from TIMING the exposure, not from leverage.\n")
print("   %-12s %10s %10s %9s %10s %10s"%("series","raw SR","vol-mgd SR","change","raw MaxDD","vmg MaxDD"))
rows=[]
for nm,r in [("DXY",dxr)]+[(c,ret[c]) for c in ret.columns]:
    v,lev=volmanage(r)
    a,b=E.summary(r),E.summary(v)
    rows.append(b["Sharpe"]-a["Sharpe"])
    print("   %-12s %10.3f %10.3f %+9.3f %10.1f%% %9.1f%%"
          %(nm,a["Sharpe"],b["Sharpe"],b["Sharpe"]-a["Sharpe"],100*a["MaxDD"],100*b["MaxDD"]))
print("\n   mean Sharpe change across the seven series: %+.3f"%np.mean(rows))
print("   series improved: %d of %d"%(sum(1 for x in rows if x>0),len(rows)))

print("\n   the same on the 2011-2025 futures window, for comparison with earlier results:")
m=(dxr.index.year>=2011)&(dxr.index.year<=2025)
v,_=volmanage(dxr)
print("      dollar raw Sharpe %+.3f -> vol-managed %+.3f"%(E.sharpe(dxr[m]),E.sharpe(v[m])))

print("\n"+"="*100); print("IS THE IMPROVEMENT REAL, OR AN ARTEFACT OF FAT TAILS?"); print("="*100)
print("   Null: vol targeting adds nothing. Bootstrap the return series in blocks (which")
print("   destroys volatility clustering but keeps the marginal distribution) and re-apply.")
nn=len(dxr); gains=[]
for seed in range(20):
    rng=np.random.default_rng(50+seed)
    idx=E.stationary_bootstrap_idx(nn,1,rng,mean_block=21)[0]
    rb=pd.Series(dxr.values[idx],index=dxr.index)
    vb,_=volmanage(rb)
    gains.append(E.sharpe(vb)-E.sharpe(rb))
obs=E.sharpe(volmanage(dxr)[0])-E.sharpe(dxr)
print("   observed Sharpe gain on the dollar        : %+.3f"%obs)
print("   gain on block-bootstrapped returns        : mean %+.3f, sd %.3f, 90th pct %+.3f"
      %(np.mean(gains),np.std(gains),np.percentile(gains,90)))
print("   -> %s"%("the gain exceeds what the null produces" if obs>np.percentile(gains,90)
                  else "the gain is within what the null produces"))

print("\n"+"="*100); print("WHAT IT DOES AND DOES NOT DO"); print("="*100)
v,lev=volmanage(dxr)
print("   mean leverage %.2f | leverage range %.2f - %.2f"%(lev.mean(),lev.min(),lev.max()))
print("   realised vol: raw %.2f%% -> managed %.2f%%  (target 8.00%%)"
      %(100*dxr.std()*np.sqrt(TD),100*v.std()*np.sqrt(TD)))
print("   vol of realised 21d vol: raw %.3f -> managed %.3f  (this is the real benefit)"
      %( (dxr.rolling(21).std()*np.sqrt(TD)).std(), (v.rolling(21).std()*np.sqrt(TD)).std()))
print("   worst day: raw %.2f%% -> managed %.2f%%"%(100*dxr.min(),100*v.min()))
print("   total return: raw %+.1f%% -> managed %+.1f%%"%(100*((1+dxr).prod()-1),100*((1+v).prod()-1)))
