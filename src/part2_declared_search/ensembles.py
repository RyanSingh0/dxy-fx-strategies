"""Two checks the verdict turns on.

First, whether correcting over 144 to 216 configurations is too harsh. If lookback 126
and cardinality 2 were genuinely fixed in advance, meaning six months and the bottom third
of six currencies, then the declared search is only frequency by slice by regime on or
off. The correction is redone over that smaller and more favourable space.

Second, selection is the whole problem, so this avoids selecting. An equal-weight ensemble
over a family of configurations tests the hypothesis rather than a tuned corner and needs
no multiple-testing correction at all, because nothing was chosen."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
exec(open(os.path.join(_HERE, "engine.py")).read())

UNI=[("A. exchange futures, 5 pairs (tradeable)","futures"),
     ("B. futures + FRED SEK, 6 pairs","futures6"),
     ("C. Federal Reserve H.10 spot, 6 pairs","spot")]
DAT={k:universe(k) for _,k in UNI}

print("="*100); print("Q1.  THE AUTHOR-FAVOURABLE SEARCH SPACE"); print("="*100)
print("     lookback fixed at 126 and cardinality fixed at 2 a priori;")
print("     only rebalance frequency (6) x cross-section slice (3) x regime on/off (2) searched.\n")
print("%-42s %6s %11s %11s %8s"%("universe","configs","White RC p","Hansen SPA","RW surv"))
for lab,kind in UNI:
    R,lvl,dxr=DAT[kind]
    cols={}
    for f in FREQS:
        for sl in SLICES:
            for sg in (True,False):
                r=strat(R,lvl,f,sl,2,sg,126)
                if len(r)>500: cols[(f,sl,"reg" if sg else "plain")]=r
    G=pd.DataFrame(cols).dropna()
    p1,p2,s=multiple_testing(G,None,B=5000)
    print("%-42s %6d %11.4f %11.4f %8d %s"%(lab,G.shape[1],p1,p2,len(s),
          ("  <- "+"/".join(map(str,s[0]))) if s else ""))

print("\n"+"="*100); print("Q2.  ENSEMBLES, no configuration is chosen, so nothing needs correcting"); print("="*100)
FAMS={
 "all 'bottom' slices, regime-signed, every freq/card/lookback": lambda c: c[1]=="bottom" and c[3].startswith("reg"),
 "all 'bottom' slices, regime-signed, quarterly only":           lambda c: c[1]=="bottom" and c[3].startswith("reg") and c[0]=="Q",
 "all regime-signed configurations (any slice)":                 lambda c: c[3].startswith("reg"),
 "every configuration in the grid":                              lambda c: True,
}
for lab,kind in UNI:
    R,lvl,dxr=DAT[kind]
    G,_=build_grid(R,lvl)
    b=dxr.reindex(G.index).fillna(0)
    print("\n   %s   (DXY buy-and-hold End %.2f)"%(lab,perf(b)["End"]))
    print("      %-58s %6s %8s %8s %8s %7s"%("ensemble","n cfg","End","Sharpe","t_NW","p"))
    for nm,f in FAMS.items():
        cs=[c for c in G.columns if f(c)]
        if not cs: continue
        e=G[cs].mean(axis=1); t,p=nw(e); pf=perf(e)
        print("      %-58s %6d %8.2f %8.3f %8.2f %7.4f"%(nm,len(cs),pf["End"],pf["Sharpe"],t,p))
    # ensemble alpha vs factors
    cs=[c for c in G.columns if FAMS["all 'bottom' slices, regime-signed, every freq/card/lookback"](c)]
    e=G[cs].mean(axis=1); m=attribution(e,factors(R,dxr))
    print("      -> factor alpha of that ensemble: %+.2f%%/yr (t %+.2f, p %.4f); betas DOLLAR %+.2f CARRY %+.2f XSMOM %+.2f"
          %(m.params["const"]*TD*100,m.tvalues["const"],m.pvalues["const"],
            m.params["DOLLAR"],m.params["CARRY"],m.params["XSMOM"]))
    # walk-forward the ensemble (nothing to select, so this is just the OOS half)
    yr=e.index.year
    for nmm,msk in [("2011-2018",yr<=2018),("2019-2025",yr>=2019)]:
        t,p=nw(e[msk]); print("      -> %s: End %7.2f Sharpe %+6.3f t %+5.2f p %.4f"%(nmm,perf(e[msk])["End"],sr(e[msk]),t,p))
