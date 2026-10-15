"""Revision 2: the np.where correction in Section 10, before and after.

The committed rule built the regime sign with np.where(mom < 0, 1, -1). Over the first
LB sessions the momentum is NaN, np.where maps NaN to -1, and the book traded as though
the dollar were rising. The fix holds no position in a signed configuration until its
own lookback exists. Unsigned configurations never use the regime and are unchanged.

I rerun every Section 10 quantity twice, with nan_as_rising=True (as committed) and
False (as corrected), and nothing else changed, so the effect of this one defect can be
read off directly. Settings match run_battery.py and ensembles.py (B = 3,000 on the full
grid, 5,000 on the 36-configuration subset, seed 20260828, mean block 21)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "part2_declared_search"))
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np, pandas as pd
import paths
import engine as G

OUT = os.path.join(str(paths.RESULTS), "revision")
UNI = [("futures", "A. exchange futures, 5 pairs"), ("futures6", "B. futures + FRED SEK, 6 pairs"),
       ("spot", "C. Federal Reserve H.10 spot, 6 pairs")]
LAB = {True: "committed (NaN -> -1)", False: "corrected (no position)"}
res = {}

def headline(R, lvl, bug, **kw):
    a = dict(freq="Q", slice_="bottom", card=2, signed=True, LB=126); a.update(kw)
    return G.strat(R, lvl, nan_as_rising=bug, **a)

print("=" * 100)
print("1. HEADLINE CONFIGURATION (quarterly, bottom 2, 126-session regime)")
print("=" * 100)
for kind, lab in UNI:
    R, lvl, dxr = G.universe(kind)
    for bug in (True, False):
        r = headline(R, lvl, bug); t, p = G.nw(r); pf = G.perf(r)
        print("   %-32s %-24s first day %s  n %d  End %7.2f  Sharpe %.3f  t %.2f  p %.4f"
              % (lab, LAB[bug], r.index[0].date(), len(r), pf["End"], pf["Sharpe"], t, p))
        res.setdefault(kind, {})[str(bug)] = dict(first=str(r.index[0].date()), n=len(r), sharpe=pf["Sharpe"], t=t, p=p)

print("\n" + "=" * 100)
print("2. FULL DECLARED GRID (multiple-testing window = sessions on which every configuration trades)")
print("=" * 100)
for kind, lab in UNI:
    R, lvl, dxr = G.universe(kind)
    for bug in (True, False):
        Gr, _ = G.build_grid(R, lvl, nan_as_rising=bug)
        s = Gr.apply(G.sr); b = dxr.reindex(Gr.index).fillna(0)
        rc0, spa0, rw0 = G.multiple_testing(Gr, None, B=3000)
        rc1, spa1, rw1 = G.multiple_testing(Gr, b, B=3000)
        r = headline(R, lvl, bug); d, e = G.dsr(r, s.values, Gr.shape[1])
        hd = ("Q", "bottom", 2, "reg126")
        rank = 1 + int((s > s[hd]).sum())
        print("   %-32s %-24s configs %d  window %s..%s (%d)  best %.3f  headline rank %d"
              % (lab, LAB[bug], Gr.shape[1], Gr.index[0].date(), Gr.index[-1].date(), len(Gr), s.max(), rank))
        print("   %-32s %-24s null zero: RC %.4f SPA %.4f RW %d | null long dollar: RC %.4f SPA %.4f RW %d | E[max] %.3f DSR %.3f"
              % ("", "", rc0, spa0, len(rw0), rc1, spa1, len(rw1), e, d))
        res[kind][str(bug)].update(configs=Gr.shape[1], window=[str(Gr.index[0].date()), str(Gr.index[-1].date()), len(Gr)],
                                   best=float(s.max()), rank=rank, rc0=rc0, spa0=spa0, rw0=len(rw0),
                                   rc1=rc1, spa1=spa1, rw1=len(rw1), emax=e, dsr=d)

print("\n" + "=" * 100)
print("3. ONE-AT-A-TIME SWEEPS ON THE TRADEABLE FUTURES PANEL")
print("=" * 100)
R, lvl, dxr = G.universe("futures")
for bug in (True, False):
    lbs = ["none %+.3f" % G.sr(headline(R, lvl, bug, signed=False))]
    lbs += ["%d %+.3f" % (lb, G.sr(headline(R, lvl, bug, LB=lb))) for lb in (63, 126, 252)]
    fq = ["%s %+.3f" % (f, G.sr(headline(R, lvl, bug, freq=f))) for f in G.FREQS]
    print("   %-24s lookback: %s" % (LAB[bug], " | ".join(lbs)))
    print("   %-24s frequency: %s" % ("", " | ".join(fq)))

print("\n" + "=" * 100)
print("4. EQUAL-WEIGHT ENSEMBLES AND THE 36-CONFIGURATION AUTHOR-FAVOURABLE SUBSET")
print("=" * 100)
for kind, lab in UNI:
    R, lvl, dxr = G.universe(kind)
    for bug in (True, False):
        Gr, _ = G.build_grid(R, lvl, nan_as_rising=bug)
        e1 = Gr[[c for c in Gr.columns if c[1] == "bottom" and c[3].startswith("reg")]].mean(axis=1)
        e2 = Gr[[c for c in Gr.columns if c[3].startswith("reg")]].mean(axis=1)
        cols = {}
        for f in G.FREQS:
            for sl in G.SLICES:
                for sg in (True, False):
                    r = G.strat(R, lvl, f, sl, 2, sg, 126, nan_as_rising=bug)
                    if len(r) > 500: cols[(f, sl, "reg" if sg else "plain")] = r
        Gs = pd.DataFrame(cols).dropna()
        p1, p2, sv = G.multiple_testing(Gs, None, B=5000)
        t1, q1 = G.nw(e1); t2, q2 = G.nw(e2)
        print("   %-32s %-24s bottom regime-signed %.3f (p %.2f) | all regime-signed %.3f (p %.2f) | subset of %d: RC %.4f SPA %.4f RW %d"
              % (lab, LAB[bug], G.sr(e1), q1, G.sr(e2), q2, Gs.shape[1], p1, p2, len(sv)))
        res[kind][str(bug)].update(ens_bottom=(G.sr(e1), q1), ens_all=(G.sr(e2), q2), subset=(Gs.shape[1], p1, p2, len(sv)))

json.dump(res, open(os.path.join(OUT, "npwhere_effect.json"), "w"), indent=1, default=float)
