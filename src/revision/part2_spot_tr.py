"""Revision: universe C of the declared search (Federal Reserve spot) recomputed on total
returns, i.e. spot change plus the three-month interbank differential, so that the spot
universe is comparable with the futures universes, whose returns embed the differential."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "part2_declared_search"))
import pandas as pd
from engine import RATES, RCOL, universe, build_grid, strat, nw, sr, multiple_testing, dsr
R, lvl, dxr = universe("spot")
rt = RATES[[RCOL[c] for c in R.columns] + ["IR3TIB01USM156N"]].reindex(R.index.union(RATES.index)).ffill().reindex(R.index)
diff = pd.DataFrame({c: (rt[RCOL[c]] - rt["IR3TIB01USM156N"]) / 100.0 for c in R.columns})
Rtr = R + diff.shift(1).bfill() / 252.0
for lab, RR in [("spot change only (as in Table)", R), ("total return", Rtr)]:
    G, dr = build_grid(RR, lvl)
    r = strat(RR, lvl); t, p = nw(r)
    s = G.apply(sr)
    rc, spa, rw = multiple_testing(G, None, B=3000)
    d_, sr0 = dsr(r, s.values, G.shape[1])
    print("%-32s configs %d | headline Sharpe %.3f t %.2f p %.4f | best %.3f | RC %.4f SPA %.4f RW %d | DSR %.3f (E[max] %.3f)"
          % (lab, G.shape[1], sr(r), t, p, s.max(), rc, spa, len(rw), d_, sr0))
