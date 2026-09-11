import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths
import numpy as np, pandas as pd, warnings
from scipy import stats as st
warnings.filterwarnings("ignore")
exec(open(os.path.join(_HERE, "engine.py")).read())

print("=" * 104)
print("0.  BUG FIX CONFIRMATION")
print("=" * 104)
_i = pd.date_range("2011-01-03", "2025-12-30", freq="B")
print("   pandas to_period('Q')  -> %d periods" % len(set(_i.to_period("Q").astype(str))))
print("   pandas to_period('2Q') -> %d periods   <- identical, this was the duplicated row"
      % len(set(_i.to_period("2Q").astype(str))))
print("   explicit semiannual    -> %d periods   <- the fix" % len(set(period_key(_i, "S"))))

UNI = [("A. exchange futures, 5 pairs (fully tradeable)", "futures"),
       ("B. exchange futures + FRED SEK, 6 pairs", "futures6"),
       ("C. Federal Reserve H.10 spot, 6 pairs", "spot")]
RES = {}
for lab, kind in UNI:
    R, lvl, dxr = universe(kind)
    G, dropped = build_grid(R, lvl, verbose=False)
    RES[kind] = (R, lvl, dxr, G, dropped)

print("\n" + "=" * 104)
print("1.  THE SEARCH SPACE, DECLARED IN FULL")
print("=" * 104)
G0 = RES["futures"][3]
print("   free parameters: rebalance frequency (5) x cross-section slice (3) x cardinality (3)")
print("                    x regime filter on/off and when on, momentum lookback (3)")
print("   configurations evaluated: %d   (the original implementation reported 6 of these)" % G0.shape[1])
for _lab, _kind in UNI:
    _G, _dr = RES[_kind][3], RES[_kind][4]
    print("   %-46s kept %3d, dropped %3d" % (_lab, _G.shape[1], len(_dr)))
    _rz = {}
    for _k, _w in _dr: _rz.setdefault(_w.split(":")[0], 0); _rz[_w.split(":")[0]] += 1
    for _w, _c in _rz.items(): print("        %3d - %s" % (_c, _w))
print("   sessions: %d   window %s .. %s" % (len(G0), G0.index[0].date(), G0.index[-1].date()))

print("\n" + "=" * 104)
print("2.  HEADLINE CONFIGURATION ACROSS THE THREE UNIVERSES")
print("     quarterly, bottom-2 by prior-quarter return, signed by the 126-day DXY regime")
print("=" * 104)
print("%-46s %8s %7s %7s %8s %7s %7s %8s" % ("universe", "End", "Sharpe", "Sortino", "MaxDD%", "t_NW", "p", "DXY"))
for lab, kind in UNI:
    R, lvl, dxr, G, _dr = RES[kind]
    r = strat(R, lvl); pf = perf(r); t, p = nw(r)
    print("%-46s %8.2f %7.3f %7.3f %8.1f %7.2f %7.4f %8.2f"
          % (lab, pf["End"], pf["Sharpe"], pf["Sortino"], 100 * pf["MaxDD"], t, p,
             perf(dxr.reindex(r.index).fillna(0))["End"]))

print("\n" + "=" * 104)
print("3.  IS 'QUARTERLY, BOTTOM-2, 126-DAY' A LUCKY CORNER? THE WHOLE GRID")
print("=" * 104)
for lab, kind in UNI:
    G = RES[kind][3]
    s = G.apply(sr).sort_values(ascending=False)
    print("\n   %s" % lab)
    print("      best %.3f | 90th pct %.3f | median %.3f | worst %.3f"
          % (s.max(), s.quantile(.9), s.median(), s.min()))
    print("      top 5: " + " | ".join("%s %.3f" % ("/".join(map(str, i)), v) for i, v in s.head(5).items()))
    hd = [c for c in G.columns if c[0] == "Q" and c[1] == "bottom" and c[2] == 2 and c[3] == "reg126"][0]
    print("      rank of the paper's configuration: %d of %d" % (1 + int((s > s[hd]).sum()), len(s)))

print("\n" + "=" * 104)
print("4.  LOOKBACK AND CARDINALITY SENSITIVITY (futures, quarterly, bottom slice)")
print("=" * 104)
R, lvl, dxr, G, _dr = RES["futures"]
print("%-10s %-10s %9s %9s %8s" % ("lookback", "cardinality", "End", "Sharpe", "p_NW"))
for lb in [63, 126, 252]:
    for k in [1, 2, 3]:
        r = strat(R, lvl, "Q", "bottom", k, True, lb); t, p = nw(r)
        print("%-10d %-10d %9.2f %9.3f %8.4f" % (lb, k, perf(r)["End"], sr(r), p))
print("\n   frequency sensitivity (bottom-2, 126-day):")
for f in FREQS:
    r = strat(R, lvl, f, "bottom", 2, True, 126)
    if len(r) < 100: print("      %-3s (too few sessions)" % f); continue
    t, p = nw(r)
    print("      %-3s End %8.2f  Sharpe %+6.3f  p %.4f" % (f, perf(r)["End"], sr(r), p))

print("\n   momentum-calendar convention (an alignment choice the paper never states):")
print("      the DXY futures file has %d sessions; the five-pair panel has %d."
      % (len(lvl), len(R)))
for mc, note in [("universe", "lookback counted on the traded panel, the original convention"),
                 ("native", "lookback counted on the DXY series' own calendar")]:
    r = strat(R, lvl, "Q", "bottom", 2, True, 126, mom_calendar=mc); t, p = nw(r)
    print("      %-9s End %8.2f  Sharpe %+6.3f  p %.4f   <- %s" % (mc, perf(r)["End"], sr(r), p, note))

print("\n" + "=" * 104)
print("5.  MULTIPLE TESTING OVER THE FULL %d-CONFIGURATION SEARCH" % G0.shape[1])
print("=" * 104)
print("   The book is self-financing long/short, so the null is zero excess return.")
print("   The original implementation tested against DXY buy-and-hold; both are shown.\n")
print("%-46s %-22s %9s %9s %8s" % ("universe", "null", "White RC p", "Hansen SPA", "RW surv"))
for lab, kind in UNI:
    R, lvl, dxr, G, _dr = RES[kind]
    b = dxr.reindex(G.index).fillna(0)
    for nl, bb in [("zero (correct)", None), ("DXY buy-and-hold", b)]:
        p1, p2, s = multiple_testing(G, bb, B=3000)
        print("%-46s %-22s %9.4f %9.4f %8d %s"
              % (lab if nl.startswith("zero") else "", nl, p1, p2, len(s),
                 ("  <- " + "/".join(map(str, s[0]))) if s else ""))

print("\n" + "=" * 104)
print("6.  DEFLATED SHARPE OVER THE FULL SEARCH  (a confidence level; must exceed 0.95)")
print("=" * 104)
for lab, kind in UNI:
    R, lvl, dxr, G, _dr = RES[kind]
    r = strat(R, lvl); allsr = G.apply(sr).values
    d, sr0 = dsr(r, allsr, G.shape[1])
    print("   %-46s PSR(0) %.4f | expected max Sharpe under the null %+.3f | DSR %.4f %s"
          % (lab, psr(r), sr0, d, "PASS" if d > 0.95 else "fail"))

print("\n" + "=" * 104)
print("7.  BACKTEST OVERFITTING (CSCV)")
print("=" * 104)
for lab, kind in UNI:
    G = RES[kind][3]
    for S in (12, 16):
        p, med, k = pbo(G, S)
        print("   %-46s S=%2d  PBO %.3f  median OOS rank of IS winner %.0f/%d" % (lab if S == 12 else "", S, p, med, k))

print("\n" + "=" * 104)
print("8.  FACTOR ATTRIBUTION, is this just dollar beta, carry, or plain FX momentum?")
print("=" * 104)
for lab, kind in UNI:
    R, lvl, dxr, G, _dr = RES[kind]
    r = strat(R, lvl); F = factors(R, dxr); m = attribution(r, F)
    a = m.params["const"] * TD * 100
    print("\n   %s" % lab)
    print("      annualised alpha %+.2f%%  (t = %+.2f, p = %.4f)   R2 = %.3f"
          % (a, m.tvalues["const"], m.pvalues["const"], m.rsquared))
    for f in ["DOLLAR", "CARRY", "XSMOM"]:
        print("      beta %-8s %+7.3f  (t = %+.2f, p = %.4f)" % (f, m.params[f], m.tvalues[f], m.pvalues[f]))

print("\n" + "=" * 104)
print("9.  HOLDOUT AND WALK-FORWARD OVER THE FULL GRID")
print("=" * 104)
for lab, kind in UNI:
    R, lvl, dxr, G, _dr = RES[kind]
    b = dxr.reindex(G.index).fillna(0); yr = G.index.year
    IS, OS = yr <= 2018, yr >= 2019
    isr, osr = G[IS].apply(sr), G[OS].apply(sr)
    pick = isr.idxmax(); t, p = nw(G[OS][pick])
    print("\n   %s" % lab)
    print("      single selection on 2011-2018 -> %s" % "/".join(map(str, pick)))
    print("         IS Sharpe %+6.3f -> OOS Sharpe %+6.3f  End %7.2f  p %.4f  OOS rank %d/%d"
          % (isr[pick], osr[pick], perf(G[OS][pick])["End"], p, 1 + int((osr > osr[pick]).sum()), G.shape[1]))
    oos = []; picks = []
    for y in sorted(set(yr)):
        tr = G[G.index.year < y]
        if len(tr) < 750: continue
        pk = tr.apply(sr).idxmax(); seg = G[G.index.year == y][pk]
        oos.append(seg); picks.append(pk)
    oos = pd.concat(oos).sort_index(); bo = b.reindex(oos.index).fillna(0)
    t, p = nw(oos)
    st_ = pd.Series(["/".join(map(str, p_)) for p_ in picks]).value_counts()
    print("      annual re-selection, %d OOS years: End %7.2f Sharpe %+6.3f t %+5.2f p %.4f | DXY End %6.2f"
          % (len(picks), 100 * (1 + oos).prod(), sr(oos), t, p, 100 * (1 + bo).prod()))
    print("         configuration chosen: %s" % ", ".join("%s x%d" % (a_, b_) for a_, b_ in st_.head(3).items()))

print("\n" + "=" * 104)
print("10. COSTS AND CAPACITY (futures, headline)")
print("=" * 104)
R, lvl, dxr, G, _dr = RES["futures"]
bE = perf(dxr.reindex(G.index).fillna(0))["End"]
print("   DXY futures buy-and-hold End %.2f" % bE)
for c in (0, 1, 2, 5, 10, 20, 50, 100):
    r = strat(R, lvl, "Q", "bottom", 2, True, 126, cost_bp=c); pf = perf(r); t, p = nw(r)
    print("      %3d bp/rebalance: End %7.2f  Sharpe %+6.3f  p %.4f  %s"
          % (c, pf["End"], pf["Sharpe"], p, "beats B&H" if pf["End"] > bE else "below B&H"))
lo, hi = 0.0, 2000.0
for _ in range(40):
    mid = (lo + hi) / 2
    if sr(strat(R, lvl, "Q", "bottom", 2, True, 126, cost_bp=mid)) > 0: lo = mid
    else: hi = mid
print("   break-even cost (Sharpe -> 0): %.0f bp per rebalance" % lo)

print("\n" + "=" * 104)
print("11. RISK PROFILE AND CRISIS BEHAVIOUR (futures, headline)")
print("=" * 104)
r = strat(R, lvl); pf = perf(r); b = dxr.reindex(r.index).fillna(0)
print("   Sharpe %.3f | Sortino %.3f | ann vol %.1f%% | MaxDD %.1f%% | skew %+.2f | kurtosis %.1f"
      % (pf["Sharpe"], pf["Sortino"], 100 * pf["Vol"], 100 * pf["MaxDD"], pf["Skew"], pf["Kurt"]))
print("   correlation with the dollar: %+.3f" % r.corr(b))
for lab, a, z in [("2015 CHF de-peg", "2015-01-01", "2015-03-31"), ("2020 COVID", "2020-02-01", "2020-04-30"),
                  ("2022 dollar surge", "2022-01-01", "2022-12-31"), ("2025", "2025-01-01", "2025-12-31")]:
    m = (r.index >= a) & (r.index <= z)
    if m.sum() < 5: continue
    print("      %-20s strategy %+7.2f%%   dollar %+7.2f%%" % (lab, 100 * ((1 + r[m]).prod() - 1), 100 * ((1 + b[m]).prod() - 1)))
