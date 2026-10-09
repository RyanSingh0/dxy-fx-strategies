"""Revision: properties of the open-location binomial screen.

  A. alternative definitions: exact equality vs proximity to the extremes (fraction of range)
  B. Monte Carlo size under microstructure: tick discreteness, bid-ask bounce, fat tails,
     stochastic volatility, drift, and a deliberately asymmetric (bid-side) opening print
  C. false-alarm rate on clean exchange data: six futures series, full sample and
     non-overlapping three-year windows
  D. power: defects injected into the clean exchange DX series at controlled rates
  E. alternative session aggregation (weekly, monthly bars)
  F. year-by-year counts on the free series"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd, json
from scipy import stats as st
import paths

OUT = os.path.join(str(paths.RESULTS), "revision"); os.makedirs(OUT, exist_ok=True)
res = {}

def load_fut(sym):
    d = pd.read_csv(str(paths.FUTURES / (sym + ".csv")), skipfooter=1, engine="python")
    return pd.DataFrame({"Date": pd.to_datetime(d["Time"]), "Sym": d["Symbol"].astype(str),
        "O": pd.to_numeric(d["Open"]), "H": pd.to_numeric(d["High"]), "L": pd.to_numeric(d["Low"]),
        "C": pd.to_numeric(d["Latest"])}).sort_values("Date").reset_index(drop=True)

def load_free():
    d = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"])
    out = {}
    for p in ["DXY", "EURUSD", "GBPUSD", "JPYUSD", "CADUSD", "CHFUSD", "SEKUSD"]:
        x = d[["Date", p + "_Open", p + "_High", p + "_Low", p + "_Close"]].copy()
        x.columns = ["Date", "O", "H", "L", "C"]
        out[p] = x.dropna().reset_index(drop=True)
    return out

def counts(O, H, L, delta=0.0):
    O, H, L = map(lambda a: np.asarray(a, float), (O, H, L))
    k = np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & (H > L)
    O, H, L = O[k], H[k], L[k]
    if delta == 0.0:
        lo = np.isclose(O, L, rtol=0, atol=1e-9); hi = np.isclose(O, H, rtol=0, atol=1e-9)
    else:
        loc = (O - L) / (H - L)
        lo = loc <= delta; hi = loc >= 1 - delta
    nl, nh = int(lo.sum()), int(hi.sum())
    p = float(st.binomtest(nl, nl + nh, 0.5).pvalue) if nl + nh > 0 else np.nan
    return nl, nh, p, int(k.sum())

FREE = load_free()
EX = {s: load_fut(s) for s in ["DX", "E6", "B6", "D6", "J6", "S6"]}
NAMES = {"DX": "Dollar index futures", "E6": "Euro futures", "B6": "Sterling futures",
         "D6": "Canadian dollar futures", "J6": "Yen futures", "S6": "Swiss franc futures"}

# ------------------------------------------------------------------ A
print("A. ALTERNATIVE DEFINITIONS (delta = proximity to extreme as a fraction of the range)")
deltas = [0.0, 0.01, 0.02, 0.05, 0.10]
A = {}
series = [("DX exch", EX["DX"])] + [(s + " exch", EX[s]) for s in ["E6", "B6", "D6", "J6", "S6"]] + \
         [("DXY free", FREE["DXY"])] + [(p + " free", FREE[p]) for p in ["EURUSD", "GBPUSD", "JPYUSD", "CADUSD", "CHFUSD", "SEKUSD"]]
print("%-12s" % "series" + "".join("   d=%-4s lo/hi        p" % d for d in deltas))
for lab, d in series:
    row = []
    for de in deltas:
        nl, nh, p, n = counts(d.O, d.H, d.L, de); row.append((nl, nh, p, n))
    A[lab] = row
    print("%-12s" % lab + "".join("  %4d/%-4d %10.2e" % (r[0], r[1], r[2]) for r in row))
res["A"] = A

# ------------------------------------------------------------------ F
print("\nF. YEAR BY YEAR, free DXY and exchange DX (exact definition)")
F = {}
for lab, d in [("free", FREE["DXY"]), ("exch", EX["DX"])]:
    y = d.Date.dt.year
    for yr in sorted(y.unique()):
        m = (y == yr).values
        nl, nh, p, n = counts(d.O[m], d.H[m], d.L[m])
        F.setdefault(lab, {})[int(yr)] = (nl, nh, p, n)
for yr in sorted(F["free"]):
    a = F["free"][yr]; b = F["exch"].get(yr, (0, 0, np.nan, 0))
    print("  %d  free %3d/%-3d p=%.2e | exch %3d/%-3d p=%.3f" % (yr, a[0], a[1], a[2], b[0], b[1], b[2]))
res["F"] = F
nl, nh, p, n = counts(FREE["DXY"].O[FREE["DXY"].Date.dt.year != 2013], FREE["DXY"].H[FREE["DXY"].Date.dt.year != 2013],
                      FREE["DXY"].L[FREE["DXY"].Date.dt.year != 2013])
print("  free ex-2013: %d/%d p=%.2e" % (nl, nh, p)); res["free_ex2013"] = (nl, nh, p)

# ------------------------------------------------------------------ C
print("\nC. FALSE ALARMS ON CLEAN EXCHANGE DATA")
C = {"full": {}, "windows": []}
for s, d in EX.items():
    nl, nh, p, n = counts(d.O, d.H, d.L); C["full"][s] = (nl, nh, p, n)
    y = d.Date.dt.year
    for a in range(2011, 2026, 3):
        m = ((y >= a) & (y <= a + 2)).values
        nl, nh, p, n = counts(d.O[m], d.H[m], d.L[m])
        C["windows"].append((s, a, nl, nh, p))
pw = np.array([w[4] for w in C["windows"]])
print("  full sample p:", {k: round(v[2], 4) for k, v in C["full"].items()})
print("  %d three-year windows: share p<0.05 = %.3f, p<0.01 = %.3f, min p = %.4f" %
      (len(pw), np.mean(pw < 0.05), np.mean(pw < 0.01), pw.min()))
C["share05"] = float(np.mean(pw < 0.05)); C["share01"] = float(np.mean(pw < 0.01))
# same windows on free DXY
y = FREE["DXY"].Date.dt.year; fw = []
for a in range(2011, 2026, 3):
    m = ((y >= a) & (y <= a + 2)).values
    d = FREE["DXY"]; nl, nh, p, n = counts(d.O[m], d.H[m], d.L[m]); fw.append((a, nl, nh, p))
print("  free DXY windows:", fw); C["free_windows"] = fw
res["C"] = C

# ------------------------------------------------------------------ B
print("\nB. MONTE CARLO SIZE UNDER MICROSTRUCTURE (n = 3,800 sessions, 400 samples each)")
rng = np.random.default_rng(20261006)
dx = EX["DX"]
rng_med = float((dx.H - dx.L).median()); px = round(float(dx.C.median()) / 0.005) * 0.005  # start on the tick grid
sd_day = float(np.std(np.log(dx.C).diff().dropna()))
tick = 0.005
def simulate(n_sess=3800, K=300, reps=1000, tickrel=None, bounce=0.0, tdf=None, sv=False,
             drift=0.0, open_bias=0.0, chunk=50):
    """Returns rejection rates at 5% and 1%, mean share of extremes, mean P(low|extreme)."""
    rej5 = rej1 = 0; share = []; plow = []
    for c0 in range(0, reps, chunk):
        b = min(chunk, reps - c0)
        vol = sd_day / np.sqrt(K)
        if sv:
            vol = vol * np.exp(0.4 * rng.standard_normal((b, n_sess, 1)) - 0.08)
        z = rng.standard_t(tdf, (b, n_sess, K)) / np.sqrt(tdf / (tdf - 2)) if tdf else rng.standard_normal((b, n_sess, K))
        inc = z * vol + drift / K
        path = np.concatenate([np.zeros((b, n_sess, 1)), np.cumsum(inc, axis=2)], axis=2)
        # latent efficient price: starts at a uniformly random point inside one tick, so that
        # neither the bid nor the ask is favoured at the open
        p = (px + tick * rng.random((b, n_sess, 1))) * np.exp(path)
        if bounce > 0:   # every print is the bid or the ask around the latent mid
            bid = np.floor(p / tick + 1e-9) * tick
            p = np.where(rng.random(p.shape) < 0.5, bid, bid + tick)
        elif tickrel:
            p = np.round(p / tick) * tick
        O = p[:, :, 0] - open_bias * tick
        H = np.maximum(p.max(axis=2), O); L = np.minimum(p.min(axis=2), O)
        valid = H > L
        lo = np.isclose(O, L, atol=1e-9) & valid; hi = np.isclose(O, H, atol=1e-9) & valid
        nl = lo.sum(axis=1); nh = hi.sum(axis=1)
        for i in range(b):
            pv = st.binomtest(int(nl[i]), int(nl[i] + nh[i]), 0.5).pvalue if nl[i] + nh[i] > 0 else 1
            rej5 += pv < 0.05; rej1 += pv < 0.01
        share.append(((nl + nh) / valid.sum(axis=1)).mean()); plow.append((nl / np.maximum(nl + nh, 1)).mean())
    return dict(rej5=rej5 / reps, rej1=rej1 / reps, share_ext=float(np.mean(share)), p_low=float(np.mean(plow)))
mu_day = float(np.log(dx.C).diff().dropna().mean())
scen = [("Gaussian random walk, continuous prices", dict()),
        ("+ tick discreteness (DX tick 0.005)", dict(tickrel=True)),
        ("+ bid-ask bounce (prints at bid or ask, 1-tick spread)", dict(tickrel=True, bounce=1.0)),
        ("+ fat tails (t, 3 df) and ticks", dict(tickrel=True, tdf=3)),
        ("+ stochastic volatility and ticks", dict(tickrel=True, sv=True)),
        ("+ drift at sample mean and ticks", dict(tickrel=True, drift=mu_day)),
        ("+ drift at 10x sample mean and ticks", dict(tickrel=True, drift=10 * mu_day)),
        ("few intraday steps (K=20), ticks", dict(tickrel=True, K=20)),
        ("asymmetric open: recorded 0.5 tick below", dict(tickrel=True, open_bias=0.5)),
        ("asymmetric open: recorded 1 tick below", dict(tickrel=True, open_bias=1.0))]
B = {}
for lab, kw in scen:
    r = simulate(reps=400, **kw); B[lab] = r
    print("  %-46s size5 %.3f size1 %.3f share-at-extreme %.3f P(low|ext) %.3f" % (lab, r["rej5"], r["rej1"], r["share_ext"], r["p_low"]))
res["B"] = B
res["B_meta"] = dict(sd_day=sd_day, mu_day=mu_day, px=px, rng_med=rng_med)

# ------------------------------------------------------------------ D
print("\nD. POWER: DEFECTS INJECTED INTO THE CLEAN EXCHANGE DX SERIES (500 injections per cell)")
def nw_t(x, L=8):
    x = np.asarray(x, float); x = x - x.mean() if False else x
    n = len(x); m = x.mean(); e = x - m
    g0 = e @ e / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return m / np.sqrt(s / n)
base = EX["DX"].copy()
O0, H0, L0, C0 = (base[c].values.astype(float) for c in "OHLC")
Cp = np.concatenate([[np.nan], C0[:-1]])
rg = H0 - L0
def inject(kind, rate, rng):
    O, H, L = O0.copy(), H0.copy(), L0.copy()
    m = rng.random(len(O)) < rate; m[0] = False
    if kind == "down":       # open displaced downward by up to 30% of the range, vendor repairs the low
        O[m] = O[m] - rng.uniform(0, 0.3, m.sum()) * rg[m]
    elif kind == "atlow":    # misfielded extreme: the session low is written into the open field
        O[m] = L[m]
    elif kind == "stale":    # stale print: open = previous close
        O[m] = Cp[m]
    elif kind == "stale_up": # stale print only when the market gapped up overnight (asymmetric staleness)
        mm = m & (O0 > Cp); O[mm] = Cp[mm]
    elif kind == "outside":  # open printed outside the true range: 71% below the low, 29% above the high
        dn = rng.random(m.sum()) < 204 / 286
        u = rng.uniform(0, 0.15, m.sum()) * rg[m]
        O[m] = np.where(dn, L[m] - u, H[m] + u)
    elif kind == "noise":    # symmetric recording error of 20% of the range
        O[m] = O[m] + 0.2 * rg[m] * rng.standard_normal(m.sum())
    H = np.maximum(H, O); L = np.minimum(L, O)
    O = np.round(O / 0.005) * 0.005
    H = np.maximum(H, O); L = np.minimum(L, O)
    return O, H, L
D = {}
rates = [0.005, 0.01, 0.02, 0.035, 0.05]
kinds = [("down", "open displaced downward, low repaired"), ("atlow", "session low written into open field"),
         ("stale", "stale open = previous close"), ("stale_up", "stale open on up-gaps only"),
         ("noise", "symmetric noise, 20% of range"),
         ("outside", "open outside range, 71% below / 29% above")]
rng = np.random.default_rng(7)
for kind, lab in kinds:
    for r in rates:
        rej5 = rej1 = 0; on_sr = []; on_sig = 0
        for it in range(500):
            O, H, L = inject(kind, r, rng)
            nl, nh, p, n = counts(O, H, L)
            rej5 += p < 0.05; rej1 += p < 0.01
            co = O[1:] / C0[:-1] - 1
            on_sr.append(co.mean() / co.std() * np.sqrt(252))
            on_sig += abs(nw_t(co)) > 1.96
        D[(kind, r)] = dict(rej5=rej5 / 500, rej1=rej1 / 500, on_sr=float(np.mean(on_sr)), on_sig=on_sig / 500)
        print("  %-38s rate %5.1f%%  power5 %.3f power1 %.3f  mean overnight SR %+.3f  share |t|>1.96 %.3f" %
              (lab, 100 * r, rej5 / 500, rej1 / 500, np.mean(on_sr), on_sig / 500))
co = O0[1:] / C0[:-1] - 1
print("  baseline exchange overnight SR %+.3f, t %.2f" % (co.mean() / co.std() * np.sqrt(252), nw_t(co)))
res["D"] = {"%s_%g" % k: v for k, v in D.items()}

# ------------------------------------------------------------------ E
print("\nE. ALTERNATIVE SESSION AGGREGATION")
E_ = {}
for lab, d in [("DXY free", FREE["DXY"]), ("DX exch", EX["DX"])]:
    for fq in ["W", "M"]:
        g = d.set_index("Date").groupby(pd.Grouper(freq=fq if fq == "W" else "ME"))
        agg = pd.DataFrame({"O": g.O.first(), "H": g.H.max(), "L": g.L.min()}).dropna()
        nl, nh, p, n = counts(agg.O, agg.H, agg.L); E_[(lab, fq)] = (nl, nh, p, n)
        print("  %-9s %s bars: n=%d  at low %d  at high %d  p=%.3g" % (lab, fq, n, nl, nh, p))
res["E"] = {"%s_%s" % k: v for k, v in E_.items()}
json.dump(res, open(os.path.join(OUT, "screen_study.json"), "w"), indent=1, default=str)
