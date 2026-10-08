"""Revision: where in the bar-construction pipeline does the free series' open go wrong?

  1. timing: regress each source's overnight and daytime returns on the other's, to locate
     the free open in time relative to the exchange session
  2. stale vs displaced: on the flagged sessions, is the free open closer to the exchange
     open, to the exchange previous close, or to neither?
  3. independent spot cross-check: a synthetic DXY built from the ICE formula using
     (a) the free provider's own constituent spot bars and (b) Federal Reserve H.10 noon
     rates, compared against the free index's open and close
  4. formal constituent decomposition of the overnight leg (log identity of the index)
  5. sub-period overnight Sharpe by three-year window for both sources"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd, json
import statsmodels.api as sm
import paths

OUT = os.path.join(str(paths.RESULTS), "revision")
W = {"EURUSD": 0.576, "JPYUSD": 0.136, "GBPUSD": 0.119, "CADUSD": 0.091, "SEKUSD": 0.042, "CHFUSD": 0.036}
res = {}
fx = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"]).set_index("Date").sort_index()
d = pd.read_csv(str(paths.DX_FILE), skipfooter=1, engine="python")
ex = pd.DataFrame({"O": pd.to_numeric(d["Open"]).values, "H": pd.to_numeric(d["High"]).values,
                   "L": pd.to_numeric(d["Low"]).values, "C": pd.to_numeric(d["Latest"]).values,
                   "Sym": d["Symbol"].astype(str).values}, index=pd.to_datetime(d["Time"])).sort_index()
fr = fx[["DXY_Open", "DXY_High", "DXY_Low", "DXY_Close"]].copy(); fr.columns = ["O", "H", "L", "C"]
j = fr.join(ex, how="inner", lsuffix="_f", rsuffix="_e").dropna(subset=["O_f", "C_f", "O_e", "C_e"])
roll = (j.Sym != j.Sym.shift(1))
co_f = np.log(j.O_f / j.C_f.shift(1)); oc_f = np.log(j.C_f / j.O_f)
co_e = np.log(j.O_e / j.C_e.shift(1)); oc_e = np.log(j.C_e / j.O_e)
co_e[roll] = np.nan  # a roll moves the futures level between contracts
k = co_f.notna() & oc_f.notna() & co_e.notna() & oc_e.notna() & oc_e.shift(1).notna() & co_e.shift(-1).notna()

def hac(y, X, L=8):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": L}); return m

print("1. TIMING REGRESSIONS (log returns, roll sessions excluded, HAC 8 lags)")
X = pd.DataFrame({"co_e": co_e, "oc_e": oc_e, "oc_e_lag": oc_e.shift(1), "co_e_lead": co_e.shift(-1)})[k]
m1 = hac(co_f[k], X); m2 = hac(oc_f[k], X)
print("   free overnight on exchange legs:\n", m1.params.round(3).to_dict(), "\n   t:", m1.tvalues.round(2).to_dict(), " R2 %.3f" % m1.rsquared)
print("   free daytime on exchange legs:\n", m2.params.round(3).to_dict(), "\n   t:", m2.tvalues.round(2).to_dict(), " R2 %.3f" % m2.rsquared)
res["timing"] = dict(co=dict(b=m1.params.to_dict(), t=m1.tvalues.to_dict(), r2=m1.rsquared),
                     oc=dict(b=m2.params.to_dict(), t=m2.tvalues.to_dict(), r2=m2.rsquared))
print("   corr(close-to-close free, exchange) = %.3f" % np.corrcoef((co_f + oc_f)[k], (co_e + oc_e)[k])[0, 1])
res["cc_corr"] = float(np.corrcoef((co_f + oc_f)[k], (co_e + oc_e)[k])[0, 1])
# intercepts: mean overnight drift unexplained by exchange returns (bp per session)
print("   intercept of free overnight regression: %.2f bp/session (t %.2f)" % (1e4 * m1.params["const"], m1.tvalues["const"]))

print("\n2. STALE OR DISPLACED? flagged sessions vs the rest (offset-adjusted free open)")
off = j.C_f - j.C_e
Oadj = j.O_f - off.shift(1)          # use the previous session's offset: known before the open
rg = j.H_e - j.L_e
atlo = np.isclose(j.O_f, j.L_f, atol=1e-9) & (j.H_f > j.L_f)
athi = np.isclose(j.O_f, j.H_f, atol=1e-9) & (j.H_f > j.L_f)
stale = np.isclose(j.O_f, j.C_f.shift(1), atol=1e-9)
grp = {"free open at own low": atlo, "free open at own high": athi, "other": ~atlo & ~athi}
S2 = {}
for lab, g in grp.items():
    g = g & rg.gt(0) & Oadj.notna()
    dO = ((Oadj - j.O_e) / rg)[g]; dC = ((Oadj - j.C_e.shift(1)) / rg)[g]
    S2[lab] = dict(n=int(g.sum()), stale_share=float(stale[g].mean()),
                   med_abs_dist_to_exch_open=float(dO.abs().median()), med_abs_dist_to_exch_prevclose=float(dC.abs().median()),
                   mean_signed_dist_to_exch_open=float(dO.mean()), share_closer_to_prevclose=float((dC.abs() < dO.abs()).mean()),
                   free_co_bp=float(1e4 * co_f[g].mean()), exch_co_bp=float(1e4 * co_e[g].mean()))
    print("   %-24s n %4d | stale (open = own prev close) %5.1f%% | median |dist| to exch open %.3f, to exch prev close %.3f of range | signed dist to exch open %+.3f | closer to prev close %4.1f%% | ON bp free %+.1f exch %+.1f"
          % (lab, g.sum(), 100 * stale[g].mean(), dO.abs().median(), dC.abs().median(), dO.mean(), 100 * (dC.abs() < dO.abs()).mean(), 1e4 * co_f[g].mean(), 1e4 * co_e[g].mean()))
res["stale_vs_displaced"] = S2
# among stale sessions
g = stale & Oadj.notna()
print("   stale sessions (free open == own previous close): n %d, free ON = 0 by construction; exch ON on those sessions %+.2f bp" % (g.sum(), 1e4 * co_e[g].mean()))
# offset statistics
print("   level offset (free - exch close): mean %.4f sd %.4f" % (off.mean(), off.std()))
res["offset"] = (float(off.mean()), float(off.std()))

print("\n3. INDEPENDENT SPOT CROSS-CHECKS")
def synth(df, fld):
    lv = np.log(50.14348112)
    for p, w in W.items():
        lv = lv - w * np.log(df[p + "_" + fld])
    return np.exp(lv)
syn = pd.DataFrame({f: synth(fx, f) for f in ["Open", "Close"]})
# alignment of the constituent file against the index file
lr_idx = np.log(fr.C).diff(); lr_syn = np.log(syn.Close).diff()
for lag in (-1, 0, 1):
    c = pd.concat([lr_idx, lr_syn.shift(lag)], axis=1).dropna().corr().iloc[0, 1]
    print("   corr(free index close return, synthetic-from-free-constituents return shifted %+d) = %.3f" % (lag, c))
    res["syn_lag_%d" % lag] = float(c)
best = max((-1, 0, 1), key=lambda l: res["syn_lag_%d" % l])
syn_al = syn.shift(best)
jj = fr.join(syn_al, how="inner").dropna()
co_syn = np.log(jj.Open / jj.Close.shift(1)); co_idx = np.log(jj.O / jj.C.shift(1))
cs = pd.concat([co_syn, co_idx], axis=1).dropna()
print("   using alignment shift %+d: overnight Sharpe, free index %.3f vs synthetic from free constituents %.3f; corr %.3f"
      % (best, co_idx.mean() / co_idx.std() * np.sqrt(252), co_syn.mean() / co_syn.std() * np.sqrt(252), cs.corr().iloc[0, 1]))
res["syn_on_sr"] = (float(co_idx.mean() / co_idx.std() * np.sqrt(252)), float(co_syn.mean() / co_syn.std() * np.sqrt(252)), float(cs.corr().iloc[0, 1]), best)
# H.10 noon synthetic
h = pd.read_csv(str(paths.FRED_DAILY), parse_dates=["Date"]).set_index("Date")
h10 = 50.14348112 * h.DEXUSEU ** -0.576 * h.DEXJPUS ** 0.136 * h.DEXUSUK ** -0.119 * h.DEXCAUS ** 0.091 * h.DEXSDUS ** 0.042 * h.DEXSZUS ** 0.036
h10 = h10.dropna()
for lab, s in [("free close", fr.C), ("exchange close", ex.C)]:
    z = pd.concat([np.log(s).diff(), np.log(h10).diff()], axis=1).dropna()
    lv = pd.concat([s, h10], axis=1).dropna()
    print("   H.10 noon synthetic vs %-15s daily-change corr %.3f | mean level gap %.3f (sd %.3f)" % (lab, z.corr().iloc[0, 1], (lv.iloc[:, 0] - lv.iloc[:, 1]).mean(), (lv.iloc[:, 0] - lv.iloc[:, 1]).std()))
    res["h10_" + lab] = (float(z.corr().iloc[0, 1]), float((lv.iloc[:, 0] - lv.iloc[:, 1]).mean()))
# Does the free open on flagged sessions deviate from the synthetic open?
jj2 = jj.join(pd.DataFrame({"atlo": atlo, "athi": athi}), how="inner")
dev = np.log(jj2.O / jj2.Open) - np.log(jj2.C.shift(1) / jj2.Close.shift(1))  # open gap net of the previous close gap
for lab, g in [("at own low", jj2.atlo), ("at own high", jj2.athi), ("other", ~jj2.atlo & ~jj2.athi)]:
    print("   free index open minus synthetic open (net of prior-close gap), %-12s n %4d mean %+.2f bp, median %+.2f bp" % (lab, g.sum(), 1e4 * dev[g].mean(), 1e4 * dev[g].median()))
    res["dev_" + lab] = (int(g.sum()), float(1e4 * dev[g].mean()), float(1e4 * dev[g].median()))

print("\n4. FORMAL CONSTITUENT DECOMPOSITION OF THE OVERNIGHT LEG (log returns, sum over sample)")
# ln DXY = c - sum w_i ln(XUSD_i)  =>  r_DXY = - sum w_i r_i  + residual
cons = {}
jf = fx.copy()
for p in W:
    s_o = jf[p + "_Open"].shift(best); s_c = jf[p + "_Close"].shift(best)
    cons[p] = -W[p] * np.log(s_o / s_c.shift(1))
cons = pd.DataFrame(cons)
idx_on = np.log(fr.O / fr.C.shift(1))
dd = pd.concat([idx_on.rename("index"), cons], axis=1).dropna()
tot = dd["index"].sum(); contrib = dd[list(W)].sum()
resid = tot - contrib.sum()
print("   sessions %d | free index overnight cumulative log return %+.4f" % (len(dd), tot))
for p in W: print("      %-7s weight %.3f contribution %+.4f" % (p, W[p], contrib[p]))
print("      sum of constituents %+.4f | residual (index-specific) %+.4f (%.0f%% of total)" % (contrib.sum(), resid, 100 * resid / tot))
res["decomp_free"] = dict(n=len(dd), total=float(tot), contrib={p: float(v) for p, v in contrib.items()}, resid=float(resid))
# R2 of index overnight on synthetic overnight
m = sm.OLS(dd["index"], sm.add_constant(dd[list(W)].sum(axis=1))).fit()
print("   regression index ON on synthetic ON: slope %.3f R2 %.3f intercept %.2f bp/session (t %.2f)" % (m.params.iloc[1], m.rsquared, 1e4 * m.params.iloc[0], m.tvalues.iloc[0]))
res["decomp_reg"] = (float(m.params.iloc[1]), float(m.rsquared), float(1e4 * m.params.iloc[0]), float(m.tvalues.iloc[0]))
# time path for the figure
path = pd.DataFrame({"index": dd["index"].cumsum(), "constituents": dd[list(W)].sum(axis=1).cumsum()})
path["residual"] = path["index"] - path["constituents"]
path.to_csv(os.path.join(OUT, "decomp_path.csv"))
# exchange analogue: DX futures overnight vs the five CME futures (SEK omitted, weights renormalised not applied)
fut = {}
for s, p in [("E6", "EURUSD"), ("B6", "GBPUSD"), ("D6", "CADUSD"), ("J6", "JPYUSD"), ("S6", "CHFUSD")]:
    q = pd.read_csv(str(paths.FUTURES / (s + ".csv")), skipfooter=1, engine="python")
    q = pd.DataFrame({"O": pd.to_numeric(q.Open).values, "C": pd.to_numeric(q.Latest).values, "Sym": q.Symbol.astype(str).values},
                     index=pd.to_datetime(q.Time)).sort_index()
    r = np.log(q.O / q.C.shift(1)); r[q.Sym != q.Sym.shift(1)] = np.nan
    fut[p] = -W[p] * r
fut = pd.DataFrame(fut)
de = pd.concat([co_e.rename("index"), fut], axis=1).dropna()
ce = de[list(fut.columns)].sum()
print("   EXCHANGE: sessions %d | DX overnight cum %+.4f | five futures %+.4f | residual %+.4f" % (len(de), de["index"].sum(), ce.sum(), de["index"].sum() - ce.sum()))
res["decomp_exch"] = dict(n=len(de), total=float(de["index"].sum()), contrib={p: float(v) for p, v in ce.items()}, resid=float(de["index"].sum() - ce.sum()))

print("\n5. OVERNIGHT SHARPE BY THREE-YEAR WINDOW")
yr = j.index.year
S5 = []
for a in range(2011, 2026, 3):
    mm = (yr >= a) & (yr <= a + 2)
    x, y = co_f[mm].dropna(), co_e[mm].dropna()
    S5.append((a, float(x.mean() / x.std() * np.sqrt(252)), float(y.mean() / y.std() * np.sqrt(252))))
    print("   %d-%d  free %+.3f  exchange %+.3f" % (a, a + 2, S5[-1][1], S5[-1][2]))
res["windows"] = S5
json.dump(res, open(os.path.join(OUT, "mechanism.json"), "w"), indent=1, default=str)

print("\n6. FIELD-BY-FIELD LEVEL OFFSET, free minus exchange (index points), by window")
F6 = []
for a in range(2011, 2026, 3):
    mm = (yr >= a) & (yr <= a + 2) & ~roll & ~roll.shift(1, fill_value=True)
    q = j[mm]
    row = [a] + [float((q[f + "_f"] - q[f + "_e"]).mean()) for f in "OHLC"] + [float((q.O_f - q.O_e - (q.C_f.shift(1) - q.C_e.shift(1))).mean())]
    F6.append(row)
    print("   %d-%d  O %+.4f  H %+.4f  L %+.4f  C %+.4f | open offset minus previous close offset %+.4f" % (a, a + 2, *row[1:]))
res["field_offsets"] = F6
# exchange decomposition regression
m = sm.OLS(de["index"], sm.add_constant(de[list(fut.columns)].sum(axis=1))).fit()
print("   EXCHANGE regression DX ON on five-futures ON: slope %.3f R2 %.3f" % (m.params.iloc[1], m.rsquared))
res["decomp_exch_reg"] = (float(m.params.iloc[1]), float(m.rsquared))
pd.DataFrame({"index": de["index"].cumsum(), "constituents": de[list(fut.columns)].sum(axis=1).cumsum()}).to_csv(os.path.join(OUT, "decomp_path_exch.csv"))
json.dump(res, open(os.path.join(OUT, "mechanism.json"), "w"), indent=1, default=str)

print("\n7. OFFSET CHANGES SPLIT BY SUB-PERIOD (carry-basis convergence vs open bias), index points per session")
F7 = []
for a in [2011, 2014, 2017, 2020, 2023, 0]:
    mm = (((yr >= a) & (yr <= a + 2)) if a else np.ones(len(j), bool)) & ~roll.values & ~roll.shift(1, fill_value=True).values
    q = j
    on = (q.O_f - q.O_e) - (q.C_f - q.C_e).shift(1)
    dy = (q.C_f - q.C_e) - (q.O_f - q.O_e)
    on, dy = on[mm].dropna(), dy[mm].dropna()
    F7.append((a, float(on.mean()), float(dy.mean()), float(on.mean() + dy.mean()), float(on.mean() / (on.std() / np.sqrt(len(on))))))
    print("   %s  overnight %+.5f  daytime %+.5f  sum (basis convergence) %+.5f  | t(overnight) %.2f" % ("%d-%d" % (a, a + 2) if a else "all      ", *F7[-1][1:]))
res["offset_split"] = F7
json.dump(res, open(os.path.join(OUT, "mechanism.json"), "w"), indent=1, default=str)
