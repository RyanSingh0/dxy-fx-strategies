"""Revision: why does carry differ between the futures panel and the long spot sample?

A sequence of single changes moves from the long spot sample to the futures panel, so each
step isolates one source of the difference: sample period, universe, instrument (spot plus
interbank differential vs within-contract futures returns) and signal (OECD interbank
differential vs the roll-implied differential). Equality of mean returns across the four
sub-periods is tested with a HAC Wald test. Also reports small-family noise ceilings with
1,000 replications (replacing the 8-10 used in the first submission)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "part4_reasoned"))
import numpy as np, pandas as pd, json
import statsmodels.api as sm
import paths, data as D, evaluate as E
from carry_futures import roll_carry, xs_rank_weights
from carry_dispersion import carry_linear

OUT = os.path.join(str(paths.RESULTS), "revision"); res = {}
TD = 252
A = pd.read_csv(str(paths.FRED_LONG), parse_dates=["Date"]).set_index("Date").sort_index()
px = pd.DataFrame({"EURUSD": A.DEXUSEU, "GBPUSD": A.DEXUSUK, "JPYUSD": 1 / A.DEXJPUS,
                   "CADUSD": 1 / A.DEXCAUS, "CHFUSD": 1 / A.DEXSZUS, "SEKUSD": 1 / A.DEXSDUS})
RC = {"EURUSD": "IR3TIB01EZM156N", "GBPUSD": "IR3TIB01GBM156N", "JPYUSD": "IR3TIB01JPM156N",
      "CADUSD": "IR3TIB01CAM156N", "CHFUSD": "IR3TIB01CHM156N", "SEKUSD": "IR3TIB01SEM156N"}
rates = A[[RC[p] for p in RC] + ["IR3TIB01USM156N"]].copy(); rates.columns = list(RC) + ["USD"]
start = max(rates[c].first_valid_index() for c in rates.columns)
px = px.loc[start:].dropna(how="any"); rates = rates.reindex(px.index).ffill()
ret_s = px.pct_change().fillna(0.0)
carry_s = pd.DataFrame({p: (rates[p] - rates["USD"]) / 100.0 for p in px.columns}).ffill()
# total return of a long position in the foreign currency: spot change plus the interest
# differential accrued over the session (known at the previous close). The first submission
# used the spot change alone, which omits the income a carry trade is designed to earn.
ret_tr = ret_s + carry_s.shift(1).fillna(0.0) / 252.0

def clin(Cm):
    d = Cm.sub(Cm.mean(axis=1), axis=0); g = d.abs().sum(axis=1)
    return d.div(g.where(g > 0, np.nan), axis=0).fillna(0.0)

def run(ret, sig, a=None, b=None):
    P = {"ret": ret}
    _, n, _, _ = E.apply_positions(clin(sig.rolling(252).mean()).reindex(ret.index).fillna(0.0), P, 1.0)
    if a is not None: n = n[(n.index >= a) & (n.index <= b)]
    return n

FP = D.panel()
fut_ret = FP["ret"]; a0, b0 = fut_ret.index[0], fut_ret.index[-1]
five = [c for c in px.columns if c != "SEKUSD"]
oecd_fut = pd.DataFrame({p: (FP["rates"][p] - FP["rates"]["USD"]) / 100.0 for p in D.PAIRS})
steps = []
s0 = run(ret_s, carry_s); steps.append(("(0) Spot change only, six currencies, 2002-2026 (first submission)", s0))
s1 = run(ret_tr, carry_s); steps.append(("(1) Spot plus interest differential, six currencies, 2002-2026", s1))
s2 = run(ret_tr, carry_s, a0, b0); steps.append(("(2) Same, restricted to the futures window 2010-12 to 2025-12", s2))
s3 = run(ret_tr[five], carry_s[five], a0, b0); steps.append(("(3) Same window, five currencies (krona dropped)", s3))
s4 = run(fut_ret, oecd_fut); steps.append(("(4) Futures returns, OECD interbank signal", s4))
C = roll_carry(FP)
_, s5, _, _ = E.apply_positions(carry_linear(FP, C.rolling(252).mean()), FP, 1.0)
steps.append(("(5) Futures returns, roll-implied signal (first submission)", s5))
print("STEPWISE RECONCILIATION (linear cross-sectional carry, one-year smoothed signal, 1 bp)")
R = {}
for lab, s in steps:
    st_ = E.summary(s); R[lab] = (st_["Sharpe"], st_["t"], st_["p"], int(st_["n"]))
    print("   %-66s Sharpe %+.3f  t %+.2f  p %.3f  n %d" % (lab, *R[lab]))
res["steps"] = R
# correlation between the spot and futures carry return in the common window
cm = pd.concat([s3.rename("spot5"), s4.rename("fut_oecd"), s5.rename("fut_roll")], axis=1, sort=True).dropna()
print("   correlations in the common window:\n", cm.corr().round(3).to_string())
res["corr"] = cm.corr().values.tolist()
# correlation of the two signals' cross-sectional ranks
sigA = clin(oecd_fut.rolling(252).mean()); sigB = carry_linear(FP, C.rolling(252).mean())
cc = np.nanmean([np.corrcoef(sigA.iloc[i], sigB.iloc[i])[0, 1] for i in range(300, len(sigA), 21)])
print("   average cross-sectional correlation of the two carry weight vectors: %.3f" % cc)
res["sigcorr"] = float(cc)

print("\nSUB-PERIOD INSTABILITY, long spot sample, HAC Wald test of equal means")
def wald(s, cuts):
    y = s.values; X = np.column_stack([((s.index.year >= a) & (s.index.year <= b)).astype(float) for a, b in cuts])
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    k = X.shape[1]; Rm = np.zeros((k - 1, k))
    for i in range(k - 1): Rm[i, 0] = 1; Rm[i, i + 1] = -1
    w = m.wald_test(Rm, scalar=True)
    return float(w.statistic), float(w.pvalue), k - 1
cuts = [(2002, 2007), (2008, 2013), (2014, 2019), (2020, 2026)]
w = wald(s1, cuts); print("   long spot sample (total return): Wald %.2f, df %d, p %.3f" % (w[0], w[2], w[1])); res["wald_spot"] = w
for a, b in cuts:
    m = (s1.index.year >= a) & (s1.index.year <= b); st_ = E.summary(s1[m])
    print("      spot TR %d-%d Sharpe %+.3f t %+.2f p %.3f" % (a, b, st_["Sharpe"], st_["t"], st_["p"]))
    res.setdefault("sub_tr", []).append((a, b, st_["Sharpe"], st_["t"], st_["p"]))
    m = (s0.index.year >= a) & (s0.index.year <= b); res.setdefault("sub_spot", []).append((a, b, E.sharpe(s0[m])))
# mechanism on total returns
Wm = clin(carry_s.rolling(252).mean()); g_sp = E.apply_positions(Wm, {"ret": ret_s}, 0.0)[0]
earned = (Wm.shift(1).fillna(0) * (carry_s.shift(1).fillna(0) / TD)).sum(axis=1)
print("   mechanism (long sample): differential earned %+.2f%%/yr, spot %+.2f%%/yr, total %+.2f%%/yr" % (100 * TD * earned.mean(), 100 * TD * g_sp.mean(), 100 * TD * (earned.mean() + g_sp.mean())))
res["mech_long"] = (100 * TD * earned.mean(), 100 * TD * g_sp.mean())
cuts2 = [(2011, 2014), (2015, 2018), (2019, 2022), (2023, 2025)]
w2 = wald(s5, cuts2); print("   futures panel (2011-14, 15-18, 19-22, 23-25): Wald %.2f, p %.3f" % (w2[0], w2[1])); res["wald_fut"] = w2
for a, b in cuts2:
    m = (s5.index.year >= a) & (s5.index.year <= b); print("      futures %d-%d Sharpe %+.3f" % (a, b, E.sharpe(s5[m])))
    m = (s3.index.year >= a) & (s3.index.year <= b); print("      spot5   %d-%d Sharpe %+.3f" % (a, b, E.sharpe(s3[m])))
# rolling 3-year Sharpe for the figure
roll = pd.DataFrame({"spot6_long": s1.rolling(756).mean() / s1.rolling(756).std() * np.sqrt(TD),
                     "futures_roll_signal": s5.rolling(756).mean() / s5.rolling(756).std() * np.sqrt(TD)})
roll.to_csv(os.path.join(OUT, "carry_rolling.csv"))

print("\nSMALL-FAMILY NOISE CEILINGS, 1,000 stationary-bootstrap replications (mean block 21)")
def ceiling_carry_long(B=1000):
    rv = ret_tr.values; nn = len(ret_tr); out = []
    Ws = [clin(carry_s), clin(carry_s.rolling(252).mean())]
    def rank_w(Cm, k):
        W = pd.DataFrame(0.0, index=Cm.index, columns=Cm.columns)
        n = Cm.notna().sum(axis=1); rk = Cm.rank(axis=1, ascending=False, method="first")
        W[rk.le(k)] = 1.0; W[rk.gt(n - k, axis=0)] = -1.0
        g = W.abs().sum(axis=1); return W.div(g.where(g > 0, np.nan), axis=0).fillna(0.0)
    Ws += [rank_w(carry_s, 2), rank_w(carry_s, 1), rank_w(carry_s, 3)]
    obs = max(E.sharpe(E.apply_positions(Wk, {"ret": ret_tr}, 1.0)[1]) for Wk in Ws)
    res["carry_long_variants_tr"] = [E.sharpe(E.apply_positions(Wk, {"ret": ret_tr}, 1.0)[1]) for Wk in Ws]
    print("   carry variants on total returns:", np.round(res["carry_long_variants_tr"], 3))
    for b in range(B):
        rng = np.random.default_rng(300 + b)
        idx = E.stationary_bootstrap_idx(nn, 1, rng, mean_block=21)[0]
        Q = {"ret": pd.DataFrame(rv[idx], index=ret_tr.index, columns=ret_tr.columns)}
        out.append(max(E.sharpe(E.apply_positions(Wk, Q, 1.0)[1]) for Wk in Ws))
    return np.array(out), obs
def ceiling_factors(B=1000):
    pxx = px; ret = ret_tr
    def sigs(cx):
        return {"c": carry_s.rolling(252).mean(), "m": (cx.shift(21) / cx.shift(252) - 1.0), "v": -(cx / cx.shift(1260) - 1.0)}
    def best(retq, cx):
        Q = {"ret": retq}; rr = {}
        for k, S in sigs(cx).items():
            rr[k] = E.apply_positions(clin(S), Q, 1.0)[1]
        Rq = pd.DataFrame(rr).dropna(); zq = Rq.div(Rq.std())
        return max([E.sharpe(Rq[k]) for k in Rq] + [E.sharpe(zq.mean(axis=1)), E.sharpe(zq[["c", "v"]].mean(axis=1))])
    obs = best(ret, pxx)
    Q = {"ret": ret}; rr = {}
    for k, S in sigs(pxx).items(): rr[k] = E.apply_positions(clin(S), Q, 1.0)[1]
    Rq = pd.DataFrame(rr).dropna(); zq = Rq.div(Rq.std())
    res["factors_tr"] = {k: E.sharpe(Rq[k]) for k in Rq}; res["factors_tr"]["comb"] = E.sharpe(zq.mean(axis=1))
    res["factors_tr"]["cv"] = E.sharpe(zq[["c", "v"]].mean(axis=1))
    print("   canonical factors on total returns:", {k: round(v, 3) for k, v in res["factors_tr"].items()})
    rv = ret.values; nn = len(ret); out = []
    for b in range(B):
        rng = np.random.default_rng(880 + b)
        idx = E.stationary_bootstrap_idx(nn, 1, rng, mean_block=21)[0]
        pq = pd.DataFrame(rv[idx], index=ret.index, columns=ret.columns)
        out.append(best(pq, 100 * (1 + pq).cumprod()))
    return np.array(out), obs
def ceiling_carry_fut(B=1000):
    P = FP; ret = P["ret"]; n_ = len(ret)
    from carry_futures import roll_carry as rc_
    def cands(Q):
        Cq = rc_(Q); aq = Cq.mean(axis=1); sq = np.sign(aq).shift(1).fillna(0.0)
        return [xs_rank_weights(Cq, 2), carry_linear(Q, Cq), carry_linear(Q, Cq.rolling(252).mean()),
                pd.DataFrame({p: sq / len(D.PAIRS) for p in D.PAIRS}),
                pd.DataFrame({p: (aq / aq.abs().rolling(252).mean()).clip(-2, 2).shift(1).fillna(0) / len(D.PAIRS) for p in D.PAIRS})]
    obs = max(E.sharpe(E.apply_positions(W, P, 1.0)[1]) for W in cands(P))
    out = []
    for b in range(B):
        rng = np.random.default_rng(400 + b)
        idx = E.stationary_bootstrap_idx(n_, 1, rng, mean_block=21)[0]
        Q = dict(P); Q["ret"] = pd.DataFrame(ret.values[idx], index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod(); Q["roll"] = P["roll"]; Q["dx_ret"] = -Q["ret"].mean(axis=1)
        out.append(max(E.sharpe(E.apply_positions(W, Q, 1.0)[1]) for W in cands(Q)))
    return np.array(out), obs
B = int(os.environ.get("BSMALL", "1000"))
for nm, fn in [("carry_long", ceiling_carry_long), ("factors_long", ceiling_factors), ("carry_futures", ceiling_carry_fut)]:
    c, obs = fn(B)
    np.save(os.path.join(OUT, "ceiling_%s.npy" % nm), c)
    pv = (1 + np.sum(c >= obs)) / (1 + len(c))
    print("   %-14s observed best %+.3f | ceiling mean %+.3f (MC s.e. %.3f) | q90 %+.3f q95 %+.3f | Monte Carlo p %.3f | first-10 mean %+.3f"
          % (nm, obs, c.mean(), c.std(ddof=1) / np.sqrt(len(c)), np.quantile(c, .9), np.quantile(c, .95), pv, c[:10].mean()))
    res["ceil_" + nm] = dict(obs=float(obs), mean=float(c.mean()), se=float(c.std(ddof=1) / np.sqrt(len(c))),
                             q90=float(np.quantile(c, .9)), q95=float(np.quantile(c, .95)), p=float(pv), first10=float(c[:10].mean()),
                             q90_first10=float(np.percentile(c[:10], 90)))
json.dump(res, open(os.path.join(OUT, "carry_instability.json"), "w"), indent=1, default=str)
