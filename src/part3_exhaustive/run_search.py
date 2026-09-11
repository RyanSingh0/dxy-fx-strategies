"""The full search, run under a protocol fixed before any results were seen.

  1. the search space is enumerated from the published specifications in signals.py and
     declared in full, so every configuration built is counted in every correction
  2. costs are charged on turnover at 1 bp per unit of gross traded, then swept
  3. the comparison that matters is not whether the Sharpe ratio is positive but whether
     it exceeds what this same search produces on returns that cannot be predicted, the
     noise ceiling measured in validate_harness.py
  4. 2019 to 2025 is a holdout. Selection happens on 2011 to 2018 only and the holdout is
     evaluated once.
  5. family ensembles select nothing, so they need no correction and are the cleaner test
     of whether a family of ideas has value"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, sys, time
import data as D, evaluate as E, signals as S

TD = 252
COST_BP = 1.0
SPLIT = 2018


def build_all(P, cost_bp=COST_BP):
    cfgs = S.enumerate_configs()
    gross, net, turn = {}, {}, {}
    t0 = time.time()
    for i, (key, build) in enumerate(cfgs):
        try:
            W = build(P)
            g, nt, tu, ge = E.apply_positions(W, P, cost_bp)
            if not np.isfinite(g.std()) or g.std() == 0: continue
            gross[key] = g; net[key] = nt; turn[key] = float(tu.mean() * TD)
        except Exception as e:
            print("   BUILD FAILED %s: %s" % (str(key), e))
        if (i + 1) % 300 == 0:
            print("   built %d/%d (%.0fs)" % (i + 1, len(cfgs), time.time() - t0)); sys.stdout.flush()
    return pd.DataFrame(gross).dropna(), pd.DataFrame(net).dropna(), pd.Series(turn)


def main():
    P = D.panel()
    ret = P["ret"]
    print("=" * 104)
    print("STRATEGY 4, FULL SEARCH")
    print("=" * 104)
    print("sessions %d | %s .. %s | costs %.1f bp per unit turnover"
          % (len(ret), ret.index[0].date(), ret.index[-1].date(), COST_BP))

    Gg, Gn, turn = build_all(P)
    K = Gn.shape[1]
    print("\nconfigurations successfully built: %d" % K)

    # ---------------- benchmarks
    dxr = P["dx_ret"].reindex(Gn.index).fillna(0.0)
    ew = ret.mean(axis=1).reindex(Gn.index).fillna(0.0)
    print("\n" + "=" * 104); print("BENCHMARKS"); print("=" * 104)
    for lab, r in [("long DXY futures (dollar)", dxr), ("short DXY futures", -dxr),
                   ("equal-weight long the five pairs", ew), ("cash", pd.Series(0.0, index=Gn.index))]:
        s = E.summary(r)
        if s: print("   %-36s TotRet %+7.1f%%  Sharpe %+6.3f  t %+5.2f  p %.4f"
                    % (lab, 100 * s["TotRet"], s["Sharpe"], s["t"], s["p"]))

    # ---------------- the noise ceiling
    print("\n" + "=" * 104)
    print("THE NOISE CEILING, what this same search yields on unpredictable returns")
    print("=" * 104)
    n, m = ret.shape
    L = np.linalg.cholesky(np.corrcoef(ret.values.T) + 1e-9 * np.eye(m))
    sd = ret.std(ddof=0).values
    ceil = []
    for seed in range(6):
        rng = np.random.default_rng(5000 + seed)
        z = (rng.standard_normal((n, m)) @ L.T) * sd
        Q = dict(P)
        Q["ret"] = pd.DataFrame(z, index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod()
        Q["high"] = Q["close"] * 1.002; Q["low"] = Q["close"] * 0.998
        Q["dx_ret"] = -Q["ret"].mean(axis=1)
        Q["dx_close"] = 100 * (1 + Q["dx_ret"]).cumprod()
        cols = {}
        for key, build in S.enumerate_configs():
            try:
                W = build(Q); g, nt, _, _ = E.apply_positions(W, Q, COST_BP)
                if nt.std() > 0: cols[key] = nt
            except Exception: pass
        srs = pd.DataFrame(cols).dropna().apply(E.sharpe)
        ceil.append(float(srs.max()))
        print("   simulation %d: best net Sharpe over %d configurations = %+.3f" % (seed, len(srs), srs.max()))
    CEIL = float(np.mean(ceil))
    print("   -> noise ceiling: mean %+.3f, range %+.3f to %+.3f" % (CEIL, min(ceil), max(ceil)))
    print("      Any strategy below this is indistinguishable from luck at this search size.")

    # ---------------- the real grid
    srs_n = Gn.apply(E.sharpe).sort_values(ascending=False)
    print("\n" + "=" * 104)
    print("THE REAL SEARCH, net of costs, ranked")
    print("=" * 104)
    print("   best %+.3f | 99th %+.3f | 95th %+.3f | median %+.3f | worst %+.3f"
          % (srs_n.max(), srs_n.quantile(.99), srs_n.quantile(.95), srs_n.median(), srs_n.min()))
    print("   configurations above the noise ceiling (%.3f): %d of %d (%.1f%%)"
          % (CEIL, int((srs_n > CEIL).sum()), K, 100 * (srs_n > CEIL).mean()))
    print("\n   %-52s %8s %8s %7s %8s %9s" % ("configuration", "Sharpe", "TotRet%", "t", "p", "turn/yr"))
    for key in srs_n.head(15).index:
        s = E.summary(Gn[key])
        print("   %-52s %8.3f %8.1f %7.2f %8.4f %9.1f"
              % ("/".join(map(str, key)), s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"], turn[key]))

    # ---------------- corrections
    print("\n" + "=" * 104)
    print("MULTIPLE TESTING OVER ALL %d CONFIGURATIONS" % K)
    print("=" * 104)
    for lab, bench in [("zero (self-financing book)", None), ("long DXY futures", dxr)]:
        mt = E.multiple_testing(Gn, bench, B=2000, seed=11)
        print("   vs %-28s White RC p = %.4f | Hansen SPA p = %.4f | Romano-Wolf survivors = %d"
              % (lab, mt["rc"], mt["spa"], mt["n_rw"]))
        for c in mt["rw"][:5]: print("        survives: %s" % "/".join(map(str, c)))

    best = srs_n.index[0]
    bs = E.summary(Gn[best])
    d, sr0 = E.deflated_sharpe(Gn[best], srs_n.values, K)
    hl = E.harvey_liu_haircut(bs["t"], K)
    print("\n   best configuration: %s" % "/".join(map(str, best)))
    print("      Sharpe %+.3f | PSR vs zero %.4f" % (bs["Sharpe"], E.psr(Gn[best])))
    print("      expected best-of-%d Sharpe under the null: %+.3f" % (K, sr0))
    print("      DEFLATED SHARPE %.4f  (needs > 0.95)  -> %s" % (d, "PASS" if d > 0.95 else "FAIL"))
    print("      Harvey-Liu: single p %.4f -> Bonferroni %.4f, BHY %.4f"
          % (hl["p_single"], hl["p_bonf"], hl["p_bhy"]))
    pbo = E.cscv_pbo(Gn, 16)
    print("      CSCV probability of backtest overfitting: %.3f (median OOS rank %.0f/%d)"
          % (pbo["pbo"], pbo["med_rank"], pbo["k"]))

    # ---------------- family ensembles
    print("\n" + "=" * 104)
    print("FAMILY ENSEMBLES, nothing is selected, so nothing needs correcting")
    print("=" * 104)
    fams = sorted(set(k[0] for k in Gn.columns))
    print("   %-14s %6s %9s %9s %8s %8s %9s" % ("family", "n cfg", "Sharpe", "TotRet%", "t", "p", "vs ceiling"))
    rows = []
    for f in fams + ["ALL"]:
        cols = [c for c in Gn.columns if (f == "ALL" or c[0] == f)]
        e = Gn[cols].mean(axis=1)
        s = E.summary(e)
        rows.append((f, len(cols), s["Sharpe"], s["TotRet"], s["t"], s["p"]))
        print("   %-14s %6d %9.3f %9.1f %8.2f %8.4f %9s"
              % (f, len(cols), s["Sharpe"], 100 * s["TotRet"], s["t"], s["p"],
                 "above" if s["Sharpe"] > CEIL else "below"))

    # ---------------- holdout, evaluated once
    print("\n" + "=" * 104)
    print("HOLDOUT, select on 2011-%d, evaluate on %d-2025 exactly once" % (SPLIT, SPLIT + 1))
    print("=" * 104)
    yr = Gn.index.year
    IS, OS = yr <= SPLIT, yr > SPLIT
    isr, osr = Gn[IS].apply(E.sharpe), Gn[OS].apply(E.sharpe)
    pick = isr.idxmax()
    so = E.summary(Gn[OS][pick])
    print("   in-sample winner: %s" % "/".join(map(str, pick)))
    print("      IS Sharpe %+.3f  ->  OOS Sharpe %+.3f  (TotRet %+.1f%%, t %+.2f, p %.4f)"
          % (isr[pick], osr[pick], 100 * so["TotRet"], so["t"], so["p"]))
    print("      its rank out of sample: %d of %d" % (1 + int((osr > osr[pick]).sum()), K))
    from scipy import stats as st
    rho, pr = st.spearmanr(isr.values, osr.values)
    print("   rank correlation of IS and OOS Sharpe across all %d configurations: rho %+.3f (p %.4g)"
          % (K, rho, pr))
    print("      -> a rho near zero means in-sample ranking carries no information about the future.")
    top10 = isr.sort_values(ascending=False).head(10).index
    print("   the top 10 in-sample configurations, out of sample: mean Sharpe %+.3f (median %+.3f)"
          % (osr[top10].mean(), osr[top10].median()))
    bdx = dxr[OS]
    print("   for comparison, the dollar over the holdout: Sharpe %+.3f, TotRet %+.1f%%"
          % (E.sharpe(bdx), 100 * ((1 + bdx).prod() - 1)))

    # ---------------- cost sweep on the winner
    print("\n" + "=" * 104)
    print("COST SENSITIVITY OF THE BEST CONFIGURATION")
    print("=" * 104)
    key, build = [c for c in S.enumerate_configs() if c[0] == best][0]
    W = build(P)
    for c in (0, 0.5, 1, 2, 5, 10):
        g, nt, tu, _ = E.apply_positions(W, P, c)
        s = E.summary(nt)
        print("   %4.1f bp: Sharpe %+6.3f  TotRet %+7.1f%%  p %.4f  %s"
              % (c, s["Sharpe"], 100 * s["TotRet"], s["p"], "above ceiling" if s["Sharpe"] > CEIL else "below ceiling"))

    Gn.to_pickle(str(paths.RESULTS)+"/part3_net_returns.pkl")
    pd.Series({"noise_ceiling": CEIL, "K": K}).to_csv(str(paths.RESULTS)+"/part3_meta.csv")
    print("\nsaved results/part3_net_returns.pkl")


if __name__ == "__main__":
    main()
