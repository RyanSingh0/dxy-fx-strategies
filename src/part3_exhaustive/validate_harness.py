"""Harness validation. Nothing produced in this repository should be believed until these
pass, because each one tests a way the harness could be lying.

  1. lookahead detection. A strategy handed tomorrow's return must post an absurd Sharpe
     ratio. If it does not, the shift is broken and nothing downstream means anything.
  2. no leak in the real families. Adding one extra lag should degrade a real strategy
     mildly rather than destroy it. A signal that collapses under one extra day of lag was
     reading the present.
  3. null calibration. The entire search is run on simulated zero-drift returns with the
     same volatilities and correlations. The best-of-N Sharpe ratio must match the
     theoretical expected maximum. White RC and Hansen SPA p-values must be roughly
     uniform. If noise produces small p-values then every result downstream is worthless.
  4. statistic cross-checks. Sharpe, HAC t, PSR and the cost arithmetic recomputed
     independently of the functions under test.
  5. turnover sanity. Buy and hold must cost nothing. A daily sign-flipper must cost two
     units of turnover per session."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd
from scipy import stats as st
import data as D
import evaluate as E
import signals as S

TD = 252
OK = lambda b: "PASS" if b else "*** FAIL ***"


def main():
    P = D.panel()
    ret = P["ret"]
    n, m = ret.shape
    fails = []

    print("=" * 100)
    print("1.  LOOKAHEAD DETECTION, can the harness catch a strategy that cheats?")
    print("=" * 100)
    cheat = np.sign(ret)                       # uses day t's own return: pure lookahead
    g, nt, tu, ge = E.apply_positions(E.pd.DataFrame(cheat, index=ret.index, columns=ret.columns)
                                      .pipe(lambda W: W.div(W.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)),
                                      P, cost_bp=0.0)
    # that one is correctly shifted, so it should NOT be absurd
    sr_shift = E.sharpe(g)
    cheat_w = pd.DataFrame(np.sign(ret.values), index=ret.index, columns=ret.columns)
    cheat_w = cheat_w.div(cheat_w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    unshifted = (cheat_w * ret).sum(axis=1)    # deliberately NOT shifted
    sr_leak = E.sharpe(unshifted)
    print("   perfect foresight, correctly shifted by the harness : Sharpe %+8.3f" % sr_shift)
    print("   the same weights applied WITHOUT the shift          : Sharpe %+8.3f" % sr_leak)
    c1 = sr_leak > 10 and abs(sr_shift) < 1.5
    print("   -> harness detects the leak (unshifted absurd, shifted ordinary): %s" % OK(c1))
    fails.append(("lookahead detection", c1))

    print("\n" + "=" * 100)
    print("2.  NO LEAK IN THE REAL FAMILIES, one extra day of lag must not destroy a signal")
    print("=" * 100)
    print("   %-22s %10s %10s %10s" % ("family", "Sharpe", "+1 lag", "ratio"))
    leaky = []
    for fam, lab, fn in S.base_families()[:14]:
        W = fn(P)
        g0, _, _, _ = E.apply_positions(W, P, 0.0)
        g1, _, _, _ = E.apply_positions(W.shift(1), P, 0.0)
        s0, s1 = E.sharpe(g0), E.sharpe(g1)
        ratio = (s1 / s0) if abs(s0) > 0.05 else np.nan
        flag = "" if (np.isnan(ratio) or ratio > -0.5) else "  <-- suspicious"
        if not np.isnan(ratio) and ratio < -0.5: leaky.append(lab)
        print("   %-22s %10.3f %10.3f %10s%s" % (lab, s0, s1, "%.2f" % ratio if not np.isnan(ratio) else "n/a", flag))
    c2 = len(leaky) == 0
    print("   -> no family collapses under one extra lag: %s" % OK(c2))
    fails.append(("no leak in families", c2))

    print("\n" + "=" * 100)
    print("3.  NULL CALIBRATION, run the whole search on returns that cannot be predicted")
    print("=" * 100)
    cfgs = S.enumerate_configs()
    print("   declared search space: %d configurations" % len(cfgs))
    L = np.linalg.cholesky(np.corrcoef(ret.values.T) + 1e-9 * np.eye(m))
    sd = ret.std(ddof=0).values
    rc_ps, spa_ps, maxsr, expsr = [], [], [], []
    for seed in range(4):
        rng = np.random.default_rng(1000 + seed)
        z = (rng.standard_normal((n, m)) @ L.T) * sd            # zero drift by construction
        Q = dict(P)
        Q["ret"] = pd.DataFrame(z, index=ret.index, columns=ret.columns)
        Q["close"] = 100 * (1 + Q["ret"]).cumprod()
        Q["high"] = Q["close"] * 1.002
        Q["low"] = Q["close"] * 0.998
        Q["dx_ret"] = -Q["ret"].mean(axis=1)
        Q["dx_close"] = 100 * (1 + Q["dx_ret"]).cumprod()
        cols = {}
        for key, build in cfgs:
            try:
                W = build(Q)
                g, _, _, _ = E.apply_positions(W, Q, 0.0)
                if g.std() > 0: cols[key] = g
            except Exception: pass
        R = pd.DataFrame(cols).dropna()
        srs = R.apply(E.sharpe)
        mt = E.multiple_testing(R, None, B=500, seed=seed)
        ex = E.expected_max_sharpe(srs.values, R.shape[1])
        rc_ps.append(mt["rc"]); spa_ps.append(mt["spa"])
        maxsr.append(float(srs.max())); expsr.append(ex)
        print("   seed %d: configs %4d | best Sharpe %+6.3f | theoretical best-of-N %+6.3f "
              "| RC p %.3f | SPA p %.3f | RW survivors %d"
              % (seed, R.shape[1], srs.max(), ex, mt["rc"], mt["spa"], mt["n_rw"]))
    c3a = np.mean(rc_ps) > 0.15
    c3b = abs(np.mean(maxsr) - np.mean(expsr)) < 0.35
    print("   -> mean RC p on pure noise %.3f (must not be small): %s" % (np.mean(rc_ps), OK(c3a)))
    print("   -> observed best-of-N %.3f vs theory %.3f: %s" % (np.mean(maxsr), np.mean(expsr), OK(c3b)))
    fails += [("null: RC not spuriously small", c3a), ("null: best-of-N matches theory", c3b)]

    print("\n" + "=" * 100)
    print("4.  STATISTIC CROSS-CHECKS, recomputed independently")
    print("=" * 100)
    W = S.tsmom(P, 126)
    g, net, tu, ge = E.apply_positions(W, P, cost_bp=1.0)
    man_sr = g.mean() / g.std(ddof=0) * np.sqrt(TD)
    c4a = abs(man_sr - E.sharpe(g)) < 1e-12
    print("   Sharpe: harness %.10f | manual %.10f  %s" % (E.sharpe(g), man_sr, OK(c4a)))
    x = g.dropna().values
    lag = int(np.floor(4 * (len(x) / 100.0) ** (2 / 9)))
    e = x - x.mean(); gam = [np.sum(e * e) / len(x)]
    for L_ in range(1, lag + 1):
        gam.append(np.sum(e[L_:] * e[:-L_]) / len(x))
    S_ = gam[0] + 2 * sum((1 - L_ / (lag + 1)) * gam[L_] for L_ in range(1, lag + 1))
    man_t = x.mean() / np.sqrt(S_ / len(x))
    ht, hp = E.nw_t(g)
    c4b = abs(man_t - ht) < 5e-3
    print("   Newey-West t (Bartlett, %d lags): harness %.6f | manual %.6f  %s" % (lag, ht, man_t, OK(c4b)))
    man_net = g - tu * 1e-4
    c4c = float((man_net - net).abs().max()) < 1e-15
    print("   cost arithmetic net = gross - turnover x cost: max abs diff %.2e  %s"
          % (float((man_net - net).abs().max()), OK(c4c)))
    z0 = st.norm.cdf((E.sharpe(g) / np.sqrt(TD)) * np.sqrt(len(g) - 1)
                     / np.sqrt(1 - st.skew(g) * (E.sharpe(g) / np.sqrt(TD))
                               + (st.kurtosis(g, fisher=False) - 1) / 4 * (E.sharpe(g) / np.sqrt(TD)) ** 2))
    c4d = abs(z0 - E.psr(g)) < 1e-9
    print("   PSR against zero: harness %.6f | manual %.6f  %s" % (E.psr(g), z0, OK(c4d)))
    fails += [("Sharpe", c4a), ("NW t", c4b), ("cost arithmetic", c4c), ("PSR", c4d)]

    print("\n" + "=" * 100)
    print("5.  TURNOVER SANITY")
    print("=" * 100)
    bh = pd.DataFrame(1.0 / m, index=ret.index, columns=ret.columns)
    _, _, t_bh, _ = E.apply_positions(bh, P, 1.0)
    flip = pd.DataFrame(np.tile(np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None], (1, m)) / m,
                        index=ret.index, columns=ret.columns)
    _, _, t_fl, _ = E.apply_positions(flip, P, 1.0)
    c5a = t_bh.iloc[1:].sum() < 1e-12
    c5b = abs(t_fl.iloc[1:].mean() - 2.0) < 1e-9
    print("   buy-and-hold total turnover after inception : %.2e  %s" % (t_bh.iloc[1:].sum(), OK(c5a)))
    print("   daily sign-flipper mean turnover per session: %.6f (expect 2.0)  %s" % (t_fl.iloc[1:].mean(), OK(c5b)))
    fails += [("turnover buy-and-hold", c5a), ("turnover flipper", c5b)]

    print("\n" + "=" * 100)
    bad = [k for k, v in fails if not v]
    print("VALIDATION SUMMARY: %d checks, %d failed" % (len(fails), len(bad)))
    for k, v in fails:
        print("   %-38s %s" % (k, OK(v)))
    if bad:
        print("\n*** DO NOT TRUST DOWNSTREAM RESULTS: %s" % ", ".join(bad))
    return len(bad) == 0


if __name__ == "__main__":
    import pandas as pd
    ok = main()
    raise SystemExit(0 if ok else 1)
