"""Noise ceiling with many replications (revision).

For each replicate b the complete 1,312-configuration search is run on a resampled panel
and the full vector of 1,312 net Sharpe ratios is stored. Seeds follow the original
scripts so the first six replicates reproduce the six reported in the first submission:
  stationary bootstrap, mean block L : seed 9000 + b        (verify_search.py, V2)
  Gaussian null                      : seed 5000 + b        (run_search.py)
Usage: python3 run_ceiling.py <kind> <B> <mean_block>   kind in {boot, gauss}"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import numpy as np
from multiprocessing import Pool
import data as D, paths
import fast_search as F

P = None
def _init():
    global P
    P = D.panel()

def one(args):
    kind, b, L = args
    if kind == "gauss":
        Q = F.bootstrap_panel(P, 5000 + b, gaussian=True)
    else:
        seed = 9000 + b if L == 21 else 9000 + 100000 * L + b
        Q = F.bootstrap_panel(P, seed, mean_block=L)
    _, M = F.evaluate_all(Q)
    return b, F.sharpes(M).astype(np.float32)

if __name__ == "__main__":
    kind, B, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = os.path.join(str(paths.RESULTS), "revision")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    res = np.full((B, 1312), np.nan, np.float32)
    with Pool(int(os.environ.get("NPROC", "2")), initializer=_init) as pool:
        for k, (b, s) in enumerate(pool.imap_unordered(one, [(kind, b, L) for b in range(B)])):
            res[b] = s
            if (k + 1) % 50 == 0:
                print("%s L=%d: %d/%d done (%.0fs)" % (kind, L, k + 1, B, time.time() - t0), flush=True)
                np.save(os.path.join(out, "ceiling_%s_L%d.npy" % (kind, L)), res)
    np.save(os.path.join(out, "ceiling_%s_L%d.npy" % (kind, L)), res)
    print("done in %.0fs" % (time.time() - t0))
