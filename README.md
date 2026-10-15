# Two Blind Spots in Backtest Validation

## Repository layout

`src/`, `data/`, `results/` and `paper/` reproduce the submitted manuscript. Run
`./run_all.sh` and every table regenerates from the raw inputs.

`original/` holds the earlier notebook analysis of the same data, kept deliberately.
The manuscript identifies four defects in that implementation, and this is the code
in which they can be inspected: the `2Q` period alias collapsing to quarterly, the
pair price files being one session late relative to the index, the benchmark for the
Reality Check and SPA tests, and `np.where` mapping NaN to -1 over the first lookback
window. The notebooks are preserved as they were and are not maintained.

Replication code and data for the paper *Two Blind Spots in Backtest Validation: From a
Manufactured Dollar Index Anomaly to a Corrected Null Result* (Meena and Pinsky).

Standard backtest validation corrects for the number of strategies tried, but it takes the
input data and the declared search space as given. This repository shows what happens when
each of those fails, using the U.S. Dollar Index and its constituent currency futures.

1. **A defective input.** On a free data source the overnight leg of the dollar index has a
   Sharpe ratio of -1.36, and the overfitting diagnostics rate the result as robust (the
   probability of backtest overfitting is 0.02). On an independently sourced futures
   comparator covering the same sessions the overnight leg is +0.25 (*p* = 0.33). The
   evidence is consistent with a persistent downward bias in the free series' recorded
   opens. A binomial screen on where the open sits in the session range flags the free
   series; of the six futures series only sterling is flagged at 5 per cent, and none after
   a Bonferroni correction. The flagged sessions in the free series cluster in time, so
   the exact binomial *p*-value overstates the evidence; Newey-West versions still flag it.
2. **An undeclared search.** A regime-conditioned strategy originally reported at Sharpe
   0.79 (*t* = 3.09) drops to 0.72 (*t* = 2.78) once an `np.where` defect is fixed, and
   does not survive correction on tradeable instruments once its full 216-configuration
   search is declared instead of the six configurations reported.

A 1,312-configuration search of published specifications on the futures data finds nothing
above the **noise ceiling**, the best Sharpe ratio the same search reaches on
block-bootstrapped returns.

## Headline numbers

| Result | Value |
|---|---|
| Overnight Sharpe, free provider series | -1.36 (*p* < 0.0001) |
| Overnight Sharpe, futures comparator | +0.25 (*p* = 0.33) |
| Probability of backtest overfitting, free series | 0.016 to 0.028 |
| Probability of backtest overfitting, futures | 0.765 to 0.779 |
| Open location binomial screen, free series | *p* = 3.68e-13 (exact); 2.4e-4 (Newey-West, 21 lags) |
| Open location binomial screen, index futures | *p* = 0.788 |
| Best of 1,312 configurations, net of costs | +0.653 |
| Noise ceiling, 2,000 block-bootstrap replications | mean 0.737 (MC s.e. 0.003), 95th percentile 0.940 |
| Where the observed best sits in the null distribution | 23rd percentile (Monte Carlo *p* = 0.77) |

## Layout

```
data/          futures, free provider and FRED inputs; see data/README.md
src/
  paths.py                 all file locations resolve here
  data.py                  aligned panel, roll handling, integrity report
  evaluate.py              the one period shift, Sharpe, HAC t, bootstrap, RC/SPA/RW, DSR, CSCV
  signals.py               the 1,312 configuration search space
  part1_data_defect/       the defective input and the screen
  part2_declared_search/   the undeclared search space
  part3_exhaustive/        harness validation, the full search, verification
  part4_reasoned/          strategies built from the structure rather than enumerated
  revision/                analyses added in the first revision (see below)
results/       text output of every script; results/revision/ for the revision
paper/         the manuscript as first submitted
docs/          longer write-ups of each stage
```

## Reproducing

```bash
pip install -r requirements.txt
./run_all.sh
```

Expect two to three hours on two cores, most of it the noise-ceiling replications. Every
script writes to `results/`. Individual stages can
be run on their own, for example:

```bash
cd src
python3 part3_exhaustive/validate_harness.py   # run this first
python3 part1_data_defect/exchange_battery.py
```

## Revision analyses

`src/revision/` holds the analyses added in response to the referees. Release
`v2.0-revision1` is the first revision and `v2.1-revision2` the second. Run from `src/`:

```bash
python3 revision/fast_search.py              # fast engine, checked against the original harness
python3 revision/run_ceiling.py boot 2000 21 # noise ceiling; also gauss 1000 21, boot 500 5/63/126
python3 revision/summarise_ceiling.py
python3 revision/naive_stats.py              # effective number of trials, DSR, RC/SPA settings
python3 revision/screen_study.py             # size, power and false alarms of the open screen
python3 revision/mechanism.py                # where the free open goes wrong
python3 revision/carry_instability.py        # futures vs long-sample carry, step by step
python3 revision/part2_spot_tr.py            # spot universe on total returns
python3 revision/nonlinear_spreads.py        # nonlinear checks and OHLC spread estimates
python3 revision/text_numbers.py             # in-text numbers no other script prints
python3 revision/naive_tables.py             # Tables 2-11, the naive decomposition
python3 revision/npwhere_effect.py           # Section 10 before and after the np.where fix
python3 revision/screen_dependence.py        # the screen under dependence between sessions
python3 revision/figures_naive.py <dir>      # Figures 1-5
python3 revision/make_figures.py <dir>       # Figures 6-10
```

`fast_search.py` builds each of the 41 base signals once and applies the overlays in NumPy.
On the real data every one of the 1,312 daily net return series matches the original harness
exactly, which is what makes thousands of replications practical. `run_ceiling.py` saves the
full matrix of 1,312 Sharpe ratios per replication (`ceiling_*_L*.npy`); the per-replication
maxima are in `maxdist_*.npy`. `environment.txt` records the package versions used, and
`requirements.txt` pins them. Appendix C of the paper maps every table and figure to its
script and output file.

## Validate before you believe

`src/part3_exhaustive/validate_harness.py` runs ten checks on the evaluation code itself
and must pass before any result is trusted. The critical one is a deliberate lookahead
injection: weights formed from the same day's realised return, applied without the one
period shift, must produce an absurd Sharpe ratio. They give +24.34. Passed through the
harness the same weights give -0.11. A harness that cannot detect a strategy that cheats
cannot be trusted on one that does not.

The other nine cover leakage in each signal family, calibration on simulated zero drift
data, agreement of the observed best of *N* with theory, independent recomputation of
Sharpe, the Newey-West *t* statistic, the probabilistic Sharpe ratio and the cost
arithmetic and turnover accounting.

## The one thing worth taking away

Run your complete search many times on block-bootstrapped returns and report the
distribution of the best Sharpe ratio it produces, with the Monte Carlo *p*-value of your
real best. A result below that ceiling cannot be told apart from what the same search
finds on unpredictable returns under the chosen resampling scheme. Here the mean ceiling
was 0.725 to 0.747 across resampling schemes for the 1,312-configuration search. For the
five-strategy carry family and the canonical factor set (1,000 replications each) the real
best also sat below the 90th percentile of its null maximum. The Monte Carlo *p*-values
range from 0.21 to 0.77.

## Citation

```bibtex
@article{meena2026blindspots,
  title   = {Two Blind Spots in Backtest Validation: From a Manufactured Dollar Index
             Anomaly to a Corrected Null Result},
  author  = {Meena, Aryan and Pinsky, Eugene},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

MIT for the code. The data carries the terms described in `data/README.md`.
