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

Replication code and data for the paper *Two Blind Spots in Backtest Validation: Evidence
from the U.S. Dollar Index* (Meena and Pinsky).

Modern backtest validation defends against one failure, that too many strategies were
tried. This repository documents two failures that defence cannot see, using the U.S.
Dollar Index and its constituent currency futures.

1. **A corrupted input.** A free data source records session opening prices with a stale
   print defect that manufactures an overnight anomaly of Sharpe -1.36. The anomaly is
   absent from exchange futures on the same instrument. The overfitting diagnostics do not
   miss the artefact, they certify it: the probability of backtest overfitting is 0.02 for
   the false result and 0.77 for the correct data.
2. **An undeclared search.** A strategy reporting Sharpe 0.79 with a Newey-West *t* of
   3.09 fails every correction once its full 216 configuration search is declared instead
   of the six configurations originally reported.

The repository then runs a correctly specified search of 1,312 published specifications
and finds none exceeding the **noise ceiling**, the Sharpe ratio the identical search
attains on block bootstrapped returns.

## Headline numbers

| Result | Value |
|---|---|
| Overnight Sharpe, free provider series | -1.36 (*p* < 0.0001) |
| Overnight Sharpe, exchange futures | +0.25 (*p* = 0.33) |
| Probability of backtest overfitting, artefact | 0.016 to 0.028 |
| Probability of backtest overfitting, correct data | 0.765 to 0.779 |
| Open location binomial screen, artefact | *p* = 3.68e-13 |
| Open location binomial screen, exchange | *p* = 0.788 |
| Best of 1,312 configurations, net of costs | +0.653 |
| Noise ceiling, block bootstrap | +0.712 |
| Configurations above the ceiling | **0 of 1,312** |

## Layout

```
data/          futures, free provider and FRED inputs; see data/README.md
src/
  paths.py                 all file locations resolve here
  data.py                  aligned panel, roll handling, integrity report
  evaluate.py              the one period shift, Sharpe, HAC t, bootstrap, RC/SPA/RW, DSR, CSCV
  signals.py               the 1,312 configuration search space
  part1_data_defect/       the corrupted input and the screen
  part2_declared_search/   the undeclared search space
  part3_exhaustive/        harness validation, the full search, verification
  part4_reasoned/          strategies built from the structure rather than enumerated
results/       text output of every script
paper/         main.tex and the MDPI layout emulation
docs/          longer write-ups of each stage
```

## Reproducing

```bash
pip install -r requirements.txt
./run_all.sh
```

Expect roughly 60 to 90 minutes. Every script writes to `results/`. Individual stages can
be run on their own, for example:

```bash
cd src
python3 part3_exhaustive/validate_harness.py   # run this first
python3 part1_data_defect/exchange_battery.py
```

## Validate before you believe

`src/part3_exhaustive/validate_harness.py` runs ten checks on the evaluation code itself
and must pass before any result is trusted. The critical one is a deliberate lookahead
injection: weights formed from the same day's realised return, applied without the one
period shift, must produce an absurd Sharpe ratio. They give +24.34. Passed through the
harness the same weights give -0.11. A harness that cannot detect a strategy that cheats
cannot certify one that does not.

The other nine cover leakage in each signal family, calibration on simulated zero drift
data, agreement of the observed best of *N* with theory, independent recomputation of
Sharpe, the Newey-West *t* statistic, the probabilistic Sharpe ratio and the cost
arithmetic and turnover accounting.

## The one thing worth taking away

Run your complete search once on block bootstrapped returns and report the best Sharpe
ratio it produces. Any real result below that number is indistinguishable from luck at
your search size. Here the ceiling was 0.71 to 0.75 for the 1,312 configuration search,
0.38 for a five strategy carry family and 0.41 for the canonical factor set. Every real
result fell below its own ceiling.

## Citation

```bibtex
@article{meena2026blindspots,
  title   = {Two Blind Spots in Backtest Validation: Evidence from the U.S. Dollar Index},
  author  = {Meena, Aryan and Pinsky, Eugene},
  journal = {Risks},
  year    = {2026}
}
```

## License

MIT for the code. The data carries the terms described in `data/README.md`.
