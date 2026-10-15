# Part 3. A correctly specified search and the noise ceiling

Script output: `results/part3_*.txt`

## Validate the harness before believing anything

`src/part3_exhaustive/validate_harness.py` runs ten checks. All pass.

| Check | Result |
|---|---|
| Perfect foresight weights applied without the shift | Sharpe +24.34 |
| The same weights through the harness | Sharpe -0.11 |
| One extra lag on 14 real signal families | none collapses |
| Full search on simulated zero drift returns, White RC *p* | 0.20, 0.07, 0.69, 0.38 |
| Observed best of *N* on noise against theory | 0.823 against 0.846 |
| Sharpe, Newey-West *t*, PSR, cost arithmetic | reproduced to machine precision |
| Buy and hold turnover; daily sign flipper turnover | 0.00; exactly 2.0 per session |
| Winning configuration rebuilt from first principles | maximum difference 0.0e+00 |

The first two matter most. A harness that cannot detect a strategy which cheats cannot
be trusted on one that does not.

## The search space

1,312 configurations from published specifications: time series momentum, cross sectional
currency momentum, carry, value, moving average rules, channel breakouts, open interest
growth and volatility scaling, each crossed with four rebalance frequencies, the regime
overlay at three lookbacks and with or without volatility targeting. Costs are one basis
point per unit of turnover.

## The noise ceiling

The benchmark is not zero. It is the Sharpe ratio the same search attains on returns with
no predictable structure. We resample the actual returns in blocks using the stationary
bootstrap, which preserves the marginal distributions, the fat tails and the cross sectional
correlation while destroying the time series structure a signal could exploit, then run the
whole search on the resampled panel.

| Quantity | Value |
|---|---|
| Best configuration on real data, net of costs | +0.653 |
| Noise ceiling, block bootstrap (mean block 21), 2,000 replications: mean (MC s.e.) | +0.737 (0.003) |
| 90th and 95th percentiles | +0.887, +0.940 |
| Monte Carlo *p* of the observed best | 0.767 |
| Noise ceiling, correlated Gaussian null, 1,000 replications: mean; *p* | +0.725; 0.746 |
| Expected best of 1,312 under the null | +0.865 |
| **Configurations exceeding the ceiling** | **0 of 1,312** |
| White RC *p* | 0.659 |
| Hansen SPA *p* | 0.773 |
| Romano-Wolf survivors | 0 |
| Deflated Sharpe ratio (needs > 0.95) | 0.203 |
| Harvey-Liu adjusted *p*, Bonferroni and BHY | 1.000, 1.000 |
| Probability of backtest overfitting | 0.556 |

The best result found in fifteen years of exchange data sits at the 23rd percentile of what
the same search finds on block-bootstrapped returns. Across mean block lengths of 5 to 126
sessions and a Gaussian null the mean ceiling is 0.725 to 0.747 and the Monte Carlo *p*
0.75 to 0.79 (`results/revision/ceiling_summary.txt`). The first submission used six
replications (0.712 bootstrap, 0.752 Gaussian); the first six replications of each reproduce those
values exactly. The ceiling is a benchmark under the chosen resampling scheme, and the
Monte Carlo *p*-value is the test.

## Holdout

Selecting on 2011 to 2018 and evaluating once on 2019 to 2025, the in-sample winner ranks
163rd of 1,312 out of sample. The top ten in-sample configurations average +0.019.

## Family ensembles

Costed in weight space. Selecting nothing means no correction is required.

| Family | Sharpe | *p* | | Family | Sharpe | *p* |
|---|---|---|---|---|---|---|
| Cross sectional momentum | +0.218 | 0.37 | | Time series momentum | -0.214 | 0.37 |
| Value | +0.204 | 0.43 | | Volume | -0.232 | 0.37 |
| Carry | +0.135 | 0.59 | | Breakout | -0.289 | 0.23 |
| Reversal | +0.133 | 0.58 | | Moving average | -0.341 | 0.16 |
| Open interest | +0.094 | 0.72 | | Cross sectional reversal | -0.360 | 0.14 |

Not one is significant in either direction.

## An error found and fixed during verification

The first version of the ensemble calculation averaged the net returns of the members,
which charges every member's full turnover. A real ensemble holds the average weight and
offsetting trades never reach the market. Rebuilt in weight space, the all-configuration
ensemble moved from -0.820 to -0.176.
