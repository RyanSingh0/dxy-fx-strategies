# Part 2. An undeclared search space

Script output: `results/part2_*.txt`

## The claim

A regime conditioned cross sectional currency strategy, ranking six currencies each quarter
and holding the two weakest with a sign set by the 126 day momentum of the dollar index,
reported a Sharpe ratio of 0.79 with a Newey-West *t* of 3.09 and *p* = 0.0020.

## Three defects found in the implementation

1. `to_period('2Q')` silently returns quarterly periods in pandas, so the reported
   semiannual row was a byte identical duplicate of the quarterly row.
2. The grid loop was wrapped in a bare `except: pass`, which concealed that the annual
   frequency alias had been renamed between pandas versions. Every annual configuration was
   silently discarded and the reported grid size was wrong.
3. The constituent price files are date stamped one trading day late relative to the index
   file, confirmed against FRED.

## Declaring the search

The published account reported six configurations. Rebalance frequency, cross sectional
slice, cardinality, the regime filter and the momentum lookback are all free parameters.
Enumerated in full that is 144 configurations on the tradeable five currency panel and 216
on the six currency panels.

| Universe | Configs | Sharpe | White RC *p* | Hansen SPA *p* | DSR |
|---|---|---|---|---|---|
| Exchange futures, 5 currencies | 144 | 0.540 | 0.2077 | 0.3077 | 0.338 |
| Futures plus krona spot, 6 | 216 | 0.587 | 0.0823 | 0.0907 | 0.322 |
| Federal Reserve spot, 6 | 216 | 0.792 | 0.0470 | 0.0650 | 0.701 |

No configuration survives stepwise familywise error control in any universe. The deflated
Sharpe ratio is a confidence level requiring more than 0.95 and reaches 0.701 at best.

## It is a spike, not a plateau

Holding everything else fixed and moving one parameter:

| Regime lookback | none | 63d | **126d** | 252d |
|---|---|---|---|---|
| Sharpe | -0.031 | -0.121 | **+0.540** | +0.110 |

Positive at one value and negative at two neighbours. That describes a noisy surface.

## A benchmark error, in the authors' favour

The committed statistics tested against a long dollar position. The strategy is a self
financing long and short book, so the correct null is zero. Both are reported.

## The ensemble

An equal weight ensemble across a family selects nothing and needs no correction. Costed in
weight space so that offsetting trades across members are never charged, the ensemble of all
regime signed configurations gives a Sharpe ratio of 0.207 at *p* = 0.42.
