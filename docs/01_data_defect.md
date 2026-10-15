# Part 1. A data defect that the diagnostics rank above the comparator

Script output: `results/part1_*.txt`, `results/revision/mechanism.txt`,
`results/revision/screen_study.txt`, `results/revision/screen_dependence.txt`

## What was claimed

Splitting the daily return of the U.S. Dollar Index into an overnight leg, previous close
to open, and a daytime leg, open to close, produced an overnight leg with a Sharpe ratio of
-1.36 at *p* < 0.0001 on the free provider series. That is a very large effect.

## What the comparator shows

On exchange sourced Dollar Index futures over the same sessions the overnight leg has a
Sharpe ratio of +0.25 at *p* = 0.33, indistinguishable from zero. Both figures use the
3,793 overnight returns common to the two series. On its own full sample the free series
gives -1.43 and the exchange series +0.27. The futures are a different instrument from the
spot index, so they are a comparator, not ground truth.

## Isolating the open

A naive substitution of one vendor's prices into the other's series is invalid here. The
two series carry a persistent level offset of 0.041 index points, which is 6.7 per cent of a
typical daily range, so a price level substitution injects that offset into every return.
Done that way (free closes with exchange opens) Strategy 7 ends at 1,179, which is the
offset and not a finding.

The valid version transfers the close to open **return** so each row uses one vendor's
price level throughout.

| Price level, opening return | Overnight final | Sharpe | *p* |
|---|---|---|---|
| Free, free open | 61.68 | -1.358 | 0.0000 |
| Free, exchange open | 108.08 | +0.252 | 0.3280 |
| Exchange, exchange open | 108.08 | +0.252 | 0.3280 |
| Exchange, free open | 61.68 | -1.358 | 0.0000 |

The overnight leg follows whichever vendor's open is used and is unaffected by whose closes
are used. That locates the difference in the opening prices. It does not say why they
differ; `src/revision/mechanism.py` narrows that down, and the mechanism stays inferred.

With the offset removed, the free open lies outside the exchange high to low range on 7.30
per cent of sessions. On the 204 sessions where it equals its own daily low it sits at
-0.073 of the exchange range and falls outside that range 56.9 per cent of the time against
3.5 per cent on ordinary sessions.

## What the diagnostics say

| Series | White RC *p* | Hansen SPA *p* | Romano-Wolf survivors | PBO |
|---|---|---|---|---|
| Free provider (defective opens) | 0.1032 | 0.0132 | 3 of 24 | 0.016 to 0.028 |
| Exchange futures (comparator) | 0.9924 | 1.0000 | 0 of 24 | 0.765 to 0.779 |

The diagnostics rank the free series' result far above the comparator's. Both readings are
correct on their own terms. The probability of backtest overfitting asks whether an
in-sample winner keeps winning out of sample, and a persistent data defect does exactly
that.

## The screen

The null is a symmetry condition: the open is as likely to equal the session low as the
session high. Counting the two gives an exact binomial test needing no second data source.
It clears the exchange series at *p* = 0.788 and flags the free series at *p* = 3.68e-13.

That exact *p*-value assumes independent sessions. In the free series the flagged sessions
cluster (runs test *z* = -4.20, most of them in 2013). Newey-West versions of the test still
flag it (*p* = 7e-7, 2e-4 and 0.013 at 8, 21 and 63 lags), a one-vote-per-year sign test
does not (10 of 15 years, *p* = 0.30). The futures series show no such clustering.

The screen also flags EURUSD at *p* = 0.0061, which barely survives Bonferroni across eight
tests, and sterling futures at *p* = 0.015, which does not survive across six. It has been
studied on one instrument complex, two vendors and simulations. It is a screen whose
rejection calls for investigation, not a general test for corruption, and it cannot see
errors that are symmetric in direction.

## Corrections made during this work

The claim that the free open falls outside the true range on 7.3 per cent of sessions was
first computed by comparing the index open against the futures range, which are different
instruments. It was retracted and recomputed after removing the level offset. The mean
position of the open within the range, 0.479 against 0.499, was also over-claimed as a
discriminator; the count of extremes carries the information, not the average.
