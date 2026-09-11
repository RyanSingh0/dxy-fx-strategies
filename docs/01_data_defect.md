# Part 1. A data defect that the diagnostics certify

Script output: `results/part1_*.txt`

## What was claimed

Splitting the daily return of the U.S. Dollar Index into an overnight leg, previous close
to open and a daytime leg, open to close, produced an overnight leg with a Sharpe ratio of
-1.36 at *p* < 0.0001 on the free provider series. That is a very large effect.

## What is actually there

On exchange sourced Dollar Index futures over the same sessions the overnight leg has a
Sharpe ratio of +0.25 at *p* = 0.33. Nothing. Both figures use the 3,793 overnight returns
common to the two series. On its own full sample the free series gives -1.49 and the
exchange series on its own gives +0.269.

## Isolating the cause

A naive substitution of one vendor's prices into the other's series is invalid here. The
two series carry a persistent level offset of 0.041 index points, which is 6.7 per cent of a
typical daily range, so a price level substitution injects that offset into every return.
Done that way it produced a terminal value of 1178 for one strategy, which is an artefact.

The valid version transfers the close to open **return** so each row uses one vendor's
price level throughout.

| Price level, opening return | Overnight final | Sharpe | *p* |
|---|---|---|---|
| Free, free open | 61.68 | -1.358 | 0.0000 |
| Free, exchange open | 108.08 | +0.252 | 0.3283 |
| Exchange, exchange open | 108.08 | +0.252 | 0.3283 |
| Exchange, free open | 61.68 | -1.358 | 0.0000 |

The overnight leg follows whichever vendor's open is used and is unaffected by whose closes
are used. The opening prices are the whole story.

With the offset removed, the free open lies outside the exchange high to low range on 7.30
per cent of sessions. On the 204 sessions where it equals its own daily low it sits at
-0.073 of the exchange range and falls outside that range 56.9 per cent of the time against
3.5 per cent on ordinary sessions. A price outside the session range is not a traded price.

## The part that matters

| Series | White RC *p* | Hansen SPA *p* | Romano-Wolf survivors | PBO |
|---|---|---|---|---|
| Free provider (the artefact) | 0.1032 | 0.0132 | 3 of 24 | 0.016 to 0.028 |
| Exchange futures (correct) | 0.9924 | 1.0000 | 0 of 24 | 0.765 to 0.779 |

The diagnostics rank the false result far above the true one. Both readings are correct on
their own terms. The probability of backtest overfitting asks whether an in-sample winner
keeps winning out of sample and a persistent data defect does exactly that. A stationary
artefact is indistinguishable from a real anomaly to every tool in this family.

## The screen

Under the null that a recorded open is an unbiased draw from the session, it is equally
likely to equal the low as the high. Counting the two gives an exact binomial test needing
no second data source. It clears the exchange series at *p* = 0.788 and rejects the
corrupted one at *p* = 3.68e-13.

Two honest caveats. It also flags EURUSD at *p* = 0.0061, which barely survives Bonferroni
across eight tests at 0.00625 and we cannot prove that is a true positive rather than a
multiplicity artefact. And it is validated against exactly one instrument with a known good
reference, so it is a promising diagnostic rather than an established one.

## Corrections made during this work

The claim that the free open falls outside the true range on 7.3 per cent of sessions was
first computed by comparing the index open against the futures range, which are different
instruments. It was retracted and recomputed after removing the level offset. The mean
position of the open within the range, 0.479 against 0.499, was also over-claimed as a
discriminator; the count of extremes carries the information, not the average.
