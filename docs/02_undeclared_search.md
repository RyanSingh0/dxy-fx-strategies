# Part 2. An undeclared search space

Script output: `results/part2_*.txt`, `results/revision/npwhere_effect.txt`,
`results/revision/part2_spot_tr.txt`

## The claim

A regime conditioned cross sectional currency strategy, ranking six currencies each quarter
and holding the two weakest with a sign set by the 126 day momentum of the dollar index,
reported a Sharpe ratio of 0.79 with a Newey-West *t* of 3.09 and *p* = 0.0020 on Federal
Reserve spot data, as originally coded. With the implementation defects below corrected the
same configuration gives 0.72, *t* = 2.78, *p* = 0.0055.

## Problems found in the implementation

Implementation defects, which make the numbers wrong for the stated specification:

1. `to_period('2Q')` silently returns quarterly periods in pandas, so the reported
   semiannual row was a byte identical duplicate of the quarterly row.
2. The grid loop was wrapped in a bare `except: pass`, which concealed that the annual
   frequency alias had been renamed between pandas versions. Every annual configuration was
   silently discarded and the reported grid size was wrong.
3. `np.where(mom < 0, 1, -1)` maps a missing momentum value to -1, so over the first
   lookback window the book traded as though the dollar were rising. A signed configuration
   now holds nothing until its lookback exists.
4. The free provider's constituent price files are date stamped one trading day late
   relative to its index file. Nothing in this part uses them.

Specification choices, stated and reported both ways: the null for the multiple-testing
corrections (zero, since the book is self financing, against the long dollar the committed
code used), the calendar on which the momentum lookback is counted, and whether spot
returns include the interest differential. Search decision: six configurations reported
out of 216.

## What defect 3 changes

`src/revision/npwhere_effect.py` reruns everything twice with nothing else changed.

| | Futures, 5 | | Futures + krona, 6 | | Fed spot, 6 | |
|---|---|---|---|---|---|---|
| | committed | corrected | committed | corrected | committed | corrected |
| Headline Sharpe | 0.540 | 0.458 | 0.587 | 0.498 | 0.792 | 0.722 |
| White RC *p*, null zero | 0.2077 | 0.2333 | 0.0823 | 0.2203 | 0.0470 | 0.0963 |
| Hansen SPA *p*, null zero | 0.3077 | 0.3833 | 0.0907 | 0.2830 | 0.0650 | 0.1570 |
| DSR | 0.337 | 0.227 | 0.321 | 0.217 | 0.701 | 0.612 |

Before the fix the spot universe rejected at 5 per cent on the Reality Check over the full
grid, and on the Reality Check and SPA over the 36-configuration subset (0.015 and 0.023, one
Romano-Wolf survivor). After it none of these rejects. Nothing on the futures panels
rejected before or after.

## Declaring the search

Rebalance frequency (6), cross sectional slice (3), cardinality (3) and the regime filter
(none, or a 63, 126 or 252 session lookback) give 216 configurations on the six currency
panels. Cardinality three is infeasible with five currencies, which leaves 144.

| Universe | Configs | Sharpe | White RC *p* | Hansen SPA *p* | DSR |
|---|---|---|---|---|---|
| Exchange futures, 5 currencies | 144 | 0.458 | 0.2333 | 0.3833 | 0.227 |
| Futures plus krona spot, 6 | 216 | 0.498 | 0.2203 | 0.2830 | 0.217 |
| Federal Reserve spot, 6 | 216 | 0.722 | 0.0963 | 0.1570 | 0.612 |
| Federal Reserve spot, total return, 6 | 216 | 0.780 | 0.0470 | 0.0877 | 0.678 |

Null of zero excess return. No configuration survives stepwise familywise error control in
any universe. The deflated Sharpe ratio is a probability, not a *p*-value, and needs more
than 0.95. The Sharpe column is the headline configuration on its own sessions; the tests
use the sessions on which every configuration trades (from January 2012).

## It is a spike, not a plateau

Holding everything else fixed on the futures panel and moving one parameter:

| Regime lookback | none | 63d | **126d** | 252d |
|---|---|---|---|---|
| Sharpe | -0.319 | +0.136 | **+0.458** | +0.259 |

Rebalance frequency gives -0.234 daily, -0.081 weekly, +0.047 monthly, +0.458 quarterly,
+0.112 semiannually and +0.456 annually.

## The ensemble

An equal weight ensemble across a family selects nothing and needs no correction. The 36
regime signed configurations in the bottom slice give 0.148 (*p* = 0.57); all 108 regime
signed configurations give -0.160 (*p* = 0.53).
