# Part 4. Strategies built from the structure

Script output: `results/part4_*.txt`

An exhaustive search that finds nothing invites the objection that the search was the wrong
instrument. So we also proceeded analytically.

## What the data looks like

| Question | Answer | Evidence |
|---|---|---|
| Is direction predictable? | No | AR(1) *R²* on returns 0.0000 to 0.0011; variance ratios 0.83 to 1.02; Ljung-Box *p* = 0.24 |
| Is volatility predictable? | Overwhelmingly | AR(1) *R²* on log realised volatility 0.966 to 0.972 |
| Is there a tradeable cross section? | Barely | The dollar factor is 54.8 per cent of variance; residual half lives 158 to 726 days |
| Is the index mispriced against its basket? | No | Matched expiry on 98.2 per cent of sessions; 98 per cent of the tracking error is the unhedged krona |
| Calendar effects? | None | Day of week all *p* > 0.27; every month end position *p* > 0.10 |

Volatility is roughly three orders of magnitude more forecastable than direction. If
direction cannot be forecast then only a genuine risk premium or a relationship true by
construction can pay.

## The basket convergence trade

The dollar index is an exact deterministic function of its six constituents and the index
future shares an expiry code with the five currency futures on 98.2 per cent of sessions.
It is also efficiently priced: Sharpe +0.09 gross, -0.06 at half a basis point per leg and
-0.20 at one basis point. Six legs against a spread whose daily change has a standard
deviation of five basis points.

## Carry, the serious candidate

The roll gap between the expiring and deferred contract is the market's own quoted interest
differential, validated against Federal Reserve interbank differentials to within 0.43
percentage points per currency.

On the futures panel it returns Sharpe 0.377 and the mechanism verifies. Of the 1.34 per
cent annual gross return, 1.10 per cent is the differential actually held and 0.23 per cent
is spot movement, which is the forward premium puzzle behaving as described. Against a
noise ceiling for this five-strategy family (1,000 replications) the 0.377 sits below the
90th percentile of the null maximum (0.470), Monte Carlo *p* = 0.21.

Then the sample was extended with Federal Reserve spot rates and OECD interbank rates.
The first submission did this on spot changes alone, which leaves out the interest
differential, the income a carry position exists to earn, and reported -0.249. That figure
is withdrawn. Every long-sample number below is on total returns, spot change plus the
differential known at the previous close (`src/revision/carry_instability.py`).

| Step | Change made | Sharpe |
|---|---|---|
| (0) Spot change only, 2002 to 2026, withdrawn | | -0.249 |
| (1) Spot plus differential, 2002 to 2026 | return definition | +0.019 |
| (2) Same, 2010-12 to 2025-12 | sample period | +0.059 |
| (3) Same window, krona dropped | universe | -0.004 |
| (4) Futures returns, OECD interbank signal | instrument | +0.128 |
| (5) Futures returns, roll-implied signal | signal | +0.377 |
| (4') Spot total returns, roll-implied signal | signal, from (3) | +0.318 |

The signal accounts for +0.25 to +0.32 of the gap and the instrument for +0.06 to +0.13,
depending on the order of the last two steps. The sample period adds little once returns
are defined consistently.

| Long sample, total return | Sharpe | *p* |
|---|---|---|
| 2002 to 2007 | +0.952 | 0.026 |
| 2008 to 2013 | -0.626 | 0.106 |
| 2014 to 2019 | -0.070 | 0.865 |
| 2020 to 2026 | +0.380 | 0.306 |

The swing coincides with the 2008 carry crash, but a HAC Wald test of equal means across the
four periods gives *p* = 0.064, so it is not significant at 5 per cent.

## A hypothesis of ours that was falsified

We proposed that carry returns should scale with the cross sectional dispersion of carry,
by analogy with the value spread predicting value returns. Regressing carry returns on
lagged dispersion gives a slope of -0.000007 at *t* = -0.19 and *p* = 0.85 and sorting by
dispersion the high tercile is the worst at Sharpe 0.016 against 0.471 and 0.541. The
hypothesis is false and we do not claim it.

## The canonical factors on the long sample

On total returns.

| Factor | Sharpe | *p* |
|---|---|---|
| Carry | +0.019 | 0.93 |
| Momentum, 12 minus 1 month | +0.063 | 0.75 |
| Value, five year reversal | +0.097 | 0.62 |
| Combined at equal risk | +0.122 | 0.54 |

Best observed +0.122 against a noise ceiling with mean +0.174 and 90th percentile +0.371
(1,000 replications, Monte Carlo *p* = 0.62).

## Volatility targeting

Over 27 years of price returns it changes the Sharpe ratio of a long position by -0.009 on
average across seven series and improves three of them. On total returns from 2002 the
average change is -0.026, again three of seven. For the dollar the change is inside the
bootstrap null. What it does deliver is a reduction in the volatility of realised volatility
from 0.025 to 0.015. That is risk control and not return generation.

## Conclusion

No strategy here survives correction for the search, within the scope tested: five CME
currency futures and the ICE dollar index futures from December 2010 to December 2025, and
six G10 currencies on Federal Reserve and OECD data from 2002, at a daily horizon, with
proportional costs of one basis point per unit of turnover. None of this is an estimate of
attainable trading profit, since the costs are assumptions. There is no evidence of linear
directional predictability, the one candidate risk premium is close to zero over the full
sample on total returns, and five to six highly correlated currencies with a common factor
of 54.8 per cent leave little room for a cross sectional strategy. Emerging market
currencies, options and intraday data lie outside what was tested.
