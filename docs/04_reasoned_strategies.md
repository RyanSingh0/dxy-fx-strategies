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
cent annual return, 1.10 per cent is the differential actually held and 0.23 per cent is
spot movement, which is the forward premium puzzle behaving as described. Turnover is 1.9
times a year, so the break-even cost is roughly 70 basis points against a real cost near
one.

Then the sample was extended.

| Sample | Currencies | Sharpe |
|---|---|---|
| Futures panel, 2011 to 2025 | 5 | +0.377 |
| Federal Reserve spot, 2002 to 2026 | 6 | -0.249 |
| 2002 to 2007 | 6 | +0.418 |
| 2008 to 2013 | 6 | -0.760 |
| 2014 to 2019 | 6 | -0.246 |
| 2020 to 2026 | 6 | -0.022 |

The futures panel begins in December 2010, immediately after the 2008 carry drawdown. The
positive result is a start date artefact. Dropping the yen alone takes it from 0.377 to
0.114.

## A hypothesis of ours that was falsified

We proposed that carry returns should scale with the cross sectional dispersion of carry,
by analogy with the value spread predicting value returns. Regressing carry returns on
lagged dispersion gives a slope of -0.000007 at *t* = -0.19 and *p* = 0.85 and sorting by
dispersion the high tercile is the worst at Sharpe 0.016 against 0.471 and 0.541. The
hypothesis is false and we do not claim it.

## The canonical factors on the long sample

| Factor | Sharpe | *p* |
|---|---|---|
| Carry | -0.249 | 0.21 |
| Momentum, 12 minus 1 month | +0.028 | 0.88 |
| Value, five year reversal | +0.096 | 0.63 |
| Combined at equal risk | -0.085 | 0.67 |

Best observed +0.096 against a noise ceiling whose ninetieth percentile is +0.41.

## Volatility targeting

Over 27 years it changes the Sharpe ratio by -0.009 on average across seven series and
improves three of them, which is inside the bootstrap null. The 2011 to 2025 improvement
was the same window artefact as carry. What it does deliver is a reduction in the
volatility of realised volatility from 0.025 to 0.015. That is risk control and not return
generation.

## Conclusion

No profitable strategy. The reason is structural. There is no directional predictability,
the only candidate risk premium in this universe is negative over the full sample and five
to six highly correlated currencies with a common factor of 54.8 per cent cannot support a
cross sectional strategy. What would change the answer is emerging market currencies where
carry dispersion actually lives, options where the variance risk premium is the most robust
premium in the asset class, or intraday data.
