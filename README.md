# dxy-fx-strategies
24 intraday trading strategies on DXY &amp; 6 FX pairs · 15-year backtest (2010–2025) · Best: +219% vs +22% B&amp;H · Sharpe 1.06 · Traders Magazine (est. 1982)

# DXY & FX Trading Strategy Research
> 📰 Short paper accepted — **Traders Magazine (est. 1982)**, 2026
> Research conducted at Boston University under Prof. Eugene Pinsky

## Overview
A two-track quantitative research project on the DXY U.S. Dollar Index and
6 major FX pairs (EUR, GBP, JPY, CAD, CHF, SEK) using **15 years of daily OHLC
data (2010–2025)**, covering both cross-sectional momentum and intraday sub-period strategies.

**Dataset:** 3,906 trading days (2010-12-20 → 2025-12-18), downloaded via yfinance,
validated (0 duplicates, 0 text contamination), USD-direction aligned, 6 rows removed
where >3 instruments had missing data on the same day.

---

## Track 1 — Cross-Sectional Momentum (data_processing.ipynb)
Ranks all 6 FX pairs by past returns across 7 horizons (1d, 5d, 10d, 21d, 63d, 126d, 252d),
then goes long the Top-3 and short the Bottom-3.

| Horizon | Top-3 Final Value | Bottom-3 Final Value | B&H Final Value | Bottom-3 Sharpe |
|---------|------------------|---------------------|----------------|----------------|
| Daily | $75.55 | **$165.27** | $121.03 | **0.48** |
| Quarterly | **$124.37** | $111.16 | $121.03 | 0.13 |

> Bottom-3 (short losers) at daily rebalancing: **+65% return** vs **+21% buy-and-hold**.

---

## Track 2 — Intraday Sub-Period Strategies (DXY_24.ipynb)
Backtested **24 intraday strategies** by decomposing each trading day into:
- **Overnight** (previous close → today's open)
- **Daytime** (today's open → today's close)

Position rules: Long, Short, Cash, Inertia (momentum), Reversal.

### DXY vs. Buy & Hold
| Strategy | Overnight | Daytime | Final Balance | Sharpe | Max Drawdown |
|----------|-----------|---------|--------------|--------|-------------|
| **#7 — Best (0 bps)** | Short | Long | **$318.80** | **1.06** | −13.1% |
| #20 — Best (1 bps) | Short | Reversal | $171.96 | 0.52 | −36.4% |
| **Buy & Hold (benchmark)** | — | — | **$121.72** | **0.22** | −15.3% |

Best intraday strategy: **+219% cumulative return vs. +22% buy-and-hold** (Sharpe 1.06 vs 0.22).

---

## Files
| File | Description |
|------|-------------|
| `data_processing.ipynb` | Data download (yfinance), cleaning, return layers, momentum strategy |
| `DXY_24.ipynb` | 24 intraday sub-period strategies on DXY + 6 FX pairs |
| `results/*_0bps.xlsx` | Full strategy × asset table (0 bps cost) |
| `results/*_1bps.xlsx` | Full strategy × asset table (1 bps cost) |
| `results/*_2bps.xlsx` | Full strategy × asset table (2 bps cost) |

## How to Run
```bash
pip install -r requirements.txt
# Step 1: data pipeline + momentum
jupyter notebook data_processing.ipynb
# Step 2: intraday strategies
jupyter notebook DXY_24.ipynb
```

## Author
Aryan Meena | [LinkedIn](https://linkedin.com/in/aryan-meena-32685415a) | araj7042@gmail.com
Boston University, MS Applied Data Analytics
