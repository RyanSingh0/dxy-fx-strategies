# Data

## Sources

| Directory | Contents | Origin |
|---|---|---|
| `futures/` | `E6, B6, D6, J6, S6` CME currency futures and `DX` ICE Dollar Index futures, daily nearby OHLC with volume and open interest, 2010-12 to 2025-12 | Commercial vendor, exchange sourced (Barchart, downloaded 27 and 28 August 2026) |
| `vendor/` | `fx_ohlc_2010_2025.csv`, the free provider series used in the original study | Free redistribution of the ICE spot index |
| `fred/` | `fred_daily.csv` and `rates.csv` covering 2010 onward, `fred_long.csv` covering 1956 onward (the long sample in the paper starts in January 1999) | Federal Reserve H.10, OECD three month interbank rates, and Federal Reserve and ECB policy rates, retrieved through FRED |

## Files

| File | Rows | Dates | Columns |
|---|---|---|---|
| `futures/E6.csv`, `B6.csv`, `D6.csv`, `J6.csv`, `S6.csv` | 3,800 each | 2010-12-01 to 2025-12-30 | Symbol (contract, e.g. `E6Z10`), Time, Open, High, Low, Latest, Change, %Change, Volume, Open Int |
| `futures/DX.csv` | 3,889 | 2010-12-01 to 2025-12-30 | as above |
| `vendor/fx_ohlc_2010_2025.csv` | 3,932 | 2010-12-01 to 2025-12-30 | Date, then Open, High, Low, Close for EURUSD, GBPUSD, CADUSD, CHFUSD, JPYUSD, SEKUSD and DXY |
| `fred/fred_daily.csv` | 3,946 | 2010-11-22 to 2026-01-05 | Date, DTWEXBGS, DTWEXAFEGS (Federal Reserve broad and advanced-foreign-economies dollar indexes), DEXUSEU, DEXJPUS, DEXUSUK, DEXCAUS, DEXSDUS, DEXSZUS (H.10 noon rates) |
| `fred/rates.csv` | 5,849 | 2010-01-01 to 2026-01-05 | Date, DTB3 (3-month T-bill), DFF (fed funds), ECBDFR, ECBMRRFR (ECB deposit and main refinancing rates), IR3TIB01 three month interbank rates for JP, GB, CA, SE, CH, EZ, US (monthly) |
| `fred/fred_long.csv` | 14,886 | 1956-01-01 to 2026-08-21 | Date, DEXUSEU, DEXUSUK, DEXJPUS, DEXCAUS, DEXSDUS, DEXSZUS, IR3TIB01 three month interbank rates for EZ, GB, JP, CA, SE, CH, US (monthly; start dates differ by country, the euro rate from 1999 and the yen rate from 2002) |

Each futures file ends with one line that is not data ("Downloaded from Barchart.com as of
..."), which `src/data.py` drops. The rate columns are monthly or have gaps on holidays;
the loaders forward-fill them onto the trading calendar.

## Licensing

The FRED series are public domain and may be redistributed freely. The futures files are
redistributed here only so that the results in the paper can be reproduced. Anyone
intending to use them for another purpose should check the vendor's redistribution terms
first. Anyone who cannot use them can download the same nearby continuous series from any
exchange-sourced vendor in the same layout (symbol, date, open, high, low, last, volume,
open interest) and every script will run. The free-series, Federal Reserve spot and
long-sample results need only the public files.

## Refreshing the long FRED sample

`fred_long.csv` can be rebuilt at any time from the public API:

```python
import urllib.request, pandas as pd, io
series = ["DEXUSEU","DEXUSUK","DEXJPUS","DEXCAUS","DEXSDUS","DEXSZUS",
          "IR3TIB01EZM156N","IR3TIB01GBM156N","IR3TIB01JPM156N",
          "IR3TIB01CAM156N","IR3TIB01SEM156N","IR3TIB01CHM156N","IR3TIB01USM156N"]
frames = {}
for s in series:
    raw = urllib.request.urlopen(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + s).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["Date", s]
    frames[s] = df.assign(Date=pd.to_datetime(df.Date)).set_index("Date")[s]
pd.concat(frames, axis=1).sort_index().to_csv("fred/fred_long.csv")
```

## Roll handling

The futures files are nearby continuous series. The printed price jumps on the session
where the front contract changes and that jump is the spread between two contracts rather
than a return any holder earns. `src/data.py` computes returns within contract only and
assigns zero to the 61 roll sessions in each series. Do not take a naive `pct_change` of
the close column.
