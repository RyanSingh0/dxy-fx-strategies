# Data

## Sources

| Directory | Contents | Origin |
|---|---|---|
| `futures/` | `E6, B6, D6, J6, S6` CME currency futures and `DX` ICE Dollar Index futures, daily nearby OHLC with volume and open interest, 2010-12 to 2025-12 | Commercial vendor, exchange sourced (Barchart, downloaded 27 August 2026) |
| `vendor/` | `fx_ohlc_2010_2025.csv`, the free provider series used in the original study | Free redistribution of the ICE spot index |
| `fred/` | `fred_daily.csv` and `rates.csv` covering 2010 onward, `fred_long.csv` covering 1999 onward | Federal Reserve H.10 and OECD three month interbank rates, retrieved through FRED |

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
