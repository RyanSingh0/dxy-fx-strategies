"""Repository paths. Every script resolves data through here so the repository runs from
any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FUTURES = DATA / "futures"
FRED = DATA / "fred"
VENDOR = DATA / "vendor"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FUT_FILES = {"E6": "EURUSD", "B6": "GBPUSD", "D6": "CADUSD", "J6": "JPYUSD", "S6": "CHFUSD"}
DX_FILE = FUTURES / "DX.csv"
VENDOR_FILE = VENDOR / "fx_ohlc_2010_2025.csv"
FRED_DAILY = FRED / "fred_daily.csv"
FRED_LONG = FRED / "fred_long.csv"
FRED_RATES = FRED / "rates.csv"
