"""Data layer.

A position formed from information available at the close of day t earns the return from
t to t+1. The shift that enforces this lives in evaluate.apply_positions and nowhere
else, so a signal can use anything up to and including day t without having to remember
to lag it.

These are nearby continuous series. On the session where the front contract changes, the
printed price jump is the spread between two contracts rather than a return anyone earns.
Returns are computed within contract only. Roll sessions get a return of zero and are
flagged so that no signal or cost calculation treats them as tradeable moves."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

DIR = str(paths.FUTURES)
SYM = {"E6": "EURUSD", "B6": "GBPUSD", "D6": "CADUSD", "J6": "JPYUSD", "S6": "CHFUSD"}
PAIRS = list(SYM.values())
RATE_COL = {"EURUSD": "IR3TIB01EZM156N", "GBPUSD": "IR3TIB01GBM156N", "JPYUSD": "IR3TIB01JPM156N",
            "CADUSD": "IR3TIB01CAM156N", "CHFUSD": "IR3TIB01CHM156N", "SEKUSD": "IR3TIB01SEM156N"}
US_RATE = "IR3TIB01USM156N"


def _read(path):
    d = pd.read_csv(path, skipfooter=1, engine="python")
    # .values on every column: passing Series that carry their own RangeIndex alongside an
    # explicit index= makes pandas align them and silently produce an all-NaN frame.
    out = pd.DataFrame({
        "Sym": d["Symbol"].astype(str).values,
        "O": pd.to_numeric(d["Open"], errors="coerce").values,
        "H": pd.to_numeric(d["High"], errors="coerce").values,
        "L": pd.to_numeric(d["Low"], errors="coerce").values,
        "C": pd.to_numeric(d["Latest"], errors="coerce").values,
        "V": pd.to_numeric(d["Volume"], errors="coerce").values,
        "OI": pd.to_numeric(d["Open Int"], errors="coerce").values},
        index=pd.DatetimeIndex(pd.to_datetime(d["Time"]).values, name="Date"))
    if out["C"].isna().any() or (out["Sym"] == "nan").any():
        raise ValueError("%s: parsed NaNs in close or symbol, check the file layout" % path)
    return out.sort_index()


def load_raw():
    fut = {SYM[k]: _read(os.path.join(DIR, k + ".csv")) for k in SYM}
    dx = _read(os.path.join(DIR, "DX.csv"))
    return fut, dx


def panel():
    """Aligned daily panel across the five pairs and DX.

    Returns a dict of DataFrames indexed by date, columns = the five pairs:
      ret   within-contract simple return (0.0 on roll sessions)
      roll  boolean, True on a roll session
      close, open_, high, low, vol, oi
    plus DX equivalents as Series and the FRED rate panel.
    """
    fut, dx = load_raw()
    idx = None
    for d in fut.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    idx = idx.sort_values()

    def block(field):
        return pd.DataFrame({nm: d[field].reindex(idx) for nm, d in fut.items()})[PAIRS]

    roll = pd.DataFrame({nm: (d["Sym"] != d["Sym"].shift(1)).reindex(idx).fillna(True)
                         for nm, d in fut.items()})[PAIRS].astype(bool)
    close = block("C")
    raw_ret = close.pct_change()
    ret = raw_ret.where(~roll, 0.0)
    ret.iloc[0] = 0.0

    dxr_raw = dx["C"].pct_change()
    dx_roll = (dx["Sym"] != dx["Sym"].shift(1)).fillna(True)
    dxr = dxr_raw.where(~dx_roll, 0.0).fillna(0.0)
    dx_cont = 100.0 * (1 + dxr).cumprod()          # back-adjusted, roll gaps removed

    rates = pd.read_csv(str(paths.FRED_RATES), parse_dates=["Date"]).set_index("Date")
    rt = rates[[RATE_COL[p] for p in PAIRS] + [US_RATE]]
    rt.columns = PAIRS + ["USD"]
    rt = rt.reindex(idx).ffill()

    return dict(ret=ret, roll=roll, close=close, open_=block("O"), high=block("H"),
                low=block("L"), vol=block("V"), oi=block("OI"),
                dx_ret=dxr.reindex(idx).fillna(0.0), dx_close=dx_cont.reindex(idx).ffill(),
                dx_raw=dx["C"].reindex(idx).ffill(), rates=rt, index=idx)


# ---------------------------------------------------------------- integrity
def integrity_report(P):
    """Everything a referee would check about the inputs, before any strategy runs."""
    lines = []
    a = lines.append
    a("sessions %d | %s .. %s" % (len(P["index"]), P["index"][0].date(), P["index"][-1].date()))
    a("")
    a("%-9s %7s %7s %8s %9s %9s %10s %9s" %
      ("pair", "rows", "rolls", "NaN ret", "OHLC bad", "O==prevC", "med vol", "med OI"))
    for p in PAIRS:
        o, h, l, c = P["open_"][p], P["high"][p], P["low"][p], P["close"][p]
        bad = ((o < l) | (o > h) | (c < l) | (c > h)).sum()
        a("%-9s %7d %7d %8d %9d %8.2f%% %10.0f %9.0f" %
          (p, len(c), int(P["roll"][p].sum()) - 1, int(P["ret"][p].isna().sum()), int(bad),
           100 * np.isclose(o, c.shift(1), rtol=0, atol=1e-12).mean(),
           P["vol"][p].median(), P["oi"][p].median()))
    a("")
    a("DX: sessions %d | ann vol %.2f%% | total return %.1f%%" %
      (len(P["dx_ret"]), 100 * P["dx_ret"].std() * np.sqrt(252),
       100 * ((1 + P["dx_ret"]).prod() - 1)))
    a("")
    a("pair return correlation matrix:")
    a(P["ret"].corr().round(3).to_string())
    a("")
    a("annualised vol and mean by pair:")
    for p in PAIRS:
        r = P["ret"][p]
        a("   %-9s vol %5.2f%%  mean %+6.2f%%/yr  skew %+5.2f  kurt %5.1f  min %+.2f%% max %+.2f%%"
          % (p, 100 * r.std() * np.sqrt(252), 100 * r.mean() * 252,
             float(pd.Series(r).skew()), float(pd.Series(r).kurt()) + 3.0,
             100 * r.min(), 100 * r.max()))
    a("")
    a("rate coverage (3M interbank, forward filled):")
    for c in P["rates"].columns:
        s = P["rates"][c]
        a("   %-6s non-missing %5.1f%%   range %+.2f%% .. %+.2f%%"
          % (c, 100 * s.notna().mean(), s.min(), s.max()))
    return "\n".join(lines)


if __name__ == "__main__":
    P = panel()
    print(integrity_report(P))
