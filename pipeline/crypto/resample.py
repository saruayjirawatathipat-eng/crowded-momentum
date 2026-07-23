from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "taker_buy_volume": "sum",
}

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df.set_index("open_time")
        .resample(rule, label="left", closed="left")
        .agg(_AGG)
        .dropna(subset=["open"])
        .reset_index()
    )
    return out
