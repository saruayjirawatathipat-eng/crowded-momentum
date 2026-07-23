import os
from pathlib import Path

import pandas as pd

from binance_rest import fetch_klines_rest
from discover_symbols import list_usdt_symbols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = os.environ.get("CMC_START", "2019-01")


def _symbols() -> list[str]:
    # CMC_SYMBOLS lets a bounded/smoke run override real symbol discovery
    # with a small fixed list, so the pipeline can be wired up and validated
    # without a full archive-wide fetch.
    override = os.environ.get("CMC_SYMBOLS")
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    return list_usdt_symbols()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.utcnow().strftime("%Y-%m")
    symbols = _symbols()
    frames = []
    for sym in symbols:
        df = fetch_klines_rest(sym, "1d", START, end, columns=["open_time", "quote_volume"])
        if df.empty:
            continue
        df = df.rename(columns={"open_time": "date"})
        df["symbol"] = sym
        frames.append(df[["date", "symbol", "quote_volume"]])

    daily_volume = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["date", "symbol", "quote_volume"])
    daily_volume.to_parquet(DATA_DIR / "daily_volume.parquet", index=False)
    print(f"Fetched daily klines for {len(frames)}/{len(symbols)} candidate symbols")


if __name__ == "__main__":
    main()
