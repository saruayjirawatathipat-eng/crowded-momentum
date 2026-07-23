from pathlib import Path

import pandas as pd

from resample import resample_ohlcv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KLINES_1H = DATA_DIR / "klines_1h"

TIMEFRAME_RULES = {"4h": "4h", "daily": "1D", "weekly": "1W-MON", "monthly": "1MS"}


def main() -> None:
    per_symbol = {path.stem: pd.read_parquet(path) for path in sorted(KLINES_1H.glob("*.parquet"))}

    # 1h panel: same long-panel shape as the resampled TFs, no resampling needed.
    hourly_frames = []
    for symbol, df in per_symbol.items():
        d = df.copy()
        d["symbol"] = symbol
        hourly_frames.append(d)
    hourly_panel = pd.concat(hourly_frames, ignore_index=True)
    hourly_panel.to_parquet(DATA_DIR / "klines_1h_panel.parquet", index=False)
    print(f"1h: wrote {len(hourly_panel)} rows across {hourly_panel['symbol'].nunique()} symbols")

    for tf, rule in TIMEFRAME_RULES.items():
        frames = []
        for symbol, df in per_symbol.items():
            resampled = resample_ohlcv(df, rule)
            resampled["symbol"] = symbol
            frames.append(resampled)
        panel = pd.concat(frames, ignore_index=True)
        panel.to_parquet(DATA_DIR / f"klines_{tf}.parquet", index=False)
        print(f"{tf}: wrote {len(panel)} rows across {panel['symbol'].nunique()} symbols")


if __name__ == "__main__":
    main()
