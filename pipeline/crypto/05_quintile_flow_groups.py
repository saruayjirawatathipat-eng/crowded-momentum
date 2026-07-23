from pathlib import Path

import pandas as pd

from groups import split_by_median_flow, top_quintile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LADDER = ["monthly", "weekly", "daily", "4h", "1h"]


def main() -> None:
    for tf in LADDER:
        signals = pd.read_parquet(DATA_DIR / f"signals_{tf}.parquet")
        momentum = signals[["open_time", "symbol", "momentum"]]
        top = top_quintile(momentum)
        merged = top.merge(signals[["open_time", "symbol", "flow"]], on=["open_time", "symbol"], how="inner")
        grouped = split_by_median_flow(merged)
        grouped.to_parquet(DATA_DIR / f"groups_{tf}.parquet", index=False)
        print(f"{tf}: {len(grouped)} grouped rows")


if __name__ == "__main__":
    main()
