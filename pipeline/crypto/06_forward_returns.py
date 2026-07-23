from pathlib import Path

import pandas as pd

from crypto_forward_returns import compute_forward_returns, join_forward_returns_with_groups

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LADDER = ["monthly", "weekly", "daily", "4h", "1h"]
PANEL_FILES = {
    "monthly": "klines_monthly.parquet",
    "weekly": "klines_weekly.parquet",
    "daily": "klines_daily.parquet",
    "4h": "klines_4h.parquet",
    "1h": "klines_1h_panel.parquet",
}


def main() -> None:
    for tf in LADDER:
        panel = pd.read_parquet(DATA_DIR / PANEL_FILES[tf])
        forward = compute_forward_returns(panel, horizons=(1, 2, 4))
        groups = pd.read_parquet(DATA_DIR / f"groups_{tf}.parquet")
        joined = join_forward_returns_with_groups(groups, forward)
        joined.to_parquet(DATA_DIR / f"forward_returns_{tf}.parquet", index=False)
        print(f"{tf}: {len(joined)} forward-return rows")


if __name__ == "__main__":
    main()
