from pathlib import Path

import pandas as pd

from signals import compute_momentum, compute_net_taker_flow
from universe import apply_membership

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
    membership = pd.read_parquet(DATA_DIR / "universe.parquet")
    for tf in LADDER:
        panel = pd.read_parquet(DATA_DIR / PANEL_FILES[tf])
        panel = apply_membership(panel, membership)
        momentum = compute_momentum(panel, formation=12, skip=1)
        flow = compute_net_taker_flow(panel, window=11)
        merged = momentum.merge(flow, on=["open_time", "symbol"], how="inner")
        merged.to_parquet(DATA_DIR / f"signals_{tf}.parquet", index=False)
        print(f"{tf}: {len(merged)} momentum+flow signal rows")


if __name__ == "__main__":
    main()
