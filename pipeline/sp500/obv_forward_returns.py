from pathlib import Path

import pandas as pd

from forward_returns import compute_forward_returns, join_forward_returns_with_groups

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    price_df = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    groups_df = pd.read_parquet(DATA_DIR / "decile10_obv_groups.parquet")
    forward_returns_df = compute_forward_returns(price_df)
    panel_df = join_forward_returns_with_groups(groups_df, forward_returns_df)
    panel_df.to_parquet(DATA_DIR / "obv_forward_returns_summary.parquet", index=False)
    print(f"Wrote {len(panel_df)} stock-month-horizon rows")


if __name__ == "__main__":
    main()
