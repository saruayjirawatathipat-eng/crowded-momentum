from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_forward_returns(price_df: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    wide = price_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    frames = []
    for horizon in horizons:
        forward = wide.shift(-horizon) / wide - 1
        long = forward.stack().reset_index()
        long.columns = ["date", "ticker", "forward_return"]
        long["horizon"] = horizon
        frames.append(long.dropna(subset=["forward_return"]))
    return pd.concat(frames, ignore_index=True)


def join_forward_returns_with_groups(decile10_groups_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> pd.DataFrame:
    merged = decile10_groups_df[["date", "ticker", "group"]].merge(
        forward_returns_df, on=["date", "ticker"], how="inner"
    )
    return merged[["date", "ticker", "group", "horizon", "forward_return"]]


def main() -> None:
    price_df = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    groups_df = pd.read_parquet(DATA_DIR / "decile10_groups.parquet")
    forward_returns_df = compute_forward_returns(price_df)
    panel_df = join_forward_returns_with_groups(groups_df, forward_returns_df)
    panel_df.to_parquet(DATA_DIR / "forward_returns_summary.parquet", index=False)
    print(f"Wrote {len(panel_df)} stock-month-horizon rows")


if __name__ == "__main__":
    main()
