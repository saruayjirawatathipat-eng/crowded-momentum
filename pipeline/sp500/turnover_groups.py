from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_turnover(decile10_df: pd.DataFrame, price_df: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
    merged = decile10_df.merge(price_df[["date", "ticker", "volume"]], on=["date", "ticker"], how="left")
    merged = merged.merge(shares_df, on="ticker", how="left")
    merged = merged.dropna(subset=["volume", "shares_outstanding"])
    merged["turnover"] = merged["volume"] / merged["shares_outstanding"]
    return merged[["date", "ticker", "momentum", "turnover"]]


def split_by_median_turnover(turnover_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for date_val, group_df in turnover_df.groupby("date"):
        group_df = group_df.copy()
        median = group_df["turnover"].median()
        group_df["group"] = group_df["turnover"].apply(lambda t: "high" if t >= median else "low")
        group_df["date"] = date_val
        results.append(group_df)

    result = pd.concat(results, ignore_index=True)
    # Reorder columns to match expected output
    return result[["date", "ticker", "momentum", "turnover", "group"]]


def main() -> None:
    momentum_df = pd.read_parquet(DATA_DIR / "momentum_deciles.parquet")
    price_df = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    shares_df = pd.read_csv(DATA_DIR / "shares_outstanding.csv")
    decile10_df = momentum_df[momentum_df["decile"] == 10]
    turnover_df = compute_turnover(decile10_df, price_df, shares_df)
    grouped_df = split_by_median_turnover(turnover_df)
    grouped_df.to_parquet(DATA_DIR / "decile10_groups.parquet", index=False)
    print(f"Wrote {len(grouped_df)} rows across {grouped_df['date'].nunique()} months")


if __name__ == "__main__":
    main()
