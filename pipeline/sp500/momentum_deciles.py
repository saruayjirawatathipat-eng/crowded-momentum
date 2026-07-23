from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_momentum(price_df: pd.DataFrame) -> pd.DataFrame:
    wide = price_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    momentum = wide.shift(1) / wide.shift(12) - 1
    long = momentum.stack().reset_index()
    long.columns = ["date", "ticker", "momentum"]
    return long.dropna(subset=["momentum"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def assign_deciles(momentum_df: pd.DataFrame) -> pd.DataFrame:
    def _rank_date(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 10:
            return group.iloc[0:0]
        group = group.copy()
        # duplicates="drop" can yield fewer than 10 bins if momentum is heavily
        # tied within a month; acceptable for this pass, documented as a limitation.
        group["decile"] = pd.qcut(group["momentum"], 10, labels=False, duplicates="drop") + 1
        return group

    # Preserve date column which is lost in groupby().apply()
    result_list = []
    for date_val, group_df in momentum_df.groupby("date"):
        ranked = _rank_date(group_df)
        if len(ranked) > 0:
            ranked["date"] = date_val
            result_list.append(ranked)

    if result_list:
        result = pd.concat(result_list, ignore_index=True)
        # Reorder columns to match interface: date, ticker, momentum, decile
        result = result[["date", "ticker", "momentum", "decile"]]
    else:
        result = pd.DataFrame(columns=["date", "ticker", "momentum", "decile"])

    return result.reset_index(drop=True)


def main() -> None:
    price_df = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    momentum_df = compute_momentum(price_df)
    decile_df = assign_deciles(momentum_df)
    decile_df.to_parquet(DATA_DIR / "momentum_deciles.parquet", index=False)
    print(f"Wrote {len(decile_df)} rows across {decile_df['date'].nunique()} months")


if __name__ == "__main__":
    main()
