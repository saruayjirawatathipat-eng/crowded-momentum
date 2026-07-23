from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_net_flow(price_df: pd.DataFrame, window: int = 11) -> pd.DataFrame:
    price_wide = price_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    volume_wide = price_df.pivot(index="date", columns="ticker", values="volume").sort_index()
    # Signed monthly volume: +volume on up-months, -volume on down-months,
    # 0 on unchanged months. Rolling sum of this over the formation window
    # equals OBV_{t-1} - OBV_{t-12}.
    signed_volume = np.sign(price_wide.diff()) * volume_wide
    delta_obv = signed_volume.shift(1).rolling(window).sum()
    total_volume = volume_wide.shift(1).rolling(window).sum()
    flow = delta_obv / total_volume
    long = flow.stack().reset_index()
    long.columns = ["date", "ticker", "flow"]
    return long.dropna(subset=["flow"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def split_by_median_flow(flow_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for date_val, group_df in flow_df.groupby("date"):
        group_df = group_df.copy()
        median = group_df["flow"].median()
        group_df["group"] = group_df["flow"].apply(
            lambda f: "confirmed" if f >= median else "divergent"
        )
        group_df["date"] = date_val
        results.append(group_df)

    result = pd.concat(results, ignore_index=True)
    return result[["date", "ticker", "momentum", "flow", "group"]]


def negative_flow_diagnostic(flow_df: pd.DataFrame) -> pd.DataFrame:
    return (
        flow_df.groupby("date")["flow"]
        .agg(n_stocks="count", n_negative=lambda s: int((s < 0).sum()))
        .reset_index()
    )


def main() -> None:
    momentum_df = pd.read_parquet(DATA_DIR / "momentum_deciles.parquet")
    price_df = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    decile10_df = momentum_df[momentum_df["decile"] == 10]
    flow_df = compute_net_flow(price_df)
    merged = decile10_df.merge(flow_df, on=["date", "ticker"], how="inner")
    grouped_df = split_by_median_flow(merged[["date", "ticker", "momentum", "flow"]])
    grouped_df.to_parquet(DATA_DIR / "decile10_obv_groups.parquet", index=False)

    diagnostic_df = negative_flow_diagnostic(grouped_df)
    diagnostic_df.to_csv(DATA_DIR / "obv_negative_flow_diagnostic.csv", index=False)
    share_negative = diagnostic_df["n_negative"].sum() / diagnostic_df["n_stocks"].sum()
    print(f"Wrote {len(grouped_df)} rows across {grouped_df['date'].nunique()} months")
    print(f"Strictly negative flow (textbook divergence): {share_negative:.1%} of Decile 10 stock-months")


if __name__ == "__main__":
    main()
