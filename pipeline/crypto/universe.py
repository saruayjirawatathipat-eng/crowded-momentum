from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def build_membership(daily_volume_df: pd.DataFrame, n: int = 75,
                     lookback_days: int = 30) -> pd.DataFrame:
    wide = (
        daily_volume_df.pivot(index="date", columns="symbol", values="quote_volume")
        .sort_index()
    )
    trailing = wide.rolling(lookback_days, min_periods=lookback_days).sum()
    members = []
    for date_val, row in trailing.iterrows():
        ranked = row.dropna()
        if ranked.empty:
            continue
        top = ranked.sort_values(ascending=False).head(n).index
        members.extend({"date": date_val, "symbol": s} for s in top)
    return pd.DataFrame(members, columns=["date", "symbol"])

def apply_membership(panel_df: pd.DataFrame, membership_df: pd.DataFrame,
                     time_col: str = "open_time") -> pd.DataFrame:
    panel = panel_df.copy()
    panel["date"] = panel[time_col].dt.floor("D")
    merged = panel.merge(membership_df, on=["date", "symbol"], how="inner")
    return merged.drop(columns="date").reset_index(drop=True)
