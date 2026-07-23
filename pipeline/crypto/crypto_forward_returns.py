from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_forward_returns(panel_df: pd.DataFrame, horizons=(1, 2, 4)) -> pd.DataFrame:
    wide = panel_df.pivot(index="open_time", columns="symbol", values="close").sort_index()
    frames = []
    for h in horizons:
        forward = wide.shift(-h) / wide - 1
        long = forward.stack().reset_index()
        long.columns = ["open_time", "symbol", "forward_return"]
        long["horizon"] = h
        frames.append(long.dropna(subset=["forward_return"]))
    return pd.concat(frames, ignore_index=True)


def join_forward_returns_with_groups(groups_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> pd.DataFrame:
    merged = groups_df[["open_time", "symbol", "group"]].merge(
        forward_returns_df, on=["open_time", "symbol"], how="inner"
    )
    return merged[["open_time", "symbol", "group", "horizon", "forward_return"]]
