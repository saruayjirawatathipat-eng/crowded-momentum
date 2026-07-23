from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def top_quintile(momentum_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for t, g in momentum_df.groupby("open_time"):
        if len(g) < 5:
            continue
        g = g.copy()
        keep = g["momentum"].rank(pct=True) > 0.8
        frames.append(g[keep])
    if not frames:
        return momentum_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)

def split_by_median_flow(merged_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for t, g in merged_df.groupby("open_time"):
        g = g.copy()
        median = g["flow"].median()
        g["group"] = g["flow"].apply(lambda f: "confirmed" if f >= median else "divergent")
        frames.append(g)
    result = pd.concat(frames, ignore_index=True)
    return result[["open_time", "symbol", "momentum", "flow", "group"]]

def divergence_rate(grouped_df: pd.DataFrame) -> float:
    if len(grouped_df) == 0:
        return 0.0
    return float((grouped_df["flow"] < 0).mean())
