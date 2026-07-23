from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LADDER = ["monthly", "weekly", "daily", "4h", "1h"]

def winsorize_returns(panel_df: pd.DataFrame, col: str = "forward_return",
                      lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    # Crypto forward returns computed off a near-zero base price (e.g. LUNA
    # post-collapse at ~$0.0001) can explode to absurd magnitudes (+5,000,000%)
    # that swamp the spread and its variance. Clip each horizon's forward
    # returns to its [lower, upper] quantiles so a single degenerate penny-price
    # observation cannot dominate the result. Standard practice in return-based
    # cross-sectional studies.
    out = panel_df.copy()
    lo = out.groupby("horizon")[col].transform(lambda s: s.quantile(lower))
    hi = out.groupby("horizon")[col].transform(lambda s: s.quantile(upper))
    out[col] = out[col].clip(lower=lo, upper=hi)
    return out

def compute_group_means(panel_df: pd.DataFrame) -> pd.DataFrame:
    return (
        panel_df.groupby(["open_time", "group", "horizon"])["forward_return"]
        .mean().reset_index().rename(columns={"forward_return": "mean_return"})
    )

def compute_spread(group_means_df: pd.DataFrame) -> pd.DataFrame:
    pivot = group_means_df.pivot_table(
        index=["open_time", "horizon"], columns="group", values="mean_return"
    )
    for group in ("confirmed", "divergent"):
        if group not in pivot.columns:
            pivot[group] = np.nan
    pivot["spread"] = pivot["confirmed"] - pivot["divergent"]
    return pivot.dropna(subset=["spread"])

def summarize_spread_with_tstat(spread_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        spread_df.groupby("horizon")["spread"]
        .agg(mean_spread="mean", std_spread="std", n_bars="count").reset_index()
    )
    summary["t_stat"] = summary["mean_spread"] / (
        summary["std_spread"] / np.sqrt(summary["n_bars"])
    )
    return summary

def build_sweep_table(per_tf: dict, div_rates: dict) -> pd.DataFrame:
    frames = []
    for tf in LADDER:
        if tf not in per_tf:
            continue
        s = per_tf[tf].copy()
        s["timeframe"] = tf
        s["divergence_rate"] = div_rates.get(tf, float("nan"))
        frames.append(s)
    table = pd.concat(frames, ignore_index=True)
    return table[["timeframe", "horizon", "mean_spread", "std_spread", "n_bars", "t_stat", "divergence_rate"]]
