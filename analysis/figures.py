"""Regenerate the paper's result tables and figure from committed summary data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TF_ORDER = ["monthly", "weekly", "daily", "4h", "1h"]


def _spread_series(df: pd.DataFrame, pos: str, neg: str) -> pd.DataFrame:
    """Per-(date, horizon) mean forward return of `pos` group minus `neg` group."""
    means = df.groupby(["date", "horizon", "group"])["forward_return"].mean().unstack("group")
    means = means.dropna(subset=[pos, neg])
    means["spread"] = means[pos] - means[neg]
    return means.reset_index()


def phase1_turnover_table() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "sp500_turnover_forward_returns.parquet")
    s = _spread_series(df, pos="high", neg="low")
    out = s.groupby("horizon")["spread"].mean().reset_index()
    return out.rename(columns={"spread": "mean_spread"})


def phase2_obv_table() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "sp500_obv_forward_returns.parquet")
    s = _spread_series(df, pos="confirmed", neg="divergent")
    rows = []
    for h, g in s.groupby("horizon"):
        n = len(g)
        mean = g["spread"].mean()
        std = g["spread"].std(ddof=1)
        t = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        rows.append({"horizon": h, "mean_spread": mean, "n_months": n, "t_stat": t})
    return pd.DataFrame(rows)


def phase3_sweep_table() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "crypto_sweep_table.parquet")
    df["timeframe"] = pd.Categorical(df["timeframe"], categories=TF_ORDER, ordered=True)
    return df.sort_values(["timeframe", "horizon"]).reset_index(drop=True)


def significance_figure():
    df = phase3_sweep_table()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(TF_ORDER))
    markers = {1: "o", 2: "s", 4: "^"}
    for h in (1, 2, 4):
        sub = df[df["horizon"] == h].set_index("timeframe").reindex(TF_ORDER)
        ax.plot(x, sub["t_stat"].values, marker=markers[h], label=f"horizon {h} bar(s)")
    ax.axhspan(-1.96, 1.96, color="0.85", zorder=0)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(TF_ORDER)
    ax.set_xlabel("timeframe (slow → fast)")
    ax.set_ylabel("t-statistic")
    ax.set_title("Significance of confirmed − divergent spread by timeframe")
    ax.legend()
    fig.tight_layout()
    return fig
