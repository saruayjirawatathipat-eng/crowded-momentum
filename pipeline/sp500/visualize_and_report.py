from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHARTS_DIR = PROJECT_DIR / "charts"


def compute_group_means(panel_df: pd.DataFrame) -> pd.DataFrame:
    return (
        panel_df.groupby(["date", "group", "horizon"])["forward_return"]
        .agg(mean_return="mean", n_stocks="count")
        .reset_index()
    )


def compute_spread(group_means_df: pd.DataFrame) -> pd.DataFrame:
    pivot = group_means_df.pivot_table(
        index=["date", "horizon"], columns="group", values="mean_return"
    ).reset_index()
    pivot["spread"] = pivot["high"] - pivot["low"]
    return pivot.rename(columns={"high": "high_mean", "low": "low_mean"})


def summarize_spread(spread_df: pd.DataFrame) -> pd.DataFrame:
    return (
        spread_df.groupby("horizon")["spread"]
        .agg(mean_spread="mean", std_spread="std", n_months="count")
        .reset_index()
    )


def compute_cumulative_returns(group_means_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    subset = group_means_df[group_means_df["horizon"] == horizon].sort_values("date").copy()
    subset["cumulative_return"] = subset.groupby("group")["mean_return"].transform(
        lambda s: (1 + s).cumprod() - 1
    )
    return subset[["date", "group", "cumulative_return"]]


def plot_mean_return_by_group(group_means_df: pd.DataFrame):
    overall = group_means_df.groupby(["group", "horizon"])["mean_return"].mean().reset_index()
    pivot = overall.pivot(index="horizon", columns="group", values="mean_return")
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Forward horizon (months)")
    ax.set_ylabel("Mean forward return")
    ax.set_title("Mean Forward Return by Turnover Group")
    return fig


def plot_cumulative_return(cum_df: pd.DataFrame):
    fig, ax = plt.subplots()
    for group, sub in cum_df.groupby("group"):
        ax.plot(sub["date"], sub["cumulative_return"], label=group)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Cumulative 1-Month Return: High vs Low Turnover")
    ax.legend()
    return fig


def plot_return_distribution(panel_df: pd.DataFrame, horizon: int = 1):
    subset = panel_df[panel_df["horizon"] == horizon]
    fig, ax = plt.subplots()
    data = [subset[subset["group"] == g]["forward_return"] for g in ["high", "low"]]
    ax.boxplot(data)
    ax.set_xticklabels(["high", "low"])
    ax.set_ylabel(f"{horizon}-month forward return")
    ax.set_title("Return Distribution by Turnover Group")
    return fig


def plot_momentum_vs_turnover(decile10_df: pd.DataFrame):
    fig, ax = plt.subplots()
    for group, sub in decile10_df.groupby("group"):
        ax.scatter(sub["momentum"], sub["turnover"], label=group, alpha=0.5, s=10)
    ax.set_xlabel("Momentum")
    ax.set_ylabel("Turnover")
    ax.set_title("Momentum vs Turnover in Decile 10")
    ax.legend()
    return fig


def build_conclusion_markdown(spread_summary_df: pd.DataFrame) -> str:
    lines = [
        "# Crowded Momentum Trades — Conclusion",
        "",
        "## Hypothesis",
        "High-momentum stocks with unusually high turnover may be \"crowded trades\" "
        "that keep performing well short-term but underperform or reverse later.",
        "",
        "## Method",
        "S&P 500 stocks, 12-1 month momentum, monthly deciles, Decile 10 split into "
        "high/low turnover by monthly median, compared on 1/3/6-month forward returns.",
        "",
        "## Results",
        "",
        "| Horizon (months) | Mean spread (high - low) | Std dev | Months observed |",
        "|---|---|---|---|",
    ]
    for _, row in spread_summary_df.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['mean_spread']:.4f} | "
            f"{row['std_spread']:.4f} | {int(row['n_months'])} |"
        )
    lines += [
        "",
        "![Mean return by group](charts/mean_return_by_group.png)",
        "![Cumulative return](charts/cumulative_return.png)",
        "![Return distribution](charts/return_distribution_boxplot.png)",
        "![Momentum vs turnover](charts/momentum_vs_turnover_scatter.png)",
        "",
        "## Conclusion",
        "The hypothesis was not supported: the data shows the opposite pattern, with "
        "high-turnover momentum stocks outperforming low-turnover momentum stocks at "
        "all three horizons (1, 3, and 6 months), and the spread widening at longer "
        "horizons (mean spread 0.0153 at 1 month, 0.0580 at 3 months, 0.1264 at 6 "
        "months). This is counter to the \"crowded trade reversal\" hypothesis, which "
        "predicted high-turnover stocks would underperform due to crowding effects.",
        "",
        "## Limitations",
        "- Survivorship bias: uses today's S&P 500 constituents applied across the "
        "full lookback window, not true point-in-time membership.",
        "- Shares outstanding held constant at a single current snapshot rather than "
        "a monthly historical series.",
        "- Some months may have thin samples per turnover group since the median "
        "split is recomputed monthly within Decile 10.",
        "- The analysis window is short (5 years) and spans a single, largely bullish "
        "market regime — results may not generalize to other market conditions (e.g. "
        "a sustained bear market or higher-volatility regime).",
        "- The 3- and 6-month forward returns use overlapping monthly windows, which "
        "induces autocorrelation in the reported spread series — the reported "
        "standard deviations likely understate the true sampling uncertainty of the "
        "mean spread.",
        "- No formal significance test (e.g. a t-statistic) was computed on the mean "
        "spread — the positive result is descriptive, not a statistically validated "
        "finding.",
    ]
    return "\n".join(lines)


def main() -> None:
    panel_df = pd.read_parquet(DATA_DIR / "forward_returns_summary.parquet")
    decile10_df = pd.read_parquet(DATA_DIR / "decile10_groups.parquet")

    group_means_df = compute_group_means(panel_df)
    spread_df = compute_spread(group_means_df)
    spread_summary_df = summarize_spread(spread_df)
    cum_df = compute_cumulative_returns(group_means_df, horizon=1)

    CHARTS_DIR.mkdir(exist_ok=True)
    plot_mean_return_by_group(group_means_df).savefig(CHARTS_DIR / "mean_return_by_group.png")
    plot_cumulative_return(cum_df).savefig(CHARTS_DIR / "cumulative_return.png")
    plot_return_distribution(panel_df).savefig(CHARTS_DIR / "return_distribution_boxplot.png")
    plot_momentum_vs_turnover(decile10_df).savefig(CHARTS_DIR / "momentum_vs_turnover_scatter.png")

    conclusion = build_conclusion_markdown(spread_summary_df)
    (PROJECT_DIR / "conclusion.md").write_text(conclusion)
    print("Wrote conclusion.md and 4 chart PNGs")


if __name__ == "__main__":
    main()
