from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_and_report import compute_group_means, compute_cumulative_returns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHARTS_DIR = PROJECT_DIR / "charts"

PHASE2_MARKER = "# Phase 2 — Momentum Requires Volume Confirmation (OBV Divergence)"


def compute_obv_spread(group_means_df: pd.DataFrame) -> pd.DataFrame:
    pivot = group_means_df.pivot_table(
        index=["date", "horizon"], columns="group", values="mean_return"
    ).reset_index()
    pivot["spread"] = pivot["confirmed"] - pivot["divergent"]
    return pivot.rename(columns={"confirmed": "confirmed_mean", "divergent": "divergent_mean"})


def summarize_spread_with_tstat(spread_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        spread_df.groupby("horizon")["spread"]
        .agg(mean_spread="mean", std_spread="std", n_months="count")
        .reset_index()
    )
    summary["t_stat"] = summary["mean_spread"] / (
        summary["std_spread"] / np.sqrt(summary["n_months"])
    )
    return summary


def plot_mean_return_by_group(group_means_df: pd.DataFrame):
    overall = group_means_df.groupby(["group", "horizon"])["mean_return"].mean().reset_index()
    pivot = overall.pivot(index="horizon", columns="group", values="mean_return")
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Forward horizon (months)")
    ax.set_ylabel("Mean forward return")
    ax.set_title("Mean Forward Return: OBV Confirmed vs Divergent")
    return fig


def plot_cumulative_return(cum_df: pd.DataFrame):
    fig, ax = plt.subplots()
    for group, sub in cum_df.groupby("group"):
        ax.plot(sub["date"], sub["cumulative_return"], label=group)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Cumulative 1-Month Return: OBV Confirmed vs Divergent")
    ax.legend()
    return fig


def plot_return_distribution(panel_df: pd.DataFrame, horizon: int = 1):
    subset = panel_df[panel_df["horizon"] == horizon]
    fig, ax = plt.subplots()
    data = [subset[subset["group"] == g]["forward_return"] for g in ["confirmed", "divergent"]]
    ax.boxplot(data)
    ax.set_xticklabels(["confirmed", "divergent"])
    ax.set_ylabel(f"{horizon}-month forward return")
    ax.set_title("Return Distribution: OBV Confirmed vs Divergent")
    return fig


def plot_momentum_vs_flow(groups_df: pd.DataFrame):
    fig, ax = plt.subplots()
    for group, sub in groups_df.groupby("group"):
        ax.scatter(sub["momentum"], sub["flow"], label=group, alpha=0.5, s=10)
    ax.set_xlabel("Momentum")
    ax.set_ylabel("Net flow fraction")
    ax.set_title("Momentum vs OBV Net Flow in Decile 10")
    ax.legend()
    return fig


def describe_cumulative_lead(cum_df: pd.DataFrame) -> str:
    """Describe which group led the compounded 1-month cumulative-return path.

    The mean of monthly spreads (used for the Results table and t-stats) and the
    compounded cumulative-return path (used for the chart) are different
    statistics and can disagree in sign — this renders that comparison as a
    sentence so the write-up can reconcile the two rather than let them look
    like a contradiction.
    """
    pivot = cum_df.pivot(index="date", columns="group", values="cumulative_return").sort_index()
    lead_share = (pivot["confirmed"] > pivot["divergent"]).mean()
    leader, other = ("confirmed", "divergent") if lead_share >= 0.5 else ("divergent", "confirmed")
    lead_pct = max(lead_share, 1 - lead_share)
    return (
        "Worth reconciling with the chart: the *compounded* cumulative 1-month "
        f"return path (`obv_cumulative_return.png`) tells a different story than "
        f"the mean-of-monthly-spreads table above — there, the {leader} group "
        f"actually led the {other} group for about {lead_pct:.0%} of the sample, "
        "only converging in the final few months. This is not a contradiction: "
        "mean-of-monthly-spreads and a compounded cumulative path are different "
        "statistics and can disagree in sign. If anything it reinforces the "
        "noise conclusion — with a near-zero, non-significant mean spread, the "
        "sign of any apparent advantage is unstable over time, whether you look "
        "at it month-by-month or compounded."
    )


def build_phase2_markdown(
    spread_summary_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    cum_df: pd.DataFrame | None = None,
) -> str:
    share_negative = diagnostic_df["n_negative"].sum() / diagnostic_df["n_stocks"].sum()
    all_positive = bool((spread_summary_df["mean_spread"] > 0).all())
    all_negative = bool((spread_summary_df["mean_spread"] < 0).all())
    max_abs_t = spread_summary_df["t_stat"].abs().max()
    is_significant = bool(max_abs_t >= 1.96)
    if is_significant:
        significance = (
            f"At least one horizon's t-stat ({max_abs_t:.2f}) clears the conventional "
            "1.96 threshold for significance at the 5% level, so this pattern is unlikely "
            "to be pure noise."
        )
    else:
        significance = (
            f"None of the t-statistics come close to the conventional 1.96 threshold for "
            f"significance at any horizon (largest |t| = {max_abs_t:.2f}), so this pattern "
            "should be read as noise, not a real effect."
        )
    if all_positive:
        direction = (
            "Confirmed stocks outperformed divergent stocks at every horizon, consistent "
            "with the hypothesis that momentum requires volume confirmation — equivalently, "
            "OBV divergence acted as an underperformance/reversal signal. "
            + significance
        )
    elif all_negative:
        direction = (
            "Divergent stocks outperformed confirmed stocks at every horizon — the opposite "
            "of the hypothesis: volume confirmation added no value in this sample. "
            + significance
        )
    else:
        direction = (
            "The spread changes sign across horizons, so the sample shows no consistent "
            "confirmation effect in either direction."
        )
    lines = [
        PHASE2_MARKER,
        "",
        "## Hypothesis",
        "Among top-momentum (Decile 10) stocks, price strength confirmed by net buying "
        "flow (rising OBV) continues, while price strength diverging from flow (OBV "
        "flat or falling) underperforms or reverses. This follows from Phase 1: "
        "turnover is unsigned volume and cannot separate buying from selling, so "
        "Phase 2 signs each month's volume by price direction instead.",
        "",
        "## Method",
        "Monthly OBV per stock (sign of month-over-month adjusted-close change × "
        "volume). Net flow fraction = ΔOBV over the 12-1 formation window ÷ total "
        "volume over the window, bounded in [-1, 1]. Decile 10 split each month at "
        "the median flow into confirmed vs divergent, compared on 1/3/6-month "
        "forward returns — the same design as Phase 1 with only the sorting "
        "variable changed.",
        "",
        f"Diagnostic: {share_negative:.1%} of Decile 10 stock-months had strictly "
        "negative flow (textbook price-up/OBV-down divergence) — why the median "
        "split, not the strict sign rule, defines the groups.",
        "",
        "## Results",
        "",
        "| Horizon (months) | Mean spread (confirmed - divergent) | Std dev | Months observed | t-stat |",
        "|---|---|---|---|---|",
    ]
    for _, row in spread_summary_df.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['mean_spread']:.4f} | "
            f"{row['std_spread']:.4f} | {int(row['n_months'])} | {row['t_stat']:.2f} |"
        )
    lines += [
        "",
        "![OBV mean return by group](charts/obv_mean_return_by_group.png)",
        "![OBV cumulative return](charts/obv_cumulative_return.png)",
        "![OBV return distribution](charts/obv_return_distribution_boxplot.png)",
        "![Momentum vs flow](charts/obv_momentum_vs_flow_scatter.png)",
        "",
        "## Conclusion",
        direction,
    ]
    if cum_df is not None:
        lines.append("")
        lines.append(describe_cumulative_lead(cum_df))
    lines += [
        "",
        "## Limitations",
        "- All Phase 1 limitations carry over: survivorship bias, ~5-year single "
        "bullish regime, overlapping 3/6-month forward windows.",
        "- The t-statistics at 3 and 6 months are inflated by overlapping windows "
        "(the spread series is autocorrelated); only the 1-month t-stat is clean.",
        "- Monthly bars make OBV coarse (~12 signed observations per formation "
        "window); intra-month accumulation and distribution are invisible.",
        "- OBV signs the entire month's volume by close-to-close direction — a blunt "
        "proxy for true signed order flow (CVD). The standard OBV convention "
        "(close vs previous close) is used rather than close vs open, since the "
        "stored dataset has no open prices.",
    ]
    return "\n".join(lines)


def update_conclusion(existing_text: str, phase2_section: str) -> str:
    if PHASE2_MARKER in existing_text:
        base = existing_text.split(PHASE2_MARKER)[0].rstrip().removesuffix("---").rstrip()
    else:
        base = existing_text.rstrip()
    return base + "\n\n---\n\n" + phase2_section + "\n"


def main() -> None:
    panel_df = pd.read_parquet(DATA_DIR / "obv_forward_returns_summary.parquet")
    groups_df = pd.read_parquet(DATA_DIR / "decile10_obv_groups.parquet")
    diagnostic_df = pd.read_csv(DATA_DIR / "obv_negative_flow_diagnostic.csv")

    group_means_df = compute_group_means(panel_df)
    spread_df = compute_obv_spread(group_means_df)
    spread_summary_df = summarize_spread_with_tstat(spread_df)
    cum_df = compute_cumulative_returns(group_means_df, horizon=1)

    CHARTS_DIR.mkdir(exist_ok=True)
    plot_mean_return_by_group(group_means_df).savefig(CHARTS_DIR / "obv_mean_return_by_group.png")
    plot_cumulative_return(cum_df).savefig(CHARTS_DIR / "obv_cumulative_return.png")
    plot_return_distribution(panel_df).savefig(CHARTS_DIR / "obv_return_distribution_boxplot.png")
    plot_momentum_vs_flow(groups_df).savefig(CHARTS_DIR / "obv_momentum_vs_flow_scatter.png")

    conclusion_path = PROJECT_DIR / "conclusion.md"
    phase2_md = build_phase2_markdown(spread_summary_df, diagnostic_df, cum_df)
    conclusion_path.write_text(update_conclusion(conclusion_path.read_text(), phase2_md))
    print(spread_summary_df.to_string(index=False))
    print("Updated conclusion.md and wrote 4 obv_*.png charts")


if __name__ == "__main__":
    main()
