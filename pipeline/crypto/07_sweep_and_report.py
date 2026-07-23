from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from groups import divergence_rate  # noqa: E402
from sweep import (  # noqa: E402
    build_sweep_table,
    compute_group_means,
    compute_spread,
    summarize_spread_with_tstat,
    winsorize_returns,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
LADDER = ["monthly", "weekly", "daily", "4h", "1h"]


def _plot_spread_vs_timeframe(sweep_table: pd.DataFrame) -> None:
    order = [tf for tf in LADDER if tf in sweep_table["timeframe"].unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for horizon, g in sweep_table.groupby("horizon"):
        g = g.set_index("timeframe").reindex(order)
        ax1.plot(order, g["mean_spread"], marker="o", label=f"h={horizon}")
        ax2.plot(order, g["t_stat"], marker="o", label=f"h={horizon}")

    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_title("Mean confirmed-divergent spread by timeframe")
    ax1.set_ylabel("mean spread")
    ax1.legend()

    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.axhline(2, color="red", linestyle="--", linewidth=0.8)
    ax2.axhline(-2, color="red", linestyle="--", linewidth=0.8)
    ax2.set_title("t-stat by timeframe")
    ax2.set_ylabel("t-stat")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "spread_vs_timeframe.png", dpi=150)
    plt.close(fig)


def _plot_group_means(tf: str, group_means: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for group, g in group_means.groupby("group"):
        means = g.groupby("horizon")["mean_return"].mean()
        ax.plot(means.index, means.values, marker="o", label=group)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(f"Group mean forward return -- {tf}")
    ax.set_xlabel("horizon (bars)")
    ax.set_ylabel("mean forward return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"group_means_{tf}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    per_tf_summary = {}
    div_rates = {}
    group_means_per_tf = {}

    for tf in LADDER:
        fr_path = DATA_DIR / f"forward_returns_{tf}.parquet"
        groups_path = DATA_DIR / f"groups_{tf}.parquet"
        if not fr_path.exists() or not groups_path.exists():
            continue

        joined = pd.read_parquet(fr_path)
        if joined.empty:
            continue

        # Clip extreme forward returns (e.g. LUNA post-collapse penny-price
        # blowups) before aggregating so one degenerate row cannot dominate.
        joined = winsorize_returns(joined)
        group_means = compute_group_means(joined)
        spread = compute_spread(group_means)
        if spread.empty:
            continue

        per_tf_summary[tf] = summarize_spread_with_tstat(spread)
        group_means_per_tf[tf] = group_means

        grouped = pd.read_parquet(groups_path)
        div_rates[tf] = divergence_rate(grouped)

    if not per_tf_summary:
        print("No timeframe produced a non-empty spread -- nothing to report.")
        return

    sweep_table = build_sweep_table(per_tf_summary, div_rates)
    sweep_table.to_parquet(DATA_DIR / "sweep_table.parquet", index=False)
    print(f"Sweep table: {len(sweep_table)} rows across "
          f"{sweep_table['timeframe'].nunique()} timeframes")

    _plot_spread_vs_timeframe(sweep_table)
    for tf, group_means in group_means_per_tf.items():
        _plot_group_means(tf, group_means)
    print(f"Wrote charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
