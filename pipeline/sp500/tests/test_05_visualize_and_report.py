import matplotlib.figure
import pandas as pd

from visualize_and_report import (
    build_conclusion_markdown,
    compute_cumulative_returns,
    compute_group_means,
    compute_spread,
    plot_cumulative_return,
    plot_mean_return_by_group,
    plot_momentum_vs_turnover,
    plot_return_distribution,
    summarize_spread,
)


def test_compute_group_means_known_values():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "group": ["high", "high", "low", "low"],
        "horizon": [1, 1, 1, 1],
        "forward_return": [0.10, 0.20, 0.01, 0.03],
    })
    result = compute_group_means(df)
    high_row = result[result["group"] == "high"]
    assert abs(high_row["mean_return"].iloc[0] - 0.15) < 1e-9
    assert high_row["n_stocks"].iloc[0] == 2


def test_compute_spread_known_values():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 2,
        "group": ["high", "low"],
        "horizon": [1, 1],
        "mean_return": [0.15, 0.02],
        "n_stocks": [2, 2],
    })
    result = compute_spread(df)
    assert abs(result["spread"].iloc[0] - 0.13) < 1e-9


def test_summarize_spread_known_values():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
        "horizon": [1, 1, 1],
        "high_mean": [0.1, 0.2, 0.3],
        "low_mean": [0.05, 0.05, 0.05],
        "spread": [0.05, 0.15, 0.25],
    })
    result = summarize_spread(df)
    assert abs(result["mean_spread"].iloc[0] - 0.15) < 1e-9
    assert result["n_months"].iloc[0] == 3


def test_compute_cumulative_returns_known_values():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2, freq="MS").tolist() * 2,
        "group": ["high", "high", "low", "low"],
        "horizon": [1, 1, 1, 1],
        "mean_return": [0.10, 0.10, 0.05, 0.05],
        "n_stocks": [2, 2, 2, 2],
    })
    result = compute_cumulative_returns(df, horizon=1)
    high = result[result["group"] == "high"].sort_values("date")
    assert abs(high["cumulative_return"].iloc[1] - ((1.10 * 1.10) - 1)) < 1e-9


def test_plot_functions_return_figure():
    group_means_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2, freq="MS").tolist() * 2,
        "group": ["high", "low", "high", "low"],
        "horizon": [1, 1, 1, 1],
        "mean_return": [0.1, 0.05, 0.12, 0.04],
        "n_stocks": [2, 2, 2, 2],
    })
    cum_df = compute_cumulative_returns(group_means_df, horizon=1)
    assert isinstance(plot_mean_return_by_group(group_means_df), matplotlib.figure.Figure)
    assert isinstance(plot_cumulative_return(cum_df), matplotlib.figure.Figure)

    panel_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "group": ["high", "high", "low", "low"],
        "horizon": [1, 1, 1, 1],
        "forward_return": [0.1, 0.2, 0.01, 0.02],
    })
    assert isinstance(plot_return_distribution(panel_df), matplotlib.figure.Figure)

    decile10_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "momentum": [0.1, 0.2, 0.05, 0.15],
        "turnover": [0.01, 0.02, 0.03, 0.04],
        "group": ["high", "high", "low", "low"],
    })
    assert isinstance(plot_momentum_vs_turnover(decile10_df), matplotlib.figure.Figure)


def test_build_conclusion_markdown_contains_key_sections():
    df = pd.DataFrame({
        "horizon": [1, 3, 6],
        "mean_spread": [-0.01, -0.02, -0.03],
        "std_spread": [0.05, 0.08, 0.10],
        "n_months": [48, 46, 43],
    })
    text = build_conclusion_markdown(df)
    assert "# Crowded Momentum Trades" in text
    assert "## Hypothesis" in text
    assert "## Conclusion" in text
    assert "## Limitations" in text
    assert "-0.0100" in text
