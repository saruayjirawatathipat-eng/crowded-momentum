import pandas as pd

from obv_groups import compute_net_flow, split_by_median_flow, negative_flow_diagnostic


def _price_df(prices: list[float], volumes: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="MS")
    return pd.DataFrame({
        "date": list(dates),
        "ticker": ["AAA"] * len(prices),
        "adj_close": prices,
        "volume": volumes,
    })


def test_net_flow_is_one_when_every_month_closes_up():
    # 13 months rising every month, equal volume: all volume is buy-side,
    # so flow = 1.0. Only month 13 has a full 11-month formation window.
    df = _price_df([100.0 + i for i in range(13)], [100] * 13)
    result = compute_net_flow(df)
    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2021-01-01")
    assert abs(result["flow"].iloc[0] - 1.0) < 1e-9


def test_net_flow_counts_down_month_volume_as_negative():
    # Month 7 closes down; window for month 13 covers months 2-12
    # -> 10 up-months and 1 down-month of equal volume: (10-1)/11.
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0,
              105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    df = _price_df(prices, [100] * 13)
    result = compute_net_flow(df)
    assert len(result) == 1
    assert abs(result["flow"].iloc[0] - 9.0 / 11.0) < 1e-9


def test_net_flow_requires_full_window():
    # Only 12 months of data -> no month has 11 prior signed-volume months
    # (the first month has no price change to sign).
    df = _price_df([100.0 + i for i in range(12)], [100] * 12)
    result = compute_net_flow(df)
    assert len(result) == 0


def test_split_by_median_flow_balances_groups():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "momentum": [0.1, 0.2, 0.3, 0.4],
        "flow": [-0.2, 0.1, 0.4, 0.8],
    })
    result = split_by_median_flow(df)
    counts = result["group"].value_counts()
    assert counts["confirmed"] == 2
    assert counts["divergent"] == 2
    assert result[result["ticker"] == "D"]["group"].iloc[0] == "confirmed"
    assert result[result["ticker"] == "A"]["group"].iloc[0] == "divergent"


def test_split_by_median_flow_splits_within_each_month():
    # A month where every flow is high must still split at its own median.
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 2 + [pd.Timestamp("2020-02-01")] * 2,
        "ticker": ["A", "B", "A", "B"],
        "momentum": [0.1] * 4,
        "flow": [0.7, 0.9, -0.9, -0.7],
    })
    result = split_by_median_flow(df)
    by_month = result.groupby("date")["group"].value_counts()
    assert by_month[(pd.Timestamp("2020-01-01"), "confirmed")] == 1
    assert by_month[(pd.Timestamp("2020-02-01"), "confirmed")] == 1


def test_negative_flow_diagnostic_counts_strict_divergence():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 3 + [pd.Timestamp("2020-02-01")] * 2,
        "ticker": ["A", "B", "C", "A", "B"],
        "momentum": [0.1] * 5,
        "flow": [-0.2, 0.1, 0.4, 0.3, 0.5],
    })
    result = negative_flow_diagnostic(df)
    jan = result[result["date"] == pd.Timestamp("2020-01-01")].iloc[0]
    feb = result[result["date"] == pd.Timestamp("2020-02-01")].iloc[0]
    assert jan["n_stocks"] == 3 and jan["n_negative"] == 1
    assert feb["n_stocks"] == 2 and feb["n_negative"] == 0
