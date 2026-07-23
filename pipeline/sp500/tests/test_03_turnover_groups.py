import pandas as pd

from turnover_groups import compute_turnover, split_by_median_turnover


def test_compute_turnover_known_values():
    decile10_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "ticker": ["AAA"],
        "momentum": [0.1],
        "decile": [10],
    })
    price_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "ticker": ["AAA"],
        "adj_close": [100.0],
        "volume": [1_000_000],
    })
    shares_df = pd.DataFrame({"ticker": ["AAA"], "shares_outstanding": [10_000_000]})
    result = compute_turnover(decile10_df, price_df, shares_df)
    assert abs(result["turnover"].iloc[0] - 0.1) < 1e-9


def test_split_by_median_turnover_balances_groups():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "momentum": [0.1, 0.2, 0.3, 0.4],
        "turnover": [0.01, 0.02, 0.03, 0.04],
    })
    result = split_by_median_turnover(df)
    counts = result["group"].value_counts()
    assert counts["high"] == 2
    assert counts["low"] == 2


def test_split_by_median_turnover_odd_n_puts_median_in_high():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 5,
        "ticker": ["A", "B", "C", "D", "E"],
        "momentum": [0.1] * 5,
        "turnover": [0.01, 0.02, 0.03, 0.04, 0.05],
    })
    result = split_by_median_turnover(df)
    counts = result["group"].value_counts()
    assert counts["high"] == 3
    assert counts["low"] == 2
