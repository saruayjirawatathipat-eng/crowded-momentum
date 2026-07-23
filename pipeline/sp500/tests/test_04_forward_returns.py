import pandas as pd

from forward_returns import compute_forward_returns, join_forward_returns_with_groups


def test_compute_forward_returns_known_values():
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "adj_close": [100.0, 110.0, 121.0], "volume": 1})
    result = compute_forward_returns(df, horizons=(1,))
    row0 = result[result["date"] == dates[0]]
    assert abs(row0["forward_return"].iloc[0] - 0.10) < 1e-9
    row1 = result[result["date"] == dates[1]]
    assert abs(row1["forward_return"].iloc[0] - 0.10) < 1e-9
    assert dates[2] not in result["date"].values


def test_join_forward_returns_with_groups():
    groups_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "ticker": ["AAA"],
        "momentum": [0.1],
        "turnover": [0.02],
        "group": ["high"],
    })
    returns_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        "ticker": ["AAA", "BBB"],
        "horizon": [1, 1],
        "forward_return": [0.05, 0.07],
    })
    result = join_forward_returns_with_groups(groups_df, returns_df)
    assert len(result) == 1
    assert result["group"].iloc[0] == "high"
    assert abs(result["forward_return"].iloc[0] - 0.05) < 1e-9
