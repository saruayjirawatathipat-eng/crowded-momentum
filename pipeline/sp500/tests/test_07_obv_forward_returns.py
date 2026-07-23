import pandas as pd

from forward_returns import compute_forward_returns, join_forward_returns_with_groups
from obv_forward_returns import main  # noqa: F401  # wiring exists and imports cleanly


def test_join_carries_confirmed_divergent_labels_through():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    price_df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["AAA"] * 8 + ["BBB"] * 8,
        "adj_close": [100.0 + i for i in range(8)] + [50.0 + i for i in range(8)],
        "volume": [100] * 16,
    })
    groups_df = pd.DataFrame({
        "date": [dates[0], dates[0]],
        "ticker": ["AAA", "BBB"],
        "momentum": [0.5, 0.4],
        "flow": [0.8, -0.2],
        "group": ["confirmed", "divergent"],
    })
    forward_df = compute_forward_returns(price_df)
    panel = join_forward_returns_with_groups(groups_df, forward_df)
    assert set(panel["group"].unique()) == {"confirmed", "divergent"}
    assert set(panel["horizon"].unique()) == {1, 3, 6}
    aaa_1m = panel[(panel["ticker"] == "AAA") & (panel["horizon"] == 1)]["forward_return"].iloc[0]
    assert abs(aaa_1m - 0.01) < 1e-9  # 101/100 - 1
