import numpy as np
import pandas as pd
import pytest
from sweep import (compute_group_means, compute_spread,
                   summarize_spread_with_tstat, build_sweep_table,
                   winsorize_returns)

def _panel():
    # 3 bars, confirmed beats divergent by a constant 0.02 at horizon 1
    rows = []
    for i, t in enumerate(pd.date_range("2024-01-01", periods=3, freq="1D")):
        rows += [
            {"open_time": t, "symbol": "A", "group": "confirmed", "horizon": 1, "forward_return": 0.05},
            {"open_time": t, "symbol": "B", "group": "divergent", "horizon": 1, "forward_return": 0.03},
        ]
    return pd.DataFrame(rows)

def test_spread_and_tstat():
    gm = compute_group_means(_panel())
    spread = compute_spread(gm)
    summ = summarize_spread_with_tstat(spread)
    row = summ[summ["horizon"] == 1].iloc[0]
    assert row["mean_spread"] == pytest.approx(0.02)
    assert row["n_bars"] == 3
    # zero variance spread -> std 0 -> t_stat inf or nan; just assert mean holds
    assert row["std_spread"] == pytest.approx(0.0)

def test_build_sweep_table_orders_ladder():
    summ = summarize_spread_with_tstat(compute_spread(compute_group_means(_panel())))
    table = build_sweep_table({"1h": summ, "daily": summ}, {"1h": 0.1, "daily": 0.2})
    assert list(table["timeframe"].unique()) == ["daily", "1h"]  # ladder order
    assert set(table.columns) >= {"timeframe", "horizon", "mean_spread", "t_stat", "divergence_rate"}

def test_winsorize_caps_extreme_return_per_horizon():
    # one LUNA-style blowup among many sane rows must be clipped to the
    # horizon's 99th percentile, not left to dominate the spread
    rows = [{"open_time": pd.Timestamp("2024-01-01"), "symbol": "X",
             "horizon": 4, "forward_return": 51999.0}]
    rows += [{"open_time": pd.Timestamp("2024-01-01"), "symbol": f"S{i}",
              "horizon": 4, "forward_return": 0.01 * i} for i in range(100)]
    out = winsorize_returns(pd.DataFrame(rows))
    assert out["forward_return"].max() < 2.0          # 51999 clipped away
    assert out["forward_return"].max() == pytest.approx(0.99, abs=0.05)  # ~99th pct
    # low tail is clipped to ~1st percentile (near 0 here), not left negative
    assert 0.0 <= out["forward_return"].min() <= 0.05

def test_compute_spread_handles_missing_group():
    # a bar where only the confirmed group is present must not crash;
    # it yields no spread row (needs both groups) rather than KeyError
    gm = pd.DataFrame({
        "open_time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "group": ["confirmed", "confirmed"],
        "horizon": [1, 1],
        "mean_return": [0.05, 0.03],
    })
    spread = compute_spread(gm)
    assert len(spread) == 0  # no bar had both groups, so no spread rows
