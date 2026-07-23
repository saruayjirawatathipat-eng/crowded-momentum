import pandas as pd
import pytest
from crypto_forward_returns import compute_forward_returns, join_forward_returns_with_groups


def _panel():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D")
    return pd.DataFrame({"open_time": idx, "symbol": "A", "close": [10, 11, 12, 13, 14]})


def test_forward_returns_one_bar():
    out = compute_forward_returns(_panel(), horizons=(1,))
    first = out.sort_values("open_time").iloc[0]
    assert first["forward_return"] == pytest.approx(0.1)  # 11/10 - 1


def test_join_keeps_group_label():
    fr = compute_forward_returns(_panel(), horizons=(1,))
    groups = pd.DataFrame({
        "open_time": [pd.Timestamp("2024-01-01")], "symbol": ["A"],
        "momentum": [1.0], "flow": [0.2], "group": ["confirmed"],
    })
    out = join_forward_returns_with_groups(groups, fr)
    assert list(out["group"]) == ["confirmed"]
