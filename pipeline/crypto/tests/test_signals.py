import numpy as np
import pandas as pd
from signals import compute_momentum, compute_net_taker_flow

def _panel(closes, taker, vol):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D")
    return pd.DataFrame({
        "open_time": idx, "symbol": "AAA",
        "close": closes, "volume": vol, "taker_buy_volume": taker,
    })

def test_momentum_is_12_1_return():
    closes = list(range(1, 15))  # 14 bars, strictly rising
    df = compute_momentum(_panel(closes, [1]*14, [2]*14), formation=12, skip=1)
    # first valid at bar index 12 (needs close.shift(12)); momentum = c[11]/c[0] - 1 = 12/1 - 1 = 11.0
    row = df.iloc[0]
    assert row["open_time"] == pd.Timestamp("2024-01-13")
    assert row["momentum"] == pytest_approx(11.0)

def test_net_taker_flow_all_buys_is_one():
    # taker_buy == volume every bar -> net taker (2*tb - vol)=vol -> flow = +1
    n = 14
    df = compute_net_taker_flow(_panel([1]*n, [5]*n, [5]*n), window=11)
    assert np.allclose(df["flow"], 1.0)

def test_net_taker_flow_all_sells_is_minus_one():
    n = 14
    df = compute_net_taker_flow(_panel([1]*n, [0]*n, [5]*n), window=11)
    assert np.allclose(df["flow"], -1.0)

def pytest_approx(x):
    import pytest
    return pytest.approx(x)
