import pandas as pd
from resample import resample_ohlcv

def _hourly():
    idx = pd.date_range("2024-01-01 00:00", periods=8, freq="1h")
    return pd.DataFrame({
        "open_time": idx,
        "open":  [1, 2, 3, 4, 5, 6, 7, 8],
        "high":  [2, 3, 4, 5, 6, 7, 8, 9],
        "low":   [0, 1, 2, 3, 4, 5, 6, 7],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "volume": [10] * 8,
        "quote_volume": [100] * 8,
        "taker_buy_volume": [6] * 8,
    })

def test_resample_4h_aggregates_ohlcv():
    out = resample_ohlcv(_hourly(), "4h")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 1 and first["close"] == 4.5
    assert first["high"] == 5 and first["low"] == 0
    assert first["volume"] == 40 and first["quote_volume"] == 400
    assert first["taker_buy_volume"] == 24
    assert first["open_time"] == pd.Timestamp("2024-01-01 00:00")
