import pandas as pd
import pytest

from fetch_data import (
    drop_insufficient_history,
    fetch_ticker_history,
)


def test_drop_insufficient_history_drops_short_tickers():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
        "ticker": ["AAA", "AAA", "BBB"],
        "adj_close": [1.0, 2.0, 3.0],
        "volume": [100, 200, 300],
    })
    kept, dropped = drop_insufficient_history(df, min_months=2)
    assert dropped == ["BBB"]
    assert set(kept["ticker"]) == {"AAA"}


def test_drop_insufficient_history_keeps_full_tickers():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]),
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "adj_close": [1.0, 2.0, 3.0, 4.0],
        "volume": [100, 200, 300, 400],
    })
    kept, dropped = drop_insufficient_history(df, min_months=2)
    assert dropped == []
    assert len(kept) == 4


def test_fetch_ticker_history_schema_smoke():
    try:
        df = fetch_ticker_history("AAPL", "2024-01-01", "2024-04-01")
    except Exception as exc:
        pytest.skip(f"yfinance unreachable; network-dependent smoke test skipped ({exc!r})")
    if df is None or len(df) == 0:
        pytest.skip("yfinance unreachable; network-dependent smoke test skipped (empty result)")

    assert list(df.columns) == ["date", "ticker", "adj_close", "volume"]
    assert len(df) > 0
    assert df["ticker"].eq("AAPL").all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
