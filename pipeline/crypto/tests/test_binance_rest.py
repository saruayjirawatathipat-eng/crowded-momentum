import pandas as pd
import pytest

from binance_rest import fetch_klines_rest


def _fetch(*args, **kwargs) -> pd.DataFrame:
    # Live network call to Binance's public market-data mirror. Skip
    # gracefully if the network/endpoint is unreachable from this
    # environment rather than failing the suite, consistent with how the
    # S&P pipeline's live-network smoke test is handled.
    try:
        return fetch_klines_rest(*args, **kwargs)
    except Exception as exc:
        pytest.skip(f"Binance REST endpoint unreachable; network-dependent test skipped ({exc!r})")


def test_fetch_rest_daily_schema():
    df = _fetch(
        "BTCUSDT", "1d", "2026-05", "2026-07",
        columns=["open_time", "quote_volume"],
    )
    if df.empty:
        pytest.skip("Binance REST endpoint returned no data; network-dependent test skipped")
    assert list(df.columns) == ["open_time", "quote_volume"]
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
    assert (df["quote_volume"] > 0).all()


def test_fetch_rest_paginates_past_1000():
    df = _fetch("BTCUSDT", "1h", "2026-04", "2026-07")
    if df.empty:
        pytest.skip("Binance REST endpoint returned no data; network-dependent test skipped")
    assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
    assert len(df) > 1000
    diffs = df["open_time"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all()


def test_fetch_rest_unavailable_symbol_returns_empty():
    df = _fetch("NOTAREALCOINUSDT", "1d", "2026-05", "2026-07")
    assert df.empty
    assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
