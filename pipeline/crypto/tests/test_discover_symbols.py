import pytest

from discover_symbols import list_usdt_symbols


def test_lists_usdt_symbols_including_btc():
    try:
        symbols = list_usdt_symbols()
    except Exception as exc:
        pytest.skip(f"Binance symbol listing unreachable; network-dependent test skipped ({exc!r})")
    if not symbols:
        pytest.skip("Binance symbol listing returned no data; network-dependent test skipped")

    assert "BTCUSDT" in symbols
    assert all(s.endswith("USDT") for s in symbols)
    assert len(symbols) > 50
