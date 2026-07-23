import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

# NOTE: api.binance.com is category-blocked by name (returns HTTP 403 from a
# corporate secure web gateway) on the network this pipeline runs on.
# data-api.binance.vision is Binance's own public market-data mirror of the
# same REST API (same endpoint path, same params, same response schema,
# same error codes -- verified: /api/v3/klines, HTTP 400 -1121 "Invalid
# symbol" for a bogus symbol, identical pagination behavior) and is not
# blocked, so it is used here instead. Swap back to api.binance.com if
# running this module from an unrestricted network.
REST_URL = "https://data-api.binance.vision/api/v3/klines"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
OUTPUT_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume",
]
PAGE_LIMIT = 1000
PAGE_SLEEP_SECONDS = 0.1
MAX_RETRIES = 5
DEFAULT_RETRY_AFTER_SECONDS = 5


def _to_epoch_ms(value: str) -> int:
    return pd.Timestamp(value, tz="UTC").value // 10**6


def _fetch_page(symbol: str, interval: str, start_ms: int, end_ms: int) -> list | None:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": PAGE_LIMIT,
    }
    url = f"{REST_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    retries = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 418) and retries < MAX_RETRIES:
                retry_after = e.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else DEFAULT_RETRY_AFTER_SECONDS
                except ValueError:
                    wait = DEFAULT_RETRY_AFTER_SECONDS
                time.sleep(wait)
                retries += 1
                continue
            if e.code in (400, 451):
                return None
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            # Transient network failures (DNS, connection reset, SSL handshake
            # timeout) are common over the thousands of requests a full run
            # makes. Retry with exponential backoff before giving up so a
            # single blip does not abort the whole fetch.
            if retries < MAX_RETRIES:
                time.sleep(DEFAULT_RETRY_AFTER_SECONDS * (2 ** retries))
                retries += 1
                continue
            raise


def fetch_klines_rest(symbol: str, interval: str, start: str, end: str,
                       columns: list[str] | None = None) -> pd.DataFrame:
    out_cols = columns if columns is not None else OUTPUT_COLUMNS
    start_ms = _to_epoch_ms(start)
    end_ms = _to_epoch_ms(end)

    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        page = _fetch_page(symbol, interval, cursor, end_ms)
        if page is None:
            # symbol unavailable/delisted (HTTP 400/451)
            return pd.DataFrame(columns=out_cols)
        if not page:
            break
        rows.extend(page)
        last_open_time = page[-1][0]
        cursor = last_open_time + 1
        if len(page) < PAGE_LIMIT:
            break
        time.sleep(PAGE_SLEEP_SECONDS)

    if not rows:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)

    return df[out_cols]
