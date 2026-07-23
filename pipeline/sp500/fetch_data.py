import io
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_sp500_tickers() -> list[str]:
    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    html = urllib.request.urlopen(req).read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    tickers = tables[0]["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]


def fetch_ticker_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1mo", auto_adjust=True)
    if hist.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist = hist.reset_index().rename(columns={"Date": "date", "Close": "adj_close", "Volume": "volume"})
    hist["date"] = hist["date"].dt.to_period("M").dt.to_timestamp()
    hist["ticker"] = ticker
    return hist[["date", "ticker", "adj_close", "volume"]].drop_duplicates(subset=["date"], keep="last")


def fetch_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    failed = []
    for ticker in tickers:
        try:
            df = fetch_ticker_history(ticker, start, end)
        except Exception:
            failed.append(ticker)
            continue
        if df.empty:
            failed.append(ticker)
            continue
        frames.append(df)
    if failed:
        print(f"Skipped {len(failed)} tickers with no data: {failed}")
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    return pd.concat(frames, ignore_index=True)


def drop_insufficient_history(df: pd.DataFrame, min_months: int) -> tuple[pd.DataFrame, list[str]]:
    counts = df.groupby("ticker")["date"].count()
    keep_tickers = counts[counts >= min_months].index
    dropped = sorted(counts[counts < min_months].index.tolist())
    kept_df = df[df["ticker"].isin(keep_tickers)].reset_index(drop=True)
    return kept_df, dropped


def fetch_shares_outstanding(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            shares = yf.Ticker(ticker).fast_info.get("shares")
        except Exception:
            shares = None
        if shares:
            rows.append({"ticker": ticker, "shares_outstanding": shares})
    return pd.DataFrame(rows, columns=["ticker", "shares_outstanding"])


def main() -> None:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=6)
    tickers = get_sp500_tickers()
    price_df = fetch_price_history(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price_df, dropped = drop_insufficient_history(price_df, min_months=70)
    if dropped:
        print(f"Dropped {len(dropped)} tickers with insufficient history: {dropped}")
    kept_tickers = sorted(price_df["ticker"].unique().tolist())
    shares_df = fetch_shares_outstanding(kept_tickers)
    DATA_DIR.mkdir(exist_ok=True)
    price_df.to_parquet(DATA_DIR / "raw_prices.parquet", index=False)
    shares_df.to_csv(DATA_DIR / "shares_outstanding.csv", index=False)
    print(f"Wrote {len(price_df)} price rows for {price_df['ticker'].nunique()} tickers")


if __name__ == "__main__":
    main()
