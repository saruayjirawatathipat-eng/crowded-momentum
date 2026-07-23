import os
from pathlib import Path

import pandas as pd

from universe import build_membership
from binance_rest import fetch_klines_rest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KLINES_1H = DATA_DIR / "klines_1h"
COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
START = os.environ.get("CMC_START", "2019-01")
# Persistence cap: over a multi-year window, hundreds of coins pass briefly
# through the daily top-N (alt-season flashes) but contribute little to the
# cross-section while each costs a full 1h-history fetch. Restrict the universe
# to the MAX_MEMBERS coins with the most days-in-universe. This keeps coins
# that were meaningfully liquid for a sustained period -- including ones that
# later died (they still rank high on day-count) -- and drops the flash tail.
MAX_MEMBERS = int(os.environ.get("CMC_MAX_MEMBERS", "100"))


def main() -> None:
    daily_volume = pd.read_parquet(DATA_DIR / "daily_volume.parquet")
    full = build_membership(daily_volume, n=75, lookback_days=30)
    eligible = full["symbol"].value_counts().head(MAX_MEMBERS).index
    eligible_daily = daily_volume[daily_volume["symbol"].isin(eligible)]
    membership = build_membership(eligible_daily, n=75, lookback_days=30)
    membership.to_parquet(DATA_DIR / "universe.parquet", index=False)
    print(f"Universe: {membership['symbol'].nunique()} persistent members "
          f"(capped at {MAX_MEMBERS}), {len(membership)} member-days across "
          f"{membership['date'].nunique()} days")

    ever_members = sorted(membership["symbol"].unique())
    KLINES_1H.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.utcnow().strftime("%Y-%m")
    fetched = 0
    for sym in ever_members:
        out_path = KLINES_1H / f"{sym}.parquet"
        # Resume support: a full 1h fetch is thousands of requests and can be
        # interrupted by a transient network error. Skip symbols already on
        # disk so a re-run picks up where it left off instead of restarting.
        if out_path.exists():
            fetched += 1
            continue
        df = fetch_klines_rest(sym, "1h", START, end, columns=COLS)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            fetched += 1
    print(f"Fetched 1h klines for {fetched}/{len(ever_members)} ever-member symbols "
          f"({len(list(KLINES_1H.glob('*.parquet')))} parquet files on disk)")


if __name__ == "__main__":
    main()
