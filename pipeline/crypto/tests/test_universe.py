import pandas as pd
from universe import build_membership, apply_membership

def _daily_volume():
    # 40 days, 3 symbols. AAA always highest, BBB middle, CCC lowest.
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    for d in dates:
        rows += [
            {"date": d, "symbol": "AAA", "quote_volume": 300.0},
            {"date": d, "symbol": "BBB", "quote_volume": 200.0},
            {"date": d, "symbol": "CCC", "quote_volume": 100.0},
        ]
    return pd.DataFrame(rows)

def test_build_membership_takes_top_n_after_full_lookback():
    m = build_membership(_daily_volume(), n=2, lookback_days=30)
    # first 29 days lack a full 30-day window -> excluded
    assert m["date"].min() == pd.Timestamp("2024-01-30")
    day = m[m["date"] == pd.Timestamp("2024-02-05")]
    assert set(day["symbol"]) == {"AAA", "BBB"}  # CCC (lowest) excluded

def test_apply_membership_filters_panel():
    membership = pd.DataFrame({
        "date": [pd.Timestamp("2024-02-05")],
        "symbol": ["AAA"],
    })
    panel = pd.DataFrame({
        "open_time": [pd.Timestamp("2024-02-05 03:00"), pd.Timestamp("2024-02-05 03:00")],
        "symbol": ["AAA", "BBB"],
        "close": [1.0, 1.0],
    })
    out = apply_membership(panel, membership)
    assert list(out["symbol"]) == ["AAA"]
