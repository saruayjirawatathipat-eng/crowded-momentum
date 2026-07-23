import pandas as pd
from groups import top_quintile, split_by_median_flow, divergence_rate

def test_top_quintile_keeps_top_20pct():
    t = pd.Timestamp("2024-01-01")
    df = pd.DataFrame({
        "open_time": [t]*10, "symbol": [f"S{i}" for i in range(10)],
        "momentum": list(range(10)),  # 0..9
    })
    out = top_quintile(df)
    # top 20% of 10 -> the top 2 (momentum 8, 9)
    assert set(out["symbol"]) == {"S8", "S9"}

def test_split_by_median_flow_labels_groups():
    t = pd.Timestamp("2024-01-01")
    merged = pd.DataFrame({
        "open_time": [t, t, t, t],
        "symbol": ["A", "B", "C", "D"],
        "momentum": [1, 1, 1, 1],
        "flow": [-0.5, -0.1, 0.1, 0.5],
    })
    out = split_by_median_flow(merged)
    labels = dict(zip(out["symbol"], out["group"]))
    assert labels["A"] == "divergent" and labels["D"] == "confirmed"

def test_divergence_rate_counts_negative_flow():
    df = pd.DataFrame({"flow": [-0.1, 0.2, -0.3, 0.4]})
    assert divergence_rate(df) == 0.5
