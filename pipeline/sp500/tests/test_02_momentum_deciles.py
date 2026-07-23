import pandas as pd

from momentum_deciles import assign_deciles, compute_momentum


def test_compute_momentum_known_values():
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    prices = list(range(100, 113))
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "adj_close": prices, "volume": 1})
    result = compute_momentum(df)
    expected = prices[11] / prices[0] - 1
    row = result[result["date"] == dates[12]]
    assert len(row) == 1
    assert abs(row["momentum"].iloc[0] - expected) < 1e-9


def test_compute_momentum_drops_first_12_months():
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "adj_close": range(100, 113), "volume": 1})
    result = compute_momentum(df)
    assert len(result) == 1


def test_assign_deciles_ranks_correctly():
    date = pd.Timestamp("2020-01-01")
    df = pd.DataFrame({
        "date": [date] * 10,
        "ticker": [f"T{i}" for i in range(10)],
        "momentum": [float(i) for i in range(10)],
    })
    result = assign_deciles(df)
    top = result[result["ticker"] == "T9"]
    assert top["decile"].iloc[0] == 10


def test_assign_deciles_skips_dates_with_too_few_stocks():
    date = pd.Timestamp("2020-01-01")
    df = pd.DataFrame({
        "date": [date] * 5,
        "ticker": [f"T{i}" for i in range(5)],
        "momentum": [float(i) for i in range(5)],
    })
    result = assign_deciles(df)
    assert result.empty
