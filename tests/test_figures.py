import math
import matplotlib
matplotlib.use("Agg")
import pytest
from analysis import figures


def test_phase1_turnover_matches_known_spreads():
    df = figures.phase1_turnover_table()
    spreads = dict(zip(df["horizon"], df["mean_spread"]))
    assert spreads[1] == pytest.approx(0.0153, abs=5e-4)
    assert spreads[3] == pytest.approx(0.0580, abs=5e-4)
    assert spreads[6] == pytest.approx(0.1264, abs=5e-4)


def test_phase2_obv_matches_known_tstats():
    df = figures.phase2_obv_table()
    by_h = df.set_index("horizon")
    assert by_h.loc[1, "mean_spread"] == pytest.approx(-0.0007, abs=5e-4)
    assert by_h.loc[1, "t_stat"] == pytest.approx(-0.14, abs=0.05)
    assert by_h.loc[6, "t_stat"] == pytest.approx(-0.98, abs=0.05)


def test_phase3_sweep_ordered_and_matches():
    df = figures.phase3_sweep_table()
    assert list(df["timeframe"].unique()) == ["monthly", "weekly", "daily", "4h", "1h"]
    wk4 = df[(df["timeframe"] == "weekly") & (df["horizon"] == 4)].iloc[0]
    assert wk4["t_stat"] == pytest.approx(3.857, abs=0.01)
    assert wk4["mean_spread"] == pytest.approx(0.0337, abs=5e-4)


def test_significance_figure_builds():
    fig = figures.significance_figure()
    assert fig is not None
    assert len(fig.axes) >= 1
