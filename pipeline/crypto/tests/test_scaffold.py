from pathlib import Path

import pytest


@pytest.mark.skip(reason="requires fetched klines; run pipeline first")
def test_data_and_charts_dirs_exist():
    # data/ and charts/ are created as a side effect of running
    # 01_fetch_klines.py and 07_sweep_and_report.py against fetched Binance
    # klines; that data is not committed to this repo (see .gitignore), so
    # the directories don't exist here until the pipeline has actually run.
    root = Path(__file__).resolve().parent.parent
    assert (root / "data").is_dir()
    assert (root / "charts").is_dir()
