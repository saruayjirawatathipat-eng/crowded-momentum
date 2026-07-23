import pandas as pd

from obv_visualize_and_report import compute_obv_spread, summarize_spread_with_tstat, build_phase2_markdown, describe_cumulative_lead, update_conclusion, PHASE2_MARKER


def test_compute_obv_spread_is_confirmed_minus_divergent():
    group_means = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 2,
        "group": ["confirmed", "divergent"],
        "horizon": [1, 1],
        "mean_return": [0.03, 0.01],
    })
    result = compute_obv_spread(group_means)
    assert abs(result["spread"].iloc[0] - 0.02) < 1e-9
    assert abs(result["confirmed_mean"].iloc[0] - 0.03) < 1e-9


def test_summarize_spread_with_tstat_known_values():
    # spreads [0.01, 0.03]: mean 0.02, sample std 0.0141421356,
    # se = std/sqrt(2) = 0.01, t = 0.02/0.01 = 2.0
    spread_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
        "horizon": [1, 1],
        "confirmed_mean": [0.02, 0.04],
        "divergent_mean": [0.01, 0.01],
        "spread": [0.01, 0.03],
    })
    result = summarize_spread_with_tstat(spread_df)
    assert abs(result["mean_spread"].iloc[0] - 0.02) < 1e-9
    assert result["n_months"].iloc[0] == 2
    assert abs(result["t_stat"].iloc[0] - 2.0) < 1e-9


def _spread_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [1, 3, 6],
        "mean_spread": [0.01, 0.02, 0.03],
        "std_spread": [0.02, 0.03, 0.04],
        "n_months": [59, 57, 54],
        "t_stat": [3.84, 5.03, 5.51],
    })


def _diagnostic() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
        "n_stocks": [50, 50],
        "n_negative": [5, 3],
    })


def test_build_phase2_markdown_has_results_table_and_diagnostic():
    md = build_phase2_markdown(_spread_summary(), _diagnostic())
    assert md.startswith(PHASE2_MARKER)
    assert "| 1 | 0.0100 | 0.0200 | 59 | 3.84 |" in md
    assert "8.0%" in md  # 8 negative of 100 stock-months
    assert "Confirmed stocks outperformed divergent stocks" in md


def test_build_phase2_markdown_reports_opposite_direction():
    summary = _spread_summary()
    summary["mean_spread"] = [-0.01, -0.02, -0.03]
    md = build_phase2_markdown(summary, _diagnostic())
    assert "Divergent stocks outperformed confirmed stocks" in md


def test_build_phase2_markdown_flags_insignificant_result_as_noise():
    summary = _spread_summary()
    summary["mean_spread"] = [-0.001, -0.004, -0.009]
    summary["t_stat"] = [-0.14, -0.47, -0.98]
    md = build_phase2_markdown(summary, _diagnostic())
    assert (
        "None of the t-statistics come close to the conventional 1.96 threshold for "
        "significance at any horizon (largest |t| = 0.98), so this pattern should be "
        "read as noise, not a real effect." in md
    )


def _cum_df_confirmed_leads_mostly() -> pd.DataFrame:
    # confirmed ahead of divergent for 4 of 5 months, converging (crossing) at the end
    dates = pd.date_range("2021-08-01", periods=5, freq="MS")
    confirmed = [0.10, 0.30, 0.60, 1.00, 1.20]
    divergent = [0.02, 0.10, 0.30, 0.80, 1.30]
    rows = []
    for d, c, v in zip(dates, confirmed, divergent):
        rows.append({"date": d, "group": "confirmed", "cumulative_return": c})
        rows.append({"date": d, "group": "divergent", "cumulative_return": v})
    return pd.DataFrame(rows)


def test_describe_cumulative_lead_names_the_majority_leader():
    note = describe_cumulative_lead(_cum_df_confirmed_leads_mostly())
    assert "confirmed group actually led the divergent group for about 80%" in note
    assert "not a contradiction" in note


def test_build_phase2_markdown_omits_cumulative_note_when_cum_df_not_given():
    md = build_phase2_markdown(_spread_summary(), _diagnostic())
    assert "Worth reconciling with the chart" not in md


def test_build_phase2_markdown_includes_cumulative_note_when_cum_df_given():
    md = build_phase2_markdown(_spread_summary(), _diagnostic(), _cum_df_confirmed_leads_mostly())
    assert "Worth reconciling with the chart" in md
    assert "confirmed group actually led the divergent group" in md


def test_update_conclusion_appends_then_replaces():
    phase1 = "# Crowded Momentum Trades — Conclusion\n\nPhase 1 text."
    v1 = update_conclusion(phase1, PHASE2_MARKER + "\n\nfirst run")
    assert "Phase 1 text." in v1
    assert "first run" in v1
    v2 = update_conclusion(v1, PHASE2_MARKER + "\n\nsecond run")
    assert "Phase 1 text." in v2
    assert "second run" in v2
    assert "first run" not in v2
    assert v2.count(PHASE2_MARKER) == 1
