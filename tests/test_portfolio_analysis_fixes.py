"""Regression tests for defects found while building the Analysis page.

All were reproduced against the installed pandas/numpy before being fixed.
"""

import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_analysis as pa


def _nav(days=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days, name="Date")
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days))),
        index=idx,
        name="Portfolio",
    )


_STATS_COLUMNS = {
    "Name",
    "Ann. Return",
    "Ann. Volatility",
    "Ann. Sharpe",
    "Value-at-Risk",
    "Expected Shortfall (histo)",
    "Max Drawdown",
}


def test_drawdowns_table_honours_top_n():
    out = pa.drawdowns_table(_nav(), top_n=3)
    assert len(out) <= 3


def test_drawdowns_table_honours_min_drawdown():
    nav = _nav()
    shallow = pa.drawdowns_table(nav, top_n=50, min_drawdown=0.02)
    deep = pa.drawdowns_table(nav, top_n=50, min_drawdown=0.15)

    # Pre-fix, min_drawdown was hardcoded to 0.02, so both calls returned the
    # same rows and any <=-style assertion passed vacuously.
    assert len(deep) < len(shallow)
    assert (deep["drawdown"].abs() >= 0.15).all()


def test_drawdowns_table_returns_the_expected_columns():
    out = pa.drawdowns_table(_nav(), top_n=5)
    for column in ["rank", "start_date", "trough_date", "drawdown"]:
        assert column in out.columns


def test_cagr_rolled_runs_on_modern_numpy():
    nav = _nav()
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    out = pa.cagr_rolled(yearly, max_holding_period=3)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(yearly)
    assert out.shape[1] == min(3, len(yearly))


def test_cagr_rolled_first_column_is_the_yearly_return():
    nav = _nav()
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    out = pa.cagr_rolled(yearly, max_holding_period=3)
    assert out.iloc[0, 0] == pytest.approx(float(yearly.iloc[0, 0]))


def test_stats_report_handles_a_short_history_without_raising():
    # 500 business days from 2020-01-01 spans only 2020 and 2021 - two
    # calendar years. Pre-fix, stats_report hardcoded
    # cal_perf.index[-1..-3], so it raised
    # "IndexError: index -3 is out of bounds for axis 0 with size 2" here.
    idx = pd.bdate_range("2020-01-01", periods=500, name="Date")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))),
            "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(idx)))),
        },
        index=idx,
    )
    assert len(pa.calendar_performances(prices["AAA"])) < 3  # fixture sanity check

    out = pa.stats_report(prices)

    assert len(out) == len(prices.columns)
    assert set(out["Name"]) == {"AAA", "BBB"}


def test_stats_report_still_reports_three_trailing_years_for_a_long_history():
    # A history spanning 4+ calendar years must keep reporting exactly the
    # three most recent years, unchanged from before the fix.
    idx = pd.bdate_range("2015-01-01", periods=1500, name="Date")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))},
        index=idx,
    )
    cal_perf = pa.calendar_performances(prices["AAA"])
    assert len(cal_perf) >= 4  # fixture sanity check

    out = pa.stats_report(prices)

    year_columns = set(out.columns) - _STATS_COLUMNS
    assert year_columns == set(cal_perf.index[-3:].tolist())


def test_stats_report_does_not_leak_year_columns_between_columns():
    # fd_results used to be created once outside the per-column loop and
    # only .update()-ed, so trailing-year keys from a column with a longer
    # (or later-ending) history leaked into a later column's row even when
    # that column never had data for those years.
    idx = pd.bdate_range("2015-01-01", periods=2100, name="Date")
    rng = np.random.default_rng(0)
    long_history = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))
    prices = pd.DataFrame({"AAA": long_history}, index=idx)
    end_pos = idx.searchsorted(pd.Timestamp("2020-01-01"))
    prices["BBB"] = np.where(
        np.arange(len(idx)) >= end_pos, np.nan, long_history * 1.01
    )

    aaa_years = set(pa.calendar_performances(prices["AAA"]).index[-3:].tolist())
    bbb_years = set(pa.calendar_performances(prices["BBB"]).index[-3:].tolist())
    assert aaa_years.isdisjoint(bbb_years)  # fixture sanity check

    out = pa.stats_report(prices)
    bbb_row = out.loc[out["Name"] == "BBB"].iloc[0]

    # BBB's row must not carry AAA's trailing-year values for years BBB
    # never had data for, and must carry its own.
    for year in aaa_years:
        assert pd.isna(bbb_row[year])
    for year in bbb_years:
        assert not pd.isna(bbb_row[year])
