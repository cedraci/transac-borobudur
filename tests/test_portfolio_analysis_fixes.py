"""Regression tests for two defects found while building the Analysis page.

Both were reproduced against the installed pandas/numpy before being fixed.
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
