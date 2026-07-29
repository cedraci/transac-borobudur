"""The pure analysis layer: dataset in, display-ready frames out.

No streamlit, no network. Deterministic synthetic prices so assertions can be
exact rather than approximate.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.analytics import (
    AnalyticsError,
    calendar_table,
    drawdown_episodes,
    drawdown_series,
    monthly_returns_table,
    performance_table,
    rebased_prices,
    rolling_cagr,
    to_returns,
    weighted_nav,
)


def _prices(days=760, seed=0):
    """Two assets, deterministic pseudo-random walk, business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days, name="Date")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, days)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_to_returns_drops_the_first_row():
    prices = _prices(10)
    out = to_returns(prices)
    assert len(out) == len(prices) - 1
    assert list(out.columns) == ["AAA", "BBB"]


def test_to_returns_computes_simple_returns():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0]},
        index=pd.bdate_range("2020-01-01", periods=2, name="Date"),
    )
    assert to_returns(prices)["AAA"].iloc[0] == pytest.approx(0.10)


def test_weighted_nav_equal_weights_by_default():
    prices = _prices(50)
    nav = weighted_nav(prices)
    assert isinstance(nav, pd.Series)
    assert nav.iloc[0] == pytest.approx(100.0)


def test_weighted_nav_honours_explicit_weights():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [100.0, 100.0]},
        index=pd.bdate_range("2020-01-01", periods=2, name="Date"),
    )
    nav = weighted_nav(prices, {"AAA": 1.0, "BBB": 0.0})
    assert nav.iloc[-1] == pytest.approx(110.0)


def test_weighted_nav_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(AnalyticsError, match="sum"):
        weighted_nav(_prices(10), {"AAA": 0.3, "BBB": 0.3})


def test_weighted_nav_rejects_unknown_ticker():
    with pytest.raises(AnalyticsError, match="NOPE"):
        weighted_nav(_prices(10), {"NOPE": 1.0})


def test_performance_table_has_one_row_per_column():
    out = performance_table(_prices(500))
    assert len(out) == 2
    assert "Name" in out.columns
    assert set(out["Name"]) == {"AAA", "BBB"}


def test_performance_table_reports_the_headline_statistics():
    out = performance_table(_prices(500))
    for column in ["Ann. Return", "Ann. Volatility", "Ann. Sharpe", "Max Drawdown"]:
        assert column in out.columns


def test_drawdown_series_is_never_positive():
    dd = drawdown_series(weighted_nav(_prices(500)))
    assert dd.max() <= 1e-9


def test_drawdown_series_is_zero_at_a_new_peak():
    nav = pd.Series(
        [100.0, 110.0, 120.0], index=pd.bdate_range("2020-01-01", periods=3)
    )
    assert drawdown_series(nav).iloc[-1] == pytest.approx(0.0)


def test_drawdown_episodes_returns_a_frame():
    out = drawdown_episodes(weighted_nav(_prices(500)), top_n=5)
    assert isinstance(out, pd.DataFrame)
    assert len(out) <= 5


def test_calendar_table_covers_every_year_present():
    nav = weighted_nav(_prices(760))  # spans 2020 into 2022
    out = calendar_table(nav)
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 2


def test_rolling_cagr_returns_a_frame():
    out = rolling_cagr(weighted_nav(_prices(760)), max_holding_period=2)
    assert isinstance(out, pd.DataFrame)


def test_monthly_returns_table_has_a_row_per_month():
    nav = weighted_nav(_prices(260))  # about a year of business days
    out = monthly_returns_table(nav)
    assert isinstance(out, pd.DataFrame)
    assert 10 <= len(out) <= 13


def test_rebased_prices_start_at_100_on_the_chosen_date():
    prices = _prices(100)
    anchor = prices.index[20]
    out = rebased_prices(prices, anchor)
    assert out.iloc[0].round(6).eq(100.0).all()
    assert out.index[0] == anchor


def test_rebased_prices_rejects_a_date_not_in_the_index():
    with pytest.raises(AnalyticsError, match="not a date in this dataset"):
        rebased_prices(_prices(100), pd.Timestamp("1999-01-04"))


def test_functions_reject_a_single_row_input():
    one_row = _prices(1)
    with pytest.raises(AnalyticsError, match="at least two"):
        to_returns(one_row)
