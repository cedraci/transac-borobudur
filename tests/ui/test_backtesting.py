"""The pure backtest layer: prices in, equity curve and weight history out."""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.backtesting import (
    REBALANCING_METHODS,
    BacktestError,
    BacktestResult,
    momentum_scores,
    run_backtest,
)


def _prices(days=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=days, name="Date")
    return pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.011, days))),
            "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.017, days))),
            "CCC": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, days))),
        },
        index=idx,
    )


def test_rebalancing_methods_are_the_ones_the_calendar_implements():
    assert set(REBALANCING_METHODS) == {"eom", "eoq", "eos", "eow", "bim"}


def test_run_backtest_returns_a_result_with_a_rebased_curve():
    result = run_backtest(_prices(), "minimum_variance", "2020-06-01", "2020-12-31", 250)

    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.Series)
    assert result.equity_curve.iloc[0] == pytest.approx(100.0)
    assert len(result.equity_curve) > 0


def test_equity_curve_is_indexed_by_date():
    result = run_backtest(_prices(), "minimum_variance", "2020-06-01", "2020-12-31", 250)
    assert isinstance(result.equity_curve.index, pd.DatetimeIndex)


def test_weights_history_has_a_row_per_date_and_column_per_ticker():
    prices = _prices()
    result = run_backtest(prices, "minimum_variance", "2020-06-01", "2020-12-31", 250)

    assert isinstance(result.weights, pd.DataFrame)
    assert set(result.weights.columns) <= set(prices.columns)
    assert len(result.weights) == len(result.equity_curve)


def test_weights_are_recorded_for_every_rebalancing_date():
    result = run_backtest(_prices(), "minimum_variance", "2020-06-01", "2020-12-31", 250)
    assert len(result.rebalancing_dates) > 0
    for date in result.rebalancing_dates:
        assert date in result.weights.index


def test_a_non_trading_start_date_is_snapped_and_reported():
    """A date picker returns calendar days; the index holds trading days."""
    result = run_backtest(_prices(), "minimum_variance", "2020-01-04", "2020-12-31", 250)
    assert result.start != "2020-01-04"  # snapped forward off the Saturday
    assert pd.Timestamp(result.start) in _prices().index


def test_quarterly_rebalancing_trades_less_often_than_monthly():
    prices = _prices()
    monthly = run_backtest(prices, "minimum_variance", "2020-01-02", "2021-06-30", 250, method="eom")
    quarterly = run_backtest(prices, "minimum_variance", "2020-01-02", "2021-06-30", 250, method="eoq")
    assert len(quarterly.rebalancing_dates) < len(monthly.rebalancing_dates)


def test_run_backtest_rejects_an_unknown_objective():
    with pytest.raises(BacktestError, match="unknown objective"):
        run_backtest(_prices(), "make_me_rich", "2020-06-01", "2020-12-31", 250)


def test_run_backtest_rejects_an_unknown_rebalancing_method():
    with pytest.raises(BacktestError, match="unknown rebalancing"):
        run_backtest(_prices(), "minimum_variance", "2020-06-01", "2020-12-31", 250, method="daily")


def test_run_backtest_reports_an_impossible_lookback_clearly():
    with pytest.raises(BacktestError, match="lookback"):
        run_backtest(_prices(), "minimum_variance", "2019-01-10", "2020-12-31", 600)


def test_run_backtest_rejects_a_start_after_the_end():
    with pytest.raises(BacktestError, match="after"):
        run_backtest(_prices(), "minimum_variance", "2021-06-01", "2020-06-01", 250)


def _wide_prices(tickers=6, days=800, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=days, name="Date")
    return pd.DataFrame(
        {
            f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012 + 0.002 * i, days)))
            for i in range(tickers)
        },
        index=idx,
    )


def test_stock_picking_holds_only_the_requested_number_of_names():
    """nb_securities used to be ignored: the selector hardcoded 6 names."""
    prices = _wide_prices(tickers=6)
    result = run_backtest(
        prices, "minimum_variance", "2020-06-01", "2020-09-30", 250,
        stock_picking=True, nb_securities=2,
    )
    held = result.weights.notna().sum(axis=1)
    assert held.max() <= 2


def test_stock_picking_without_a_count_uses_the_default_universe_size():
    prices = _wide_prices(tickers=6)
    result = run_backtest(
        prices, "minimum_variance", "2020-06-01", "2020-09-30", 250, stock_picking=True
    )
    assert result.weights.notna().sum(axis=1).max() > 2


def test_momentum_scores_ranks_every_ticker():
    prices = _prices()
    scores = momentum_scores(prices, prices.index[400], window=250)

    assert isinstance(scores, pd.Series)
    assert set(scores.index) == set(prices.columns)
    assert scores.is_monotonic_decreasing


def test_momentum_scores_rejects_a_window_longer_than_the_history():
    prices = _prices(300)
    with pytest.raises(BacktestError, match="window"):
        momentum_scores(prices, prices.index[-1], window=500)


def test_result_summary_mentions_the_objective_and_frequency():
    result = run_backtest(_prices(), "minimum_variance", "2020-06-01", "2020-12-31", 250, method="eoq")
    summary = result.summary()
    assert "minimum_variance" in summary
    assert "eoq" in summary
