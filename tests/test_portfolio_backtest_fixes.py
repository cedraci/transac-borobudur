"""Regression tests for defects found while building the Backtest page.

All were reproduced against the installed pandas/numpy before being fixed.
"""

import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_backtest as pb


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


# --- non-trading dates ------------------------------------------------------


def test_snap_to_trading_day_returns_an_exact_match_unchanged():
    prices = _prices()
    target = prices.index[300]
    assert pb.snap_to_trading_day(prices.index, target) == target


def test_snap_to_trading_day_moves_a_weekend_forward():
    prices = _prices()
    saturday = pd.Timestamp("2020-01-04")  # a Saturday, absent from a bdate index
    assert saturday not in prices.index

    snapped = pb.snap_to_trading_day(prices.index, saturday)
    assert snapped in prices.index
    assert snapped > saturday


def test_snap_to_trading_day_can_move_backward():
    prices = _prices()
    saturday = pd.Timestamp("2020-01-04")
    snapped = pb.snap_to_trading_day(prices.index, saturday, direction="backward")
    assert snapped in prices.index
    assert snapped < saturday


def test_snap_to_trading_day_rejects_a_date_past_the_end_of_history():
    prices = _prices()
    with pytest.raises(ValueError, match="outside"):
        pb.snap_to_trading_day(prices.index, pd.Timestamp("2099-01-02"), "forward")


def test_snap_to_trading_day_rejects_a_date_before_history_when_looking_backward():
    prices = _prices()
    with pytest.raises(ValueError, match="outside"):
        pb.snap_to_trading_day(prices.index, pd.Timestamp("1990-01-02"), "backward")


def test_snap_to_trading_day_before_history_moves_forward_to_the_first_date():
    prices = _prices()
    snapped = pb.snap_to_trading_day(prices.index, pd.Timestamp("1990-01-02"), "forward")
    assert snapped == prices.index[0]


def test_initialize_parameters_accepts_a_non_trading_start_date():
    """A user picking a Saturday in a date picker used to hit IndexError."""
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-01-04", "2021-12-31", 250)  # Saturday
    assert bt.dates_series is not None
    assert len(bt.dates_series) > 0
    assert bt.dates_series[0] in prices.index


def test_initialize_parameters_still_accepts_an_exact_trading_date():
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-01-02", "2021-12-31", 250)
    assert bt.dates_series[0] == pd.Timestamp("2020-01-02")


def test_initialize_parameters_raises_when_the_lookback_is_too_long():
    """Previously printed a message and fell through with attributes unset."""
    prices = _prices()
    bt = pb.Backtest()
    with pytest.raises(ValueError, match="lookback"):
        bt.initialize_parameters(prices, "2019-01-10", "2020-12-31", 500)


# --- universe_selection on numpy 2 ------------------------------------------


def test_universe_selection_runs_on_modern_numpy():
    """float() on a size-1 ndarray raises TypeError on numpy 2.x."""
    prices = _prices()
    idx, names, scores = pb.universe_selection(prices, prices.index[400], 250, 2)
    assert len(names) == 2
    assert len(scores) == 2
    assert set(names).issubset(set(prices.columns))


def test_universe_selection_ranks_by_score_descending():
    prices = _prices()
    _, _, scores = pb.universe_selection(prices, prices.index[400], 250, 3)
    assert list(scores) == sorted(scores, reverse=True)


# --- rebalancing calendar wired into the backtest ---------------------------


def test_backtest_defaults_to_month_end_rebalancing():
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-01-02", "2021-12-31", 250)
    monthly = len(bt.sequence_rebal)
    assert monthly > 20  # roughly two years of month-ends


def test_backtest_honours_a_quarterly_rebalancing_method():
    prices = _prices()
    monthly_bt, quarterly_bt = pb.Backtest(), pb.Backtest()
    monthly_bt.initialize_parameters(prices, "2020-01-02", "2021-12-31", 250, method="eom")
    quarterly_bt.initialize_parameters(prices, "2020-01-02", "2021-12-31", 250, method="eoq")
    assert len(quarterly_bt.sequence_rebal) < len(monthly_bt.sequence_rebal)


def test_backtest_rebalancing_dates_are_all_trading_days():
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-01-02", "2021-12-31", 250, method="eoq")
    assert all(d in prices.index for d in bt.sequence_rebal)


# --- stock_picking reachable from simulations -------------------------------


def test_simulations_accepts_stock_picking():
    """target_weights took stock_picking, but simulations never passed it."""
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-06-01", "2020-09-30", 250, method="eom")
    bt.simulations("minimum_variance", stock_picking=True)
    assert bt.strat is not None
    assert len(bt.strat) == len(bt.dates_series)


def test_simulations_without_stock_picking_uses_every_column():
    prices = _prices()
    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-06-01", "2020-09-30", 250, method="eom")
    bt.simulations("minimum_variance")
    first_weights = bt.historical_portfolios[0]["weights"]
    assert set(first_weights) == set(prices.columns)


def test_target_weights_honours_nb_securities():
    """universe_selection was called with a hardcoded 250-window and 6 names."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", periods=800, name="Date")
    prices = pd.DataFrame(
        {
            f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012 + 0.002 * i, 800)))
            for i in range(6)
        },
        index=idx,
    )

    bt = pb.Backtest()
    bt.initialize_parameters(prices, "2020-06-01", "2020-09-30", 250, method="eom")
    weights = bt.target_weights(
        bt.dates_series[0], "minimum_variance", False, False,
        stock_picking=True, nb_securities=2,
    )
    assert len(weights) == 2
