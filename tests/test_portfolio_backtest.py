import numpy as np
import pandas as pd
import pytest

from portfolio_construction.portfolio_backtest import RebalancingCalendar, Backtest


@pytest.fixture
def prices():
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    rng = np.random.default_rng(7)
    data = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=(len(dates), 2)), axis=0)
    return pd.DataFrame(data, index=dates, columns=["Asset1", "Asset2"])


@pytest.mark.parametrize("method", ["eom", "eoq", "eow", "eos", "bim", "eoy"])
def test_rebalancing_calendar_methods(prices, method):
    start = prices.index[300]
    end = prices.index[-1]
    cal = RebalancingCalendar(prices, start, end, method, lookback=250)

    assert cal.rebalCalendar is not None
    assert (cal.rebalCalendar >= start).all()
    assert (cal.rebalCalendar <= end).all()
    assert cal.rebalCalendar.is_monotonic_increasing


def test_rebalancing_calendar_insufficient_lookback(prices, capsys):
    start = prices.index[5]
    cal = RebalancingCalendar(prices, start, prices.index[-1], "eom", lookback=250)
    captured = capsys.readouterr()
    assert "not sufficient historical" in captured.out
    assert cal.rebalCalendar is None


def test_backtest_simulations_minimum_variance(prices):
    bt = Backtest()
    bt.initialize_parameters(
        prices,
        start_date="2022-01-03",
        end_date="2022-06-30",
        lookback=20,
    )
    bt.simulations(typeOpt="minimum_variance")

    assert bt.strat is not None
    assert len(bt.strat) == len(bt.dates_series)
    assert bt.strat["Strategy"].iloc[0] == pytest.approx(1.0)
    assert bt.historical_portfolios
