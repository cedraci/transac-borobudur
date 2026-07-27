import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_analysis as pa


@pytest.fixture
def prices():
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    rng = np.random.default_rng(1)
    daily_returns = rng.normal(loc=0.0003, scale=0.01, size=len(dates))
    values = 100 * np.cumprod(1 + daily_returns)
    return pd.Series(values, index=dates, name="Strategy")


def test_annualized_return_volatility_sharpe(prices):
    ann_ret = pa.annualized_return(prices)
    ann_vol = pa.annualized_volatility(prices)
    sharpe = pa.annualized_sharpe_ratio(prices, rf=0.0)
    assert isinstance(ann_ret, float)
    assert ann_vol > 0
    assert sharpe == pytest.approx((ann_ret - 0.0) / ann_vol)


def test_drawdown_helpers(prices):
    dd = pa.historical_drawdown(prices)
    assert (dd <= 1e-9).all()
    max_dd = pa.maximum_drawdown(prices)
    assert max_dd <= 0


def test_calendar_performances(prices):
    table = pa.calendar_performances(prices)
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for m in months:
        assert m in table.columns
    assert "Year" in table.columns


def test_var_and_es(prices):
    var = pa.historical_var(prices, alpha=0.05, duration=1)
    es = pa.historical_es(prices, alpha=0.05, duration=1)
    # expected shortfall should be at least as extreme as the VaR threshold
    assert es <= var


def test_gbm_multiple_path_normal():
    paths = pa.gbm_multiple_path(num=50, startprice=100, mu=0.05, sigma=0.2, days=30, distrib="normal")
    assert paths.shape == (31, 50)
    assert (paths[0] == 100).all()


def test_gbm_multiple_path_student():
    # this exercises the Student-t branch, which relies on scipy.stats internals
    paths = pa.gbm_multiple_path(num=50, startprice=100, mu=0.05, sigma=0.2, days=30, distrib="student")
    assert paths.shape == (31, 50)


def test_portfolio_path_cholesky_normal():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    paths = pa.portfolio_path_cholesky(
        n_sims=20, start_price=100, weights=[0.6, 0.4], mu=[0.05, 0.03],
        cov_matrix=cov, days=30, distrib="normal",
    )
    assert paths.shape == (31, 20)
    assert np.allclose(paths[0], 100)


def test_portfolio_path_cholesky_student():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    paths = pa.portfolio_path_cholesky(
        n_sims=20, start_price=100, weights=[0.6, 0.4], mu=[0.05, 0.03],
        cov_matrix=cov, days=30, distrib="student",
    )
    assert paths.shape == (31, 20)


def test_stats_report(prices):
    df = pd.DataFrame({"Strategy": prices})
    report = pa.stats_report(df)
    assert "Ann. Return" in report.columns
    assert report.shape[0] == 1


def test_compute_drawdown_periods_and_table(prices):
    navs = [(d.date(), v) for d, v in prices.items()]
    periods = pa.compute_drawdown_periods(navs, top_n=5, min_drawdown=0.01)
    assert isinstance(periods, list)
    if periods:
        assert periods[0].rank == 1
        # worst drawdown should be first (most negative)
        assert periods[0].drawdown <= periods[-1].drawdown

    table = pa.drawdowns_table(prices, top_n=5, min_drawdown=0.01)
    assert isinstance(table, pd.DataFrame)
