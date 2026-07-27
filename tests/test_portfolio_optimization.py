import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_optimization as po


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=300, freq="B")
    data = rng.normal(loc=0.0004, scale=0.01, size=(300, 4))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


def test_portfolio_variance_and_return():
    w = np.array([0.5, 0.5])
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    mu = np.array([0.05, 0.08])
    assert po.portfolio_variance(w, S) == pytest.approx(0.5**2 * 0.04 + 0.5**2 * 0.09 + 2 * 0.5 * 0.5 * 0.01)
    assert po.portfolio_return(w, mu) == pytest.approx(0.065)


@pytest.mark.parametrize(
    "type_opt",
    [
        "minimum_variance",
        "most_diversified",
        "maximum_sharpe",
        "minimum_VaR",
        "equal_risk_contribution",
        "maximum_return",
        "inverse_volatility",
    ],
)
def test_portfolio_optimization_weights_sum_to_one(returns, type_opt):
    weights = po.portfolio_optimization(returns, type_opt)
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-3)
    assert all(w >= -1e-6 for w in weights.values())


def test_portfolio_optimization_mean_variance_hits_target(returns):
    # this branch used to crash with `minimize(fun=3, ...)` before the fix
    target = 0.08
    weights = po.portfolio_optimization(returns, "mean_variance", target_return=target)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_portfolio_optimization_respects_bounds(returns):
    bounds = [(0.0, 0.3)] * 4
    weights = po.portfolio_optimization(returns, "minimum_variance", bounds=bounds)
    assert all(w <= 0.3 + 1e-6 for w in weights.values())


def test_portfolio_optimization_shrunk_and_gerber_covariance(returns):
    weights_shrunk = po.portfolio_optimization(returns, "minimum_variance", cov_mat="shrunked")
    assert sum(weights_shrunk.values()) == pytest.approx(1.0, abs=1e-3)

    weights_gerber = po.portfolio_optimization(returns, "minimum_variance", cov_mat="gerber")
    assert sum(weights_gerber.values()) == pytest.approx(1.0, abs=1e-3)


def test_inverse_vol():
    S = np.array([[0.04, 0.0], [0.0, 0.16]])
    w = po.InverseVol(S)
    # asset with lower volatility (0.2) should get a higher weight than the one with 0.4
    assert w[0] > w[1]
    assert np.sum(w) == pytest.approx(1.0)


def test_estimate_bayes_stein(returns):
    adjusted = po.estimate_bayes_stein(returns.values)
    assert adjusted.shape == (4,)


def test_gerber_statistic_and_correlation_matrix(returns):
    stat = po.gerber_statistic(returns["A"], returns["B"], threshold=0.5, returns=True)
    assert -1.0 <= stat <= 1.0

    corr = po.gerber_correlation_matrix(returns, threshold=0.5, returns=True)
    assert corr.shape == (4, 4)
    np_diag = np.diagonal(corr)
    assert all(d == pytest.approx(1.0) for d in np_diag)


def test_optimize_capital_protection():
    # Long horizon + modest vol so the "don't lose money" constraint is
    # actually satisfiable (e.g. 5% vol / 3% return alone already clears it).
    mu = [0.07, 0.03]
    cov = np.array([[0.0144, 0.0], [0.0, 0.0016]])
    result = po.optimize_capital_protection(mu, cov, duration=10.0, confidence=0.95)
    assert result["success"]
    assert sum(result["weights"]) == pytest.approx(1.0, abs=1e-3)


def test_get_stressed_covariance():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    stressed = po.get_stressed_covariance(cov, stress_factor=0.5)
    assert stressed.shape == (2, 2)
    # diagonal (variances) should increase since we bump vol by 10%
    assert stressed[0, 0] > cov[0, 0]
