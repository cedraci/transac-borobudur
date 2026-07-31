"""Regression tests for defects found while building the Optimization page."""

import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_optimization as po


def _returns(days=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=days, name="Date")
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.011, days))),
            "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.017, days))),
            "CCC": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, days))),
        },
        index=idx,
    )
    return prices.pct_change().dropna()


def test_risk_contribution_returns_one_value_per_asset():
    """It used to evaluate S * w.T, a matrix product only for np.matrix.

    Given a plain ndarray - which is what np.cov returns and what every caller
    passes - it broadcast elementwise and returned an n-by-n array.
    """
    returns = _returns()
    cov = np.cov(returns, rowvar=False)
    w = np.array([1 / 3, 1 / 3, 1 / 3])

    rc = np.asarray(po.risk_contribution(w, cov))
    assert rc.shape == (3,)


def test_risk_contributions_sum_to_portfolio_volatility():
    """The defining identity: absolute contributions add up to sigma_p."""
    returns = _returns()
    cov = np.cov(returns, rowvar=False)
    w = np.array([0.5, 0.3, 0.2])

    rc = np.asarray(po.risk_contribution(w, cov))
    sigma_p = np.sqrt(w @ cov @ w)
    assert rc.sum() == pytest.approx(sigma_p, rel=1e-9)


def test_equal_risk_contribution_actually_equalises_risk():
    """The objective's whole purpose.

    The broken risk_contribution corrupted ERC's error term, so the optimizer
    scored an unequal allocation (38.8/29.7/31.4) as better than the correct
    one (33.3 each).
    """
    returns = _returns()
    weights = po.portfolio_optimization(returns, "equal_risk_contribution")
    w = np.array([weights[c] for c in returns.columns])
    cov = np.cov(returns, rowvar=False)

    shares = w * (cov @ w) / (w @ cov @ w)
    assert shares.max() - shares.min() < 0.01


def test_erc_objective_scores_the_equal_allocation_best():
    """A direct check on the objective rather than on the solver."""
    returns = _returns()
    cov = np.cov(returns, rowvar=False)

    from scipy.optimize import minimize

    n = cov.shape[0]

    def spread(w):
        shares = w * (cov @ w) / (w @ cov @ w)
        return ((shares - 1 / n) ** 2).sum() * 1e4

    truly_equal = minimize(
        spread, np.ones(n) / n, bounds=[(0, 1)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1}), method="SLSQP",
    ).x

    lopsided = np.array([0.6, 0.2, 0.2])
    assert po.ERC(truly_equal, cov) < po.ERC(lopsided, cov)
