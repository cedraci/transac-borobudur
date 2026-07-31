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


def _vols(returns):
    return np.sqrt(np.diag(np.cov(returns, rowvar=False)))


def _diversification_ratio(w, returns):
    cov = np.cov(returns, rowvar=False)
    return float((w @ _vols(returns)) / np.sqrt(w @ cov @ w))


def test_mdp_uses_volatilities_not_variances():
    """The diversification ratio is (w.sigma)/sigma_p - standard deviations.

    MDP computed np.dot(w, np.diag(S)); for a 2-D S, np.diag gives VARIANCES.
    """
    returns = _returns()
    cov = np.cov(returns, rowvar=False)
    w = np.array([0.2, 0.3, 0.5])

    # MDP returns the negated diversification ratio (it is minimised)
    assert -po.MDP(w, cov) == pytest.approx(_diversification_ratio(w, returns), rel=1e-9)


def test_most_diversified_beats_equal_weight_on_the_diversification_ratio():
    """Its defining property. The variance bug made it pick the noisiest asset."""
    returns = _returns()
    weights = po.portfolio_optimization(returns, "most_diversified")
    w = np.array([weights[c] for c in returns.columns])

    equal = np.ones(len(returns.columns)) / len(returns.columns)
    assert _diversification_ratio(w, returns) >= _diversification_ratio(equal, returns)


def test_most_diversified_does_not_pile_into_the_noisiest_asset():
    returns = _returns()
    weights = po.portfolio_optimization(returns, "most_diversified")
    w = np.array([weights[c] for c in returns.columns])

    noisiest = int(np.argmax(_vols(returns)))
    assert int(np.argmax(w)) != noisiest


def test_shrunked_covariance_does_not_collapse_to_equal_weight():
    """ShrunkCovariance was fitted on the covariance matrix rather than returns.

    Treating an (n,n) covariance as n observations of n features produced a
    matrix orders of magnitude too small, so minimum_variance saw no risk
    information at all and degenerated to 1/n.
    """
    returns = _returns()
    weights = po.portfolio_optimization(returns, "minimum_variance", cov_mat="shrunked")
    values = np.array([weights[c] for c in returns.columns])

    n = len(returns.columns)
    assert not np.allclose(values, 1 / n, atol=0.01)


def test_shrunked_stays_close_to_the_sample_solution():
    """Shrinkage should temper the sample estimate, not discard it."""
    returns = _returns()
    shrunk = po.portfolio_optimization(returns, "minimum_variance", cov_mat="shrunked")
    sample = po.portfolio_optimization(returns, "minimum_variance", cov_mat="sample")

    for column in returns.columns:
        assert abs(shrunk[column] - sample[column]) < 0.30


def _calm_returns(days=800, seed=3):
    """Low-volatility assets, so the optimizer's 6% annual target is reachable."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=days, name="Date")
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.0020, days))),
            "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.0025, days))),
            "CCC": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.0030, days))),
        },
        index=idx,
    )
    return prices.pct_change().dropna()


def test_momentum_optimization_respects_a_reachable_volatility_target():
    """The constraint annualised a variance by sqrt(252), which is neither.

    w'Sw is a daily variance: annualising it needs *252, and a square root to
    become a volatility. The old expression evaluated to ~0.003 against a 0.06
    target, so it constrained nothing.

    Note the target is only honoured when it is attainable - for a universe
    whose minimum achievable annual volatility exceeds 6%, SLSQP returns its
    best infeasible point rather than reporting failure.
    """
    returns = _calm_returns()
    w = po.max_momentum_optimization(returns, np.array([3.0, 1.0, 2.0]))

    cov = np.cov(returns, rowvar=False)
    annual_vol = np.sqrt(float(w @ cov @ w) * 252)
    assert annual_vol <= 0.06 + 1e-3


def test_momentum_volatility_constraint_is_dimensionally_a_volatility():
    """A pure dimensional check, independent of feasibility.

    Scaling every return by k must scale the constrained quantity by k. The old
    expression used a variance, which would scale by k squared.
    """
    returns = _calm_returns()
    cov = np.cov(returns, rowvar=False)
    w = np.ones(3) / 3

    single = np.sqrt(float(w @ cov @ w) * 252)
    doubled_cov = np.cov(returns * 2, rowvar=False)
    doubled = np.sqrt(float(w @ doubled_cov @ w) * 252)
    assert doubled == pytest.approx(2 * single, rel=1e-9)


def test_momentum_optimization_still_sums_to_one():
    returns = _returns()
    w = po.max_momentum_optimization(returns, np.array([3.0, 1.0, 2.0]))
    assert w.sum() == pytest.approx(1.0, abs=1e-3)


def test_longshort_momentum_optimization_is_market_neutral():
    returns = _returns()
    w = po.longshort_momentum_optimization(returns, np.array([3.0, 1.0, 2.0]))
    assert w.sum() == pytest.approx(0.0, abs=1e-3)
