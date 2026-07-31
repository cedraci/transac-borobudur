"""The pure optimization layer: prices in, weights and diagnostics out."""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.optimize import (
    OBJECTIVES,
    OptimizeError,
    bayes_stein_returns,
    capital_protection,
    correlation_matrix,
    efficient_frontier,
    optimize,
    risk_contributions,
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


def test_objectives_lists_every_supported_type():
    assert set(OBJECTIVES) == {
        "minimum_variance",
        "most_diversified",
        "maximum_sharpe",
        "minimum_VaR",
        "equal_risk_contribution",
        "maximum_return",
        "inverse_volatility",
        "mean_variance",
    }


@pytest.mark.parametrize("objective", sorted(OBJECTIVES))
def test_every_objective_returns_weights_summing_to_one(objective):
    weights = optimize(_prices(), objective)
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)
    assert list(weights.index) == ["AAA", "BBB", "CCC"]


@pytest.mark.parametrize("objective", sorted(OBJECTIVES))
def test_every_objective_is_long_only_by_default(objective):
    weights = optimize(_prices(), objective)
    assert (weights >= -1e-6).all()


@pytest.mark.parametrize("cov_mat", ["sample", "shrunked", "gerber"])
def test_every_covariance_estimator_works(cov_mat):
    weights = optimize(_prices(), "minimum_variance", cov_mat=cov_mat)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_minimum_variance_is_less_volatile_than_equal_weight():
    """The defining property of the objective - if this fails it isn't minimising."""
    prices = _prices()
    returns = prices.pct_change().dropna()
    mv = optimize(prices, "minimum_variance")

    mv_vol = returns.mul(mv).sum(axis=1).std()
    eq_vol = returns.mul(pd.Series(1 / 3, index=returns.columns)).sum(axis=1).std()
    assert mv_vol <= eq_vol


def test_maximum_return_concentrates_on_the_best_performer():
    prices = _prices()
    weights = optimize(prices, "maximum_return")
    best = prices.pct_change().dropna().mean().idxmax()
    assert weights[best] == pytest.approx(weights.max())


def test_inverse_volatility_gives_the_calmest_asset_the_largest_weight():
    prices = _prices()
    weights = optimize(prices, "inverse_volatility")
    calmest = prices.pct_change().dropna().std().idxmin()
    assert weights.idxmax() == calmest


def test_optimize_rejects_an_unknown_objective():
    with pytest.raises(OptimizeError, match="unknown objective"):
        optimize(_prices(), "make_me_rich")


def test_optimize_rejects_a_single_row_input():
    with pytest.raises(OptimizeError, match="at least two"):
        optimize(_prices(1), "minimum_variance")


def test_optimize_honours_bounds():
    weights = optimize(_prices(), "minimum_variance", bounds=(0.2, 0.5))
    assert (weights >= 0.2 - 1e-4).all()
    assert (weights <= 0.5 + 1e-4).all()


def test_risk_contributions_sum_to_one():
    prices = _prices()
    weights = optimize(prices, "equal_risk_contribution")
    contributions = risk_contributions(prices, weights)
    assert isinstance(contributions, pd.Series)
    assert contributions.sum() == pytest.approx(1.0, abs=1e-3)


def test_equal_risk_contribution_really_equalises_risk():
    """The whole point of the objective; a wide spread means it failed."""
    prices = _prices()
    contributions = risk_contributions(prices, optimize(prices, "equal_risk_contribution"))
    assert contributions.max() - contributions.min() < 0.02


def test_correlation_matrix_is_square_symmetric_and_unit_diagonal():
    corr = correlation_matrix(_prices())
    assert corr.shape == (3, 3)
    assert list(corr.columns) == ["AAA", "BBB", "CCC"]
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)
    assert np.allclose(corr.values, corr.values.T, atol=1e-6)


def test_gerber_correlation_matrix_is_also_well_formed():
    corr = correlation_matrix(_prices(), method="gerber")
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)


def test_bayes_stein_pulls_estimates_toward_the_cross_sectional_mean():
    prices = _prices()
    raw = prices.pct_change().dropna().mean()
    shrunk = bayes_stein_returns(prices)

    assert isinstance(shrunk, pd.Series)
    assert list(shrunk.index) == list(raw.index)
    # shrinkage must reduce the spread of the estimates
    assert shrunk.max() - shrunk.min() <= raw.max() - raw.min() + 1e-12


def test_capital_protection_returns_weights_and_solver_diagnostics():
    result = capital_protection(_prices(), duration=5, confidence=0.95)

    assert isinstance(result.weights, pd.Series)
    assert list(result.weights.index) == ["AAA", "BBB", "CCC"]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-3)
    # the solver often fails to converge on real data; the page must be able to
    # say so rather than presenting weights from a failed solve as if they held
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)


def test_efficient_frontier_is_ordered_and_upward_sloping():
    frontier = efficient_frontier(_prices(), points=6)
    assert isinstance(frontier, pd.DataFrame)
    assert list(frontier.columns) == ["Return", "Volatility"]
    assert len(frontier) > 1
    assert frontier["Return"].is_monotonic_increasing


def test_efficient_frontier_rejects_too_few_points():
    with pytest.raises(OptimizeError, match="at least two"):
        efficient_frontier(_prices(), points=1)
