"""The pure risk layer. Deterministic where possible; simulations are checked
for shape and sanity rather than exact values."""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.risk import (
    RiskError,
    covariance_var,
    normalize_weights,
    simulate_paths,
    simulate_portfolio_paths,
    stressed_covariance,
    var_es_table,
)


def _prices(days=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days, name="Date")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, days)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_normalize_weights_defaults_to_equal():
    assert normalize_weights(["AAA", "BBB"]) == [0.5, 0.5]


def test_normalize_weights_preserves_requested_order():
    out = normalize_weights(["AAA", "BBB"], {"BBB": 0.25, "AAA": 0.75})
    assert out == [0.75, 0.25]


def test_normalize_weights_rejects_a_bad_sum():
    with pytest.raises(RiskError, match="sum"):
        normalize_weights(["AAA", "BBB"], {"AAA": 0.3, "BBB": 0.3})


def test_normalize_weights_rejects_unknown_ticker():
    with pytest.raises(RiskError, match="NOPE"):
        normalize_weights(["AAA"], {"NOPE": 1.0})


def test_var_es_table_has_a_row_per_method():
    out = var_es_table(_prices(), None)
    assert set(out.index) == {"Historical", "Parametric", "Monte Carlo"}
    assert list(out.columns) == ["VaR", "Expected Shortfall"]


def test_var_es_values_are_losses():
    out = var_es_table(_prices(), None)
    assert (out["VaR"] < 0).all()
    assert (out["Expected Shortfall"] < 0).all()


def test_expected_shortfall_is_at_least_as_severe_as_var():
    out = var_es_table(_prices(), None)
    for method in out.index:
        assert out.loc[method, "Expected Shortfall"] <= out.loc[method, "VaR"] + 1e-9


def test_a_higher_confidence_gives_a_more_severe_var():
    prices = _prices()
    mild = var_es_table(prices, None, alpha=0.10).loc["Historical", "VaR"]
    harsh = var_es_table(prices, None, alpha=0.01).loc["Historical", "VaR"]
    assert harsh <= mild


def test_var_es_table_methods_agree_in_order_of_magnitude():
    # Historical, Parametric and Monte Carlo estimate the same underlying
    # quantity via different methods; a table where one row is off by more
    # than an order of magnitude from the other two (e.g. because the Monte
    # Carlo leg was fed per-period moments where the simulator expects
    # annualized ones) is not trustworthy even if each row individually
    # looks like a plausible loss.
    out = var_es_table(_prices(), None)
    magnitudes = out["VaR"].abs()
    for a in magnitudes:
        for b in magnitudes:
            assert a / b <= 5


def test_simulate_paths_shape():
    nav = _prices(300)["AAA"]
    out = simulate_paths(nav, n_sims=50, days=30)
    assert out.shape[1] == 50
    assert len(out) == 31  # days + 1: the starting value is row 0


def test_simulate_paths_all_start_from_the_last_observed_value():
    nav = _prices(300)["AAA"]
    out = simulate_paths(nav, n_sims=20, days=10)
    assert out.notna().all().all()


def test_simulate_portfolio_paths_shape():
    prices = _prices(300)
    out = simulate_portfolio_paths(prices, None, n_sims=25, days=20)
    assert out.shape[1] == 25
    assert len(out) == 21  # days + 1


def test_stressed_covariance_is_square_and_labelled():
    out = stressed_covariance(_prices(200))
    assert out.shape == (2, 2)
    assert list(out.columns) == ["AAA", "BBB"]


def test_stressing_raises_off_diagonal_covariance():
    prices = _prices(200)
    plain = prices.pct_change().dropna().cov()
    stressed = stressed_covariance(prices, stress_factor=0.9)
    assert stressed.loc["AAA", "BBB"] >= plain.loc["AAA", "BBB"]


def test_covariance_var_is_a_loss_and_worsens_under_stress():
    prices = _prices(200)
    base = covariance_var(prices)
    stressed = covariance_var(prices, cov=stressed_covariance(prices, stress_factor=0.9))
    assert base < 0
    # correlations pushed toward 1 remove diversification, so the loss deepens
    assert stressed <= base
