"""VaR, expected shortfall and Monte Carlo simulation, as plain functions.

No streamlit import. The Risk page renders what this returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_construction import portfolio_analysis as pa
from portfolio_construction.portfolio_optimization import get_stressed_covariance
from portfolio_construction.stats import (
    annualization_factor,
    covariance_parametric_var,
    time_series_frequence_inference,
)


class RiskError(RuntimeError):
    """An input cannot support the requested risk statistic."""


def _annualization_factor(index: pd.Index) -> float:
    """The scaling factor to turn per-period moments into annualized ones.

    `gbm_multiple_path` and `portfolio_path_cholesky` both assume annualized
    mu/sigma (they scale by dt = 1/252 internally), so any per-period moment
    computed from returns must be annualized before being handed to them.
    """
    try:
        freq = time_series_frequence_inference(index)
        return float(annualization_factor(freq))
    except (ValueError, KeyError) as exc:
        raise RiskError(
            "cannot infer a time series frequency to annualize risk statistics"
        ) from exc


def normalize_weights(tickers, weights: dict[str, float] | None = None) -> list[float]:
    """A weight vector aligned to `tickers`, equal-weighted when unspecified."""
    tickers = list(tickers)
    if not tickers:
        raise RiskError("no tickers to weight")

    if weights is None:
        return [1.0 / len(tickers)] * len(tickers)

    unknown = [t for t in weights if t not in tickers]
    if unknown:
        raise RiskError(f"not in the dataset: {', '.join(unknown)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise RiskError(f"weights must sum to 1.0, got {total:.4f}")

    return [float(weights.get(t, 0.0)) for t in tickers]


def var_es_table(
    prices: pd.DataFrame,
    weights: dict[str, float] | None,
    alpha: float = 0.05,
    duration: int = 1,
    distrib: str = "normal",
    n_sims: int = 10000,
) -> pd.DataFrame:
    """VaR and expected shortfall by all three estimation families."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to estimate risk")

    vec_w = normalize_weights(prices.columns, weights)
    portfolio = prices.mul(pd.Series(vec_w, index=prices.columns)).sum(axis=1)
    returns = portfolio.pct_change().dropna()

    factor = _annualization_factor(prices.index)
    mu_annual = float(returns.mean()) * factor
    sigma_annual = float(returns.std()) * np.sqrt(factor)
    start = float(portfolio.iloc[-1])

    rows = {
        "Historical": (
            pa.historical_var(portfolio, alpha, duration),
            pa.historical_es(portfolio, alpha, duration),
        ),
        "Parametric": (
            pa.parametric_var(prices, vec_w, alpha, duration, distrib),
            pa.parametric_es(prices, vec_w, alpha, duration),
        ),
        "Monte Carlo": (
            pa.monteCarlo_var(n_sims, start, mu_annual, sigma_annual, alpha, duration, distrib),
            pa.monteCarlo_es(n_sims, start, mu_annual, sigma_annual, alpha, duration, distrib),
        ),
    }

    return pd.DataFrame(
        [{"VaR": float(v), "Expected Shortfall": float(es)} for v, es in rows.values()],
        index=list(rows),
    )


def simulate_paths(
    nav: pd.Series, n_sims: int, days: int, distrib: str = "normal"
) -> pd.DataFrame:
    """Simulated forward paths for a single series, via geometric Brownian motion."""
    if len(nav) < 2:
        raise RiskError("need at least two observations to simulate")

    returns = nav.pct_change().dropna()
    factor = _annualization_factor(nav.index)
    mu_annual = float(returns.mean()) * factor
    sigma_annual = float(returns.std()) * np.sqrt(factor)
    paths = pa.gbm_multiple_path(
        n_sims, float(nav.iloc[-1]), mu_annual, sigma_annual,
        days, distrib,
    )
    # gbm_multiple_path returns (days + 1, n_sims): one extra row for the
    # starting value, paths in columns. Verified against the installed package.
    frame = pd.DataFrame(np.asarray(paths))
    frame.columns = [f"path_{i}" for i in range(frame.shape[1])]
    return frame


def simulate_portfolio_paths(
    prices: pd.DataFrame,
    weights: dict[str, float] | None,
    n_sims: int,
    days: int,
    distrib: str = "normal",
) -> pd.DataFrame:
    """Correlated multi-asset simulation via a Cholesky factorization."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to simulate")

    vec_w = normalize_weights(prices.columns, weights)
    returns = prices.pct_change().dropna()
    start = float(prices.mul(pd.Series(vec_w, index=prices.columns)).sum(axis=1).iloc[-1])

    # portfolio_path_cholesky also assumes annualized inputs (it scales by
    # dt = 1/252 internally, same as gbm_multiple_path): both mu and the
    # covariance matrix must be annualized, not left at per-period scale.
    factor = _annualization_factor(prices.index)
    mu_annual = list(returns.mean() * factor)
    cov_annual = returns.cov() * factor

    paths = pa.portfolio_path_cholesky(
        n_sims, start, vec_w, mu_annual, cov_annual, days, distrib
    )
    # portfolio_path_cholesky also returns (days + 1, n_sims).
    frame = pd.DataFrame(np.asarray(paths))
    frame.columns = [f"path_{i}" for i in range(frame.shape[1])]
    return frame


def stressed_covariance(prices: pd.DataFrame, stress_factor: float = 0.5) -> pd.DataFrame:
    """Covariance with correlations pushed toward 1 - the panic scenario."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to estimate covariance")

    cov = prices.pct_change().dropna().cov()
    stressed = get_stressed_covariance(cov, stress_factor=stress_factor)
    return pd.DataFrame(np.asarray(stressed), index=cov.index, columns=cov.columns)


def covariance_var(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    alpha: float = 0.05,
    distrib: str = "normal",
    cov: pd.DataFrame | None = None,
) -> float:
    """VaR from a covariance matrix and weights alone - no return history needed.

    Pass `cov` to price the same weights under a stressed matrix, which is what
    makes the stress scenario comparable against the base case.
    """
    if cov is None:
        if len(prices) < 2:
            raise RiskError("need at least two observations to estimate covariance")
        cov = prices.pct_change().dropna().cov()

    vec_w = normalize_weights(prices.columns, weights)
    return float(
        covariance_parametric_var(vec_w, np.asarray(cov), alpha=alpha, distrib=distrib)
    )
