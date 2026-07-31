"""Portfolio optimization as plain functions: prices in, weights out.

No streamlit import - the Optimization page renders what this returns. Every
function takes and returns pandas objects so it is testable without a
Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_construction import portfolio_optimization as po

# The eight objectives portfolio_optimization implements, with a short gloss
# each so the page can explain them without duplicating knowledge.
OBJECTIVES: dict[str, str] = {
    "minimum_variance": "Lowest possible portfolio volatility.",
    "most_diversified": "Maximises the diversification ratio.",
    "maximum_sharpe": "Best return per unit of risk.",
    "minimum_VaR": "Minimises Value-at-Risk rather than variance.",
    "equal_risk_contribution": "Risk parity - every asset contributes equally to risk.",
    "maximum_return": "Chases return with no regard for risk.",
    "inverse_volatility": "Weights inversely proportional to each asset's volatility.",
    "mean_variance": "Lowest volatility that still hits a target return.",
}


class OptimizeError(RuntimeError):
    """An input cannot support the requested optimization."""


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or len(prices) < 2:
        raise OptimizeError("need at least two observations to optimize")
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise OptimizeError("no usable returns in this price history")
    return returns


def optimize(
    prices: pd.DataFrame,
    objective: str,
    bounds: tuple[float, float] | None = None,
    cov_mat: str = "sample",
    target_return: float = 0.0,
) -> pd.Series:
    """Optimal weights for one objective, indexed by ticker.

    `bounds` is a single (low, high) pair applied to every asset; the core
    function wants a per-asset list, which is a detail the page shouldn't carry.
    """
    if objective not in OBJECTIVES:
        raise OptimizeError(f"unknown objective '{objective}'")

    returns = _returns(prices)

    per_asset_bounds = None
    if bounds is not None:
        low, high = bounds
        if low > high:
            raise OptimizeError(f"lower bound {low} exceeds upper bound {high}")
        per_asset_bounds = [(low, high) for _ in returns.columns]

    try:
        weights = po.portfolio_optimization(
            returns,
            objective,
            bounds=per_asset_bounds,
            cov_mat=cov_mat,
            target_return=target_return,
        )
    except Exception as exc:  # solver / linear-algebra failures
        raise OptimizeError(f"{objective} failed: {exc}") from exc

    return pd.Series(weights, dtype="float64").reindex(returns.columns)


def risk_contributions(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Each asset's share of total portfolio risk, summing to 1.

    The core function returns absolute contributions, which sum to the
    portfolio volatility; shares are easier to read on a page.
    """
    returns = _returns(prices)
    aligned = weights.reindex(returns.columns).fillna(0.0).to_numpy()

    cov = np.cov(returns, rowvar=False)
    contributions = np.asarray(po.risk_contribution(aligned, cov), dtype="float64")

    total = contributions.sum()
    if not np.isfinite(total) or total <= 0:
        raise OptimizeError("risk contributions are degenerate for these weights")

    return pd.Series(contributions / total, index=returns.columns, dtype="float64")


def correlation_matrix(prices: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """Correlations between assets, by the sample or Gerber estimator."""
    returns = _returns(prices)

    if method == "gerber":
        raw = po.gerber_correlation_matrix(returns, 0.5, True)
        return pd.DataFrame(
            np.asarray(raw, dtype="float64"),
            index=returns.columns,
            columns=returns.columns,
        )
    if method == "sample":
        return returns.corr()

    raise OptimizeError(f"unknown correlation method '{method}'")


def bayes_stein_returns(prices: pd.DataFrame) -> pd.Series:
    """Expected returns shrunk toward the cross-sectional average.

    Reduces estimation error - raw sample means are famously noisy inputs to an
    optimizer.
    """
    returns = _returns(prices)
    shrunk = po.estimate_bayes_stein(returns.to_numpy())
    return pd.Series(
        np.asarray(shrunk, dtype="float64").ravel(),
        index=returns.columns,
        dtype="float64",
    )


@dataclass(frozen=True)
class CapitalProtectionResult:
    """Weights plus the solver diagnostics the page must surface.

    The solver frequently fails to converge on real data, and silently showing
    the weights from a failed solve would be worse than saying so.
    """

    weights: pd.Series
    success: bool
    message: str
    expected_return_annual: float
    volatility_annual: float
    worst_case_total_return: float


def capital_protection(
    prices: pd.DataFrame, duration: int, confidence: float = 0.95
) -> CapitalProtectionResult:
    """Max-return weights that keep a `confidence` chance of not losing money."""
    returns = _returns(prices)
    cov = np.cov(returns, rowvar=False)

    try:
        result = po.optimize_capital_protection(
            returns.mean().to_numpy(), cov, duration, confidence=confidence
        )
    except Exception as exc:
        raise OptimizeError(f"capital protection failed: {exc}") from exc

    weights = pd.Series(
        np.asarray(result["weights"], dtype="float64").ravel(),
        index=returns.columns,
        dtype="float64",
    )
    return CapitalProtectionResult(
        weights=weights,
        success=bool(result.get("success", False)),
        message=str(result.get("message", "")),
        expected_return_annual=float(result.get("expected_return_annual", float("nan"))),
        volatility_annual=float(result.get("volatility_annual", float("nan"))),
        worst_case_total_return=float(result.get("worst_case_total_return", float("nan"))),
    )


def efficient_frontier(prices: pd.DataFrame, points: int = 20) -> pd.DataFrame:
    """Return and volatility of minimum-variance portfolios across a return range.

    Built by sweeping `mean_variance` between the minimum-variance portfolio's
    return and the best single asset's, annualised so the numbers are readable.
    """
    if points < 2:
        raise OptimizeError("an efficient frontier needs at least two points")

    returns = _returns(prices)
    factor = 252.0

    lowest = optimize(prices, "minimum_variance")
    low_return = float(returns.mul(lowest).sum(axis=1).mean() * factor)
    high_return = float(returns.mean().max() * factor)

    if high_return <= low_return:
        raise OptimizeError("no return range to sweep - all assets look identical")

    rows = []
    for target in np.linspace(low_return, high_return, points):
        try:
            weights = optimize(
                prices, "mean_variance", cov_mat="sample", target_return=float(target)
            )
        except OptimizeError:
            continue

        portfolio = returns.mul(weights.fillna(0.0)).sum(axis=1)
        rows.append(
            {
                "Return": float(portfolio.mean() * factor),
                "Volatility": float(portfolio.std() * np.sqrt(factor)),
            }
        )

    if len(rows) < 2:
        raise OptimizeError("the optimizer could not solve enough points to plot")

    frame = pd.DataFrame(rows).sort_values("Return").reset_index(drop=True)
    return frame
