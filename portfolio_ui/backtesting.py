"""Rebalanced backtests as plain functions.

No streamlit import - the Backtest page renders what this returns. The core
`Backtest` class carries a lot of mutable state; this wraps one run into an
immutable result the page can hold in session state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_construction import portfolio_backtest as pb
from portfolio_ui.optimize import OBJECTIVES

# The frequencies RebalancingCalendar actually implements. Anything else falls
# back to year-end, which would silently ignore the user's choice.
REBALANCING_METHODS: dict[str, str] = {
    "eom": "End of month",
    "eoq": "End of quarter",
    "eos": "End of semester",
    "eow": "End of week",
    "bim": "Every other week",
}


class BacktestError(RuntimeError):
    """A backtest cannot be run as configured."""


@dataclass(frozen=True)
class BacktestResult:
    """One completed run, in the shapes the page and the Analysis page want."""

    equity_curve: pd.Series
    weights: pd.DataFrame
    rebalancing_dates: tuple[pd.Timestamp, ...]
    objective: str
    method: str
    start: str
    end: str
    lookback: int
    robust: bool
    stock_picking: bool
    duration_seconds: float

    def summary(self) -> str:
        picking = ", momentum picking" if self.stock_picking else ""
        robust = ", shrunk covariance" if self.robust else ""
        return (
            f"{self.objective} rebalanced {self.method} from {self.start} to "
            f"{self.end}{robust}{picking}"
        )


def momentum_scores(
    prices: pd.DataFrame, current_date, window: int = 250
) -> pd.Series:
    """Each ticker's momentum score at a date, best first."""
    if window >= len(prices):
        raise BacktestError(
            f"a {window}-observation window needs more than {len(prices)} rows of history"
        )

    date = pb.snap_to_trading_day(prices.index, current_date, "backward")
    position = int(np.where(prices.index == date)[0][0])
    if position < window:
        raise BacktestError(
            f"only {position} observations before {date:%Y-%m-%d}, need {window}"
        )

    try:
        _, names, scores = pb.universe_selection(
            prices, date, window, len(prices.columns)
        )
    except Exception as exc:
        raise BacktestError(f"momentum scoring failed: {exc}") from exc

    return pd.Series(
        np.asarray(scores, dtype="float64").ravel(), index=list(names)
    ).sort_values(ascending=False)


def run_backtest(
    prices: pd.DataFrame,
    objective: str,
    start: str,
    end: str,
    lookback: int = 250,
    method: str = "eom",
    robust: bool = False,
    stock_picking: bool = False,
    nb_securities: int | None = None,
) -> BacktestResult:
    """Run one rebalanced backtest and return its curve and weight history.

    `nb_securities` only applies when `stock_picking` is on. It is capped at
    the universe size by the core selector.
    """
    if objective not in OBJECTIVES:
        raise BacktestError(f"unknown objective '{objective}'")
    if method not in REBALANCING_METHODS:
        raise BacktestError(
            f"unknown rebalancing method '{method}' - "
            f"choose from {', '.join(sorted(REBALANCING_METHODS))}"
        )
    if prices is None or len(prices) < 2:
        raise BacktestError("need at least two observations to backtest")

    backtest = pb.Backtest()
    try:
        backtest.initialize_parameters(
            prices, str(start), str(end), lookback, method=method
        )
    except ValueError as exc:
        raise BacktestError(str(exc)) from exc
    except Exception as exc:
        raise BacktestError(f"could not set up the backtest: {exc}") from exc

    try:
        backtest.simulations(
            objective,
            robust=robust,
            stock_picking=stock_picking,
            nb_securities=int(nb_securities) if nb_securities else None,
        )
    except Exception as exc:
        raise BacktestError(f"the backtest failed: {exc}") from exc

    curve = backtest.strat["Strategy"].astype("float64")
    curve = curve.div(curve.iloc[0]).mul(100.0)
    curve.name = "Strategy"
    curve.index.name = "Date"

    weights = pd.DataFrame(
        [entry["weights"] for entry in backtest.historical_portfolios],
        index=[entry["date"] for entry in backtest.historical_portfolios],
    ).astype("float64")
    weights.index.name = "Date"

    return BacktestResult(
        equity_curve=curve,
        weights=weights,
        rebalancing_dates=tuple(backtest.sequence_rebal),
        objective=objective,
        method=method,
        start=backtest.start_date,
        end=backtest.end_date,
        lookback=lookback,
        robust=robust,
        stock_picking=stock_picking,
        duration_seconds=float(backtest.backtest_duration.total_seconds()),
    )
