"""Turn an active dataset into display-ready analysis frames.

No streamlit import - the views render what this returns. Every function takes
plain pandas objects so it can be tested without a Streamlit runtime.
"""

from __future__ import annotations

import pandas as pd

from portfolio_construction import portfolio_analysis as pa


class AnalyticsError(RuntimeError):
    """An input cannot support the requested statistic."""


def _require_two_rows(frame) -> None:
    if len(frame) < 2:
        raise AnalyticsError("need at least two observations to compute this")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period returns, first (all-NaN) row dropped."""
    _require_two_rows(prices)
    return prices.pct_change().dropna(how="all")


def weighted_nav(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """One equity curve for the whole basket, rebased to 100 at the start.

    Weights are fixed (buy-and-hold), not rebalanced - the Backtest page is
    where rebalancing lives.
    """
    _require_two_rows(prices)

    if weights is None:
        columns = list(prices.columns)
        weights = {c: 1.0 / len(columns) for c in columns}

    unknown = [t for t in weights if t not in prices.columns]
    if unknown:
        raise AnalyticsError(f"not in the dataset: {', '.join(unknown)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise AnalyticsError(f"weights must sum to 1.0, got {total:.4f}")

    used = prices[list(weights)]

    # Align to the common history BEFORE rebasing. A ticker that lists later has
    # a NaN first price; dividing by it turns its ENTIRE rebased column to NaN,
    # and .sum(axis=1) then skips it silently - so the asset vanished from every
    # row and the NAV started below 100 instead of at it.
    complete = used.dropna(how="any")
    if complete.empty:
        raise AnalyticsError(
            f"the histories of {', '.join(used.columns)} do not overlap - "
            "no date has data for all of them"
        )

    used = used.loc[complete.index[0] :].ffill()
    if len(used) < 2:
        raise AnalyticsError(
            "fewer than two dates where every selected ticker has data"
        )

    rebased = used.div(used.iloc[0])
    nav = rebased.mul(pd.Series(weights)).sum(axis=1) * 100.0
    nav.name = "Portfolio"
    return nav


def performance_table(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """The headline statistics, one row per column of `prices`."""
    _require_two_rows(prices)
    try:
        return pa.stats_report(prices, rf=rf)
    except (ValueError, IndexError) as exc:
        # stats_report needs complete calendar periods; a short fetch cannot
        # supply them. The page renders AnalyticsError, so translate rather
        # than letting a raw IndexError reach the browser.
        raise AnalyticsError(f"not enough history for performance statistics: {exc}") from exc


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drop from the running peak, as a negative fraction.

    historical_drawdown already returns a Series on the same index, so this
    only renames it - re-wrapping it in pd.Series(..., index=...) would risk
    silently reindexing to NaN.
    """
    _require_two_rows(nav)
    return pa.historical_drawdown(nav).rename("Drawdown")


def monthly_returns_table(nav: pd.Series) -> pd.DataFrame:
    """Month-end returns, one row per month."""
    _require_two_rows(nav)
    return pa.monthly_returns(nav)


def rebased_prices(prices: pd.DataFrame, rebased_dt) -> pd.DataFrame:
    """Every series rebased to 100 at a chosen date.

    rebased_from_date returns None (after printing) when the date is absent
    from the index, so translate that into an AnalyticsError the page can show.
    """
    _require_two_rows(prices)
    out = pa.rebased_from_date(prices, pd.Timestamp(rebased_dt))
    if out is None:
        raise AnalyticsError(
            f"{pd.Timestamp(rebased_dt):%Y-%m-%d} is not a date in this dataset"
        )
    return out


def drawdown_episodes(
    nav: pd.Series, top_n: int = 10, min_drawdown: float = 0.01
) -> pd.DataFrame:
    """The worst peak-to-trough episodes with their recovery dates.

    drawdowns_table takes a SERIES despite its type hint saying DataFrame -
    it relies on .items() yielding (timestamp, price). compute_drawdown_periods
    itself wants list[tuple[date, Decimal]] and is not UI-friendly.
    """
    _require_two_rows(nav)
    return pa.drawdowns_table(nav, top_n=top_n, min_drawdown=min_drawdown)


def calendar_table(nav: pd.Series) -> pd.DataFrame:
    """Performance broken down by calendar period."""
    _require_two_rows(nav)
    return pa.calendar_performances(nav)


def rolling_cagr(nav: pd.Series, max_holding_period: int = 20) -> pd.DataFrame:
    """CAGR over every rolling holding period, up to max_holding_period years."""
    _require_two_rows(nav)
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    return pa.cagr_rolled(yearly, max_holding_period=max_holding_period)
