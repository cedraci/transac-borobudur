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
    rebased = used.div(used.iloc[0])
    nav = rebased.mul(pd.Series(weights)).sum(axis=1) * 100.0
    nav.name = "Portfolio"
    return nav


def performance_table(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """The headline statistics, one row per column of `prices`.

    pa.stats_report hardcodes cal_perf.index[-1], [-2], [-3], so it raises
    IndexError whenever fewer than three calendar years are present (e.g. 500
    business days from 2020-01-01 only spans 2020-2021). Rebuild the same
    columns from the same building blocks, pulling as many trailing calendar
    years as actually exist instead of always demanding three.
    """
    _require_two_rows(prices)
    rows = []
    for column in prices.columns:
        series = prices[column]
        cal_perf = pa.calendar_performances(series)
        row = {
            "Name": column,
            "Ann. Return": "{:.2%}".format(pa.annualized_return(series)),
            "Ann. Volatility": "{:.2%}".format(pa.annualized_volatility(series)),
            "Ann. Sharpe": "{:.2%}".format(pa.annualized_sharpe_ratio(series, rf=rf)),
            "Value-at-Risk": "{:.2%}".format(pa.historical_var(series, 0.05, 1)),
            "Expected Shortfall (histo)": "{:.2%}".format(
                pa.historical_es(series, 0.05, 1)
            ),
            "Max Drawdown": "{:.2%}".format(pa.maximum_drawdown(series)),
        }
        for i in range(1, min(3, len(cal_perf)) + 1):
            row[cal_perf.index[-i]] = "{:.2%}".format(cal_perf.iloc[-i, -1])
        rows.append(row)
    return pd.DataFrame(rows)


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
