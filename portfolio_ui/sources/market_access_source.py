"""Adapter for the yfinance-backed portfolio_construction.market_access module."""

from __future__ import annotations

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    TickerNotFound,
    UpstreamError,
    normalize_price_frame,
)


class MarketAccessSource(BaseSource):
    """Free fallback path. History only - no point-in-time, realtime or bonds."""

    name = "market_access"
    capabilities = frozenset({Capability.PRICE_HISTORY})

    def __init__(self, client=None):
        if client is None:
            from portfolio_construction import market_access

            client = market_access
        self._client = client

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        tickers = list(tickers)
        columns = {}
        for ticker in tickers:
            try:
                history = self._client.yahooFinance_historical_data(ticker)
            except Exception as exc:
                raise UpstreamError(
                    f"market_access price history failed for {ticker}: {exc}"
                ) from exc

            if history is None or history.empty:
                continue
            column = "Adj Close" if "Adj Close" in history.columns else "Close"
            series = history[column]
            # Each exchange comes back in its own timezone. Strip it here, per
            # series: normalizing only after assembly is too late, because
            # pandas would already have unioned the differing tz-aware indexes
            # into UTC, doubling the rows and half-filling every column.
            if getattr(series.index, "tz", None) is not None:
                series = series.copy()
                series.index = series.index.tz_localize(None)
            columns[ticker] = series

        if not columns:
            raise TickerNotFound(f"no price history for {tickers}")

        raw = pd.DataFrame(columns)
        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"no price history for {tickers} between {start} and {end}"
            )
        return frame
