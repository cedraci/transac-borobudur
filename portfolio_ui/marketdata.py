"""The eod_api-only surface: fundamentals, corporate actions, calendars, macro.

No streamlit import - the Market Data page renders what this returns. The
client is injected so the whole module is testable without touching the
network or EOD_API_KEY.

Every method normalizes the client's output (dates to a DatetimeIndex, blank
responses to a clear error) and converts upstream failures into
MarketDataError, so the page never sees a raw traceback.
"""

from __future__ import annotations

import os

import pandas as pd


class MarketDataError(RuntimeError):
    """A market-data request could not be served."""


def _to_dated_frame(raw, date_column: str, what: str, ticker: str) -> pd.DataFrame:
    """Index a client frame by its date column, sorted ascending."""
    frame = pd.DataFrame(raw)
    if frame.empty:
        raise MarketDataError(f"no {what} returned for {ticker}")

    if date_column in frame.columns:
        frame = frame.copy()
        frame.index = pd.to_datetime(frame[date_column])
        frame = frame.drop(columns=[date_column])
        frame.index.name = "Date"

    return frame.sort_index()


class MarketDataService:
    """Everything only the synchronous EOD client can provide."""

    def __init__(self, token: str | None = None, client=None):
        if client is None:
            from portfolio_construction import eod_api

            client = eod_api
        self._client = client
        self._token = token if token is not None else os.environ.get("EOD_API_KEY", "")

    # -- availability ----------------------------------------------------

    def is_available(self) -> bool:
        return bool(self._token)

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "EOD_API_KEY is not set"

    def _require_token(self) -> str:
        if not self._token:
            raise MarketDataError("EOD_API_KEY is not set")
        return self._token

    def _call(self, what: str, fn, *args, **kwargs):
        """Run an upstream call, turning any failure into MarketDataError."""
        try:
            return fn(*args, **kwargs)
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(f"{what} failed: {exc}") from exc

    # -- discovery -------------------------------------------------------

    def search(self, query: str) -> pd.DataFrame:
        """Find tickers by name or code."""
        token = self._require_token()
        if not query or not query.strip():
            raise MarketDataError("the search query is empty")

        raw = self._call("search", self._client.search_query, token, query.strip())
        frame = pd.DataFrame(raw)
        if frame.empty:
            raise MarketDataError(f"nothing found for '{query}'")

        if {"Code", "Exchange"} <= set(frame.columns):
            # the form every other endpoint expects
            frame.insert(0, "Ticker", frame["Code"] + "." + frame["Exchange"])
        return frame

    def fundamentals(self, ticker: str, section: str | None = None) -> pd.DataFrame:
        """Company fundamentals, optionally one top-level section.

        The raw payload is deeply nested; a section is flattened to rows so a
        page can show it as a table.
        """
        token = self._require_token()

        kwargs = {"filter": section} if section else {}
        raw = self._call(
            "fundamentals", self._client.fundamentals, ticker, token, **kwargs
        )
        if not raw:
            raise MarketDataError(f"no fundamentals returned for {ticker}")

        payload = raw
        if section and isinstance(raw, dict) and section in raw:
            payload = raw[section]

        if isinstance(payload, dict) and all(
            not isinstance(v, (dict, list)) for v in payload.values()
        ):
            return pd.DataFrame({"Value": payload})

        # still nested: show what the top level offers rather than guessing
        return pd.DataFrame({"Value": {k: str(v) for k, v in dict(payload).items()}})

    def index_constituents(self, index_ticker: str) -> list[str]:
        """The tickers making up an index, e.g. GSPC.INDX."""
        token = self._require_token()
        tickers = self._call(
            "index constituents", self._client.index_constituents,
            token, index_ticker, True,
        )
        if not tickers:
            raise MarketDataError(f"no constituents returned for {index_ticker}")
        return list(tickers)

    # -- prices and corporate actions ------------------------------------

    def ohlcv(self, ticker: str) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call("ohlcv", self._client.ohlcv, token, ticker)
        return _to_dated_frame(raw, "date", "price history", ticker)

    def dividends(self, ticker: str) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call("dividends", self._client.dividends, token, ticker)
        return _to_dated_frame(raw, "date", "dividends", ticker)

    def splits(self, ticker: str) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call("splits", self._client.splits, token, ticker)
        return _to_dated_frame(raw, "date", "splits", ticker)

    def dividend_yield_history(self, ticker: str) -> pd.Series:
        """Dividend yield by calendar year."""
        token = self._require_token()
        raw = self._call(
            "dividend yield", self._client.stock_historical_dividend_yield, token, ticker
        )
        series = pd.Series(raw)
        if series.empty:
            raise MarketDataError(f"no dividend history for {ticker}")
        return series

    # -- calendars and macro ---------------------------------------------

    def earnings_calendar(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call(
            "earnings calendar", self._client.earnings_calendar, token, start, end
        )
        frame = pd.DataFrame(raw)
        if frame.empty:
            raise MarketDataError("no earnings reported in that window")
        return frame

    def macro_events(self, start: str, end: str, country: str) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call(
            "macro events", self._client.macro_events, token, start, end, country
        )
        frame = pd.DataFrame(raw)
        if frame.empty:
            raise MarketDataError(f"no macro events for {country} in that window")
        return frame

    def macro_indicators(self, country_code: str, indicator: str) -> pd.DataFrame:
        token = self._require_token()
        raw = self._call(
            "macro indicator", self._client.macro_indicators, token, country_code, indicator
        )
        frame = pd.DataFrame(raw)
        if frame.empty:
            raise MarketDataError(f"no data for {indicator} in {country_code}")
        return frame

    def fixed_income_etf(self, ticker: str) -> pd.DataFrame:
        """Duration, maturity, rating and yield for a bond ETF."""
        token = self._require_token()
        raw = self._call(
            "fixed income ETF data", self._client.fixed_income_etf, token, ticker
        )
        frame = pd.DataFrame(raw)
        if frame.empty:
            raise MarketDataError(f"no fixed-income data for {ticker}")
        return frame
