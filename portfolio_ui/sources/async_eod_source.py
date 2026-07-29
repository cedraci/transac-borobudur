"""Adapter for the parallel portfolio_construction.async_eod client."""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    SourceUnavailable,
    TickerNotFound,
    UpstreamError,
    normalize_price_frame,
    normalize_price_series,
)

CLIENT_SIDE_SLICE_NOTE = (
    "async_eod always requests 1990-01-01 to today and ignores start/end; "
    "the range was applied client-side"
)


class AsyncEodSource(BaseSource):
    """The parallel client: much faster for many tickers, fewer endpoints."""

    name = "async_eod"
    capabilities = frozenset(Capability)

    def __init__(self, token: str | None = None, client=None):
        if client is None:
            from portfolio_construction import async_eod

            client = async_eod
        self._client = client
        self._token = token if token is not None else os.environ.get("EOD_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._token)

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "EOD_API_KEY is not set"

    def _require_token(self) -> str:
        if not self._token:
            raise SourceUnavailable("EOD_API_KEY is not set")
        return self._token

    @staticmethod
    def _flatten_code_close(entries) -> dict[str, float]:
        values: dict[str, float] = {}
        for entry in entries or []:
            if not entry:
                continue
            values[entry["code"]] = entry["close"]
        return values

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        self._require_token()
        tickers = list(tickers)
        try:
            series_list = self._client.get_full_history(tickers)
        except Exception as exc:
            raise UpstreamError(f"async_eod price history failed: {exc}") from exc

        usable = [s for s in series_list if s is not None and len(s)]
        if not usable:
            raise TickerNotFound(f"no price history for {tickers}")

        raw = pd.concat(usable, axis=1)
        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"no price history for {tickers} between {start} and {end}"
            )
        return frame

    def close_at(self, tickers, on: dt.date) -> pd.Series:
        self._require_token()
        tickers = list(tickers)
        try:
            entries = self._client.get_historical(tickers, pd.Timestamp(on).to_pydatetime())
        except Exception as exc:
            raise UpstreamError(f"async_eod close_at failed: {exc}") from exc

        values = self._flatten_code_close(entries)
        if not values:
            raise TickerNotFound(f"no close on {on} for {tickers}")
        return normalize_price_series(values, tickers)

    def latest(self, tickers) -> pd.Series:
        self._require_token()
        tickers = list(tickers)
        try:
            entries = self._client.get_realtime(tickers)
        except Exception as exc:
            raise UpstreamError(f"async_eod latest failed: {exc}") from exc

        values = self._flatten_code_close(entries)
        if not values:
            raise TickerNotFound(f"no latest price for {tickers}")
        return normalize_price_series(values, tickers)

    def sovereign_yields(self, countries, tenors, on: dt.date) -> pd.DataFrame:
        self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        try:
            frame = self._client.sovereign_bonds(list(countries), list(tenors), str_date)
        except Exception as exc:
            raise UpstreamError(f"async_eod sovereign yields failed: {exc}") from exc

        if frame is None or frame.empty:
            raise TickerNotFound(f"no sovereign yields on {str_date}")
        return frame.astype("float64")
