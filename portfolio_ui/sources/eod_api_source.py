"""Adapter for the synchronous portfolio_construction.eod_api client."""

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


def sovereign_ticker(country: str, tenor: int) -> str:
    """EOD's sovereign bond ticker convention, e.g. US10Y.GBOND."""
    return f"{country}{tenor}Y.GBOND"


class EodApiSource(BaseSource):
    """The synchronous client: slower, but the widest feature set."""

    name = "eod_api"
    capabilities = frozenset(Capability)

    def __init__(self, token: str | None = None, client=None):
        if client is None:
            from portfolio_construction import eod_api

            client = eod_api
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

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        token = self._require_token()
        try:
            raw = self._client.adjusted_prices(token, list(tickers))
        except Exception as exc:  # network / parse failures
            raise UpstreamError(f"eod_api price history failed: {exc}") from exc

        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(f"no price history for {list(tickers)}")
        return frame

    def close_at(self, tickers, on: dt.date) -> pd.Series:
        token = self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        values: dict[str, float] = {}
        for ticker in tickers:
            try:
                values[ticker] = self._client.recursive_adjClose_atDate(
                    token, ticker, str_date
                )
            except (IndexError, KeyError):
                continue
            except Exception as exc:
                raise UpstreamError(f"eod_api close_at failed for {ticker}: {exc}") from exc

        if not values:
            raise TickerNotFound(f"no close on {str_date} for {list(tickers)}")
        return normalize_price_series(values, tickers)

    def latest(self, tickers) -> pd.Series:
        token = self._require_token()
        try:
            raw = self._client.last_prices_universe(token, list(tickers))
        except Exception as exc:
            raise UpstreamError(f"eod_api latest failed: {exc}") from exc

        series = normalize_price_series(dict(raw), tickers)
        if series.empty:
            raise TickerNotFound(f"no latest price for {list(tickers)}")
        return series

    def sovereign_yields(self, countries, tenors, on: dt.date) -> pd.DataFrame:
        token = self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        rates: dict[str, float] = {}
        for country in countries:
            for tenor in tenors:
                ticker = sovereign_ticker(country, tenor)
                try:
                    rates[ticker] = self._client.get_SovBond(token, ticker, str_date)
                except (IndexError, KeyError):
                    continue
                except Exception as exc:
                    raise UpstreamError(
                        f"eod_api sovereign yield failed for {ticker}: {exc}"
                    ) from exc

        if not rates:
            raise TickerNotFound(f"no sovereign yields on {str_date}")

        frame = pd.DataFrame.from_dict(rates, orient="index")
        frame.columns = ["Rates"]
        return frame.astype("float64")
