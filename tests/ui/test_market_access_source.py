"""MarketAccessSource exposes yfinance history under the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, CapabilityNotSupported, TickerNotFound
from portfolio_ui.sources.market_access_source import MarketAccessSource


DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]

# Real yfinance daily history carries an exchange-local timezone on the index,
# and "Adj Close" differs from "Close" whenever there was a split or dividend.
# The fake reproduces both so neither can silently regress again.
ADJ_CLOSE = [95.0, 96.0, 97.0]
CLOSE = [100.0, 101.0, 102.0]


class FakeMarketAccessClient:
    """Shaped like real yfinance output: tz-aware index, Adj Close != Close."""

    def __init__(self, timezones=None, default_timezone="America/New_York", adjusted=True):
        self._timezones = dict(timezones or {})
        self._default_timezone = default_timezone
        self._adjusted = adjusted

    def yahooFinance_historical_data(self, ticker):
        if ticker == "MISSING":
            return pd.DataFrame()
        tz = self._timezones.get(ticker, self._default_timezone)
        index = pd.to_datetime(DATES).tz_localize(tz)
        data = {"Open": [99.0, 100.0, 101.0], "Close": list(CLOSE)}
        if self._adjusted:
            data["Adj Close"] = list(ADJ_CLOSE)
        return pd.DataFrame(data, index=index)


def _source():
    return MarketAccessSource(client=FakeMarketAccessClient())


def test_declares_only_price_history():
    assert _source().capabilities == frozenset({Capability.PRICE_HISTORY})


def test_close_at_is_not_supported():
    with pytest.raises(CapabilityNotSupported, match="market_access"):
        _source().close_at(["AAPL"], dt.date(2024, 1, 3))


def test_latest_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        _source().latest(["AAPL"])


def test_sovereign_yields_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        _source().sovereign_yields(["US"], [10], dt.date(2024, 1, 3))


def test_price_history_builds_one_column_per_ticker():
    out = _source().price_history(
        ["AAPL", "MSFT"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_drops_empty_tickers():
    out = _source().price_history(
        ["AAPL", "MISSING"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL"]


def test_price_history_raises_when_nothing_came_back():
    with pytest.raises(TickerNotFound):
        _source().price_history(["MISSING"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_price_history_prefers_adj_close_over_close():
    out = _source().price_history(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert list(out["AAPL"]) == ADJ_CLOSE


def test_price_history_falls_back_to_close_without_adj_close():
    source = MarketAccessSource(client=FakeMarketAccessClient(adjusted=False))
    out = source.price_history(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert list(out["AAPL"]) == CLOSE


def test_price_history_returns_a_tz_naive_index():
    out = _source().price_history(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert out.index.tz is None
    assert len(out) == len(DATES)


def test_price_history_aligns_tickers_from_different_timezones():
    # A basket spanning exchanges used to assemble Series with different tz
    # values: pandas coerced the union to UTC, doubling the rows and leaving
    # every column half-NaN.
    client = FakeMarketAccessClient(
        timezones={"AAPL": "America/New_York", "BMW.DE": "Europe/Berlin"}
    )
    out = MarketAccessSource(client=client).price_history(
        ["AAPL", "BMW.DE"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.index) == list(pd.to_datetime(DATES))
    assert len(out) == len(DATES)
    assert not out.isna().to_numpy().any()
