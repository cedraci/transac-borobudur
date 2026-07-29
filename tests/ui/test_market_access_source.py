"""MarketAccessSource exposes yfinance history under the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, CapabilityNotSupported, TickerNotFound
from portfolio_ui.sources.market_access_source import MarketAccessSource


class FakeMarketAccessClient:
    def yahooFinance_historical_data(self, ticker):
        if ticker == "MISSING":
            return pd.DataFrame()
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        return pd.DataFrame(
            {
                "Open": [99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 102.0],
                "Adj Close": [100.0, 101.0, 102.0],
            },
            index=index,
        )


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
