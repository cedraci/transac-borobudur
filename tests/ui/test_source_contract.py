"""One contract every source must satisfy for each capability it declares.

This is what lets the rest of the app treat sources interchangeably.
"""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, PriceSource
from portfolio_ui.sources.local_source import LocalSource
from tests.ui.test_async_eod_source import FakeAsyncEodClient
from tests.ui.test_eod_api_source import FakeEodApiClient
from tests.ui.test_market_access_source import FakeMarketAccessClient

from portfolio_ui.sources.async_eod_source import AsyncEodSource
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource

START = dt.date(2024, 1, 1)
END = dt.date(2024, 1, 31)
TICKERS = ["AAPL.US", "MSFT.US"]


def _local_frame():
    return pd.DataFrame(
        {t: [100.0, 101.0, 102.0] for t in TICKERS},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


ALL_SOURCES = [
    pytest.param(
        lambda: EodApiSource(token="k", client=FakeEodApiClient()), id="eod_api"
    ),
    pytest.param(
        lambda: AsyncEodSource(token="k", client=FakeAsyncEodClient()), id="async_eod"
    ),
    pytest.param(
        lambda: MarketAccessSource(client=FakeMarketAccessClient()), id="market_access"
    ),
    pytest.param(
        lambda: LocalSource(_local_frame(), label="local"), id="local"
    ),
]


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_source_satisfies_the_protocol(factory):
    source = factory()
    assert isinstance(source, PriceSource)
    assert isinstance(source.name, str) and source.name
    assert isinstance(source.capabilities, frozenset)


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_price_history_contract(factory):
    source = factory()
    if not source.supports(Capability.PRICE_HISTORY):
        pytest.skip(f"{source.name} does not support price history")

    tickers = ["AAPL", "MSFT"] if source.name == "market_access" else TICKERS
    out = source.price_history(tickers, START, END)

    assert isinstance(out, pd.DataFrame)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert out.index.is_monotonic_increasing
    assert list(out.columns) == tickers
    assert all(d == "float64" for d in out.dtypes)
    assert not out.index.has_duplicates


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_close_at_contract(factory):
    source = factory()
    if not source.supports(Capability.CLOSE_AT):
        pytest.skip(f"{source.name} does not support close_at")

    out = source.close_at(TICKERS, dt.date(2024, 1, 3))
    assert isinstance(out, pd.Series)
    assert out.dtype == "float64"
    assert set(out.index).issubset(set(TICKERS))


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_latest_contract(factory):
    source = factory()
    if not source.supports(Capability.LATEST):
        pytest.skip(f"{source.name} does not support latest")

    out = source.latest(TICKERS)
    assert isinstance(out, pd.Series)
    assert out.dtype == "float64"
    assert set(out.index).issubset(set(TICKERS))


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_sovereign_yields_contract(factory):
    source = factory()
    if not source.supports(Capability.SOVEREIGN):
        pytest.skip(f"{source.name} does not support sovereign yields")

    out = source.sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["Rates"]
    assert all(d == "float64" for d in out.dtypes)
