"""EodApiSource adapts the synchronous client to the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, SourceUnavailable, TickerNotFound
from portfolio_ui.sources.eod_api_source import EodApiSource


class FakeEodApiClient:
    """Stands in for portfolio_construction.eod_api."""

    def __init__(self):
        self.calls = []

    def adjusted_prices(self, tok, tickers):
        self.calls.append(("adjusted_prices", tok, tuple(tickers)))
        return pd.DataFrame(
            {t: [100.0, 101.0, 102.0] for t in tickers},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

    def recursive_adjClose_atDate(self, token, ticker, strDate):
        self.calls.append(("recursive_adjClose_atDate", token, ticker, strDate))
        if ticker == "MISSING.US":
            raise IndexError("no data")
        return 123.75

    def last_prices_universe(self, token, tickers):
        self.calls.append(("last_prices_universe", token, tuple(tickers)))
        return pd.Series({t: 55.5 for t in tickers}, dtype="float64")

    def get_SovBond(self, token, ticker, strDate):
        self.calls.append(("get_SovBond", token, ticker, strDate))
        return 4.25


def _source():
    return EodApiSource(token="unit-test-key", client=FakeEodApiClient())


def test_declares_all_four_capabilities():
    assert _source().capabilities == frozenset(Capability)


def test_is_unavailable_without_a_token():
    source = EodApiSource(token="", client=FakeEodApiClient())
    assert not source.is_available()
    assert "EOD_API_KEY" in source.unavailable_reason()


def test_price_history_raises_when_token_missing():
    source = EodApiSource(token="", client=FakeEodApiClient())
    with pytest.raises(SourceUnavailable):
        source.price_history(["AAPL.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_price_history_returns_normalized_frame():
    out = _source().price_history(
        ["AAPL.US", "MSFT.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US", "MSFT.US"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_slices_to_requested_range():
    out = _source().price_history(
        ["AAPL.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 3)
    )
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03")


def test_close_at_returns_series_by_ticker():
    out = _source().close_at(["AAPL.US", "MSFT.US"], dt.date(2024, 1, 3))
    assert out["AAPL.US"] == 123.75
    assert out.dtype == "float64"


def test_close_at_raises_when_every_ticker_is_missing():
    with pytest.raises(TickerNotFound):
        _source().close_at(["MISSING.US"], dt.date(2024, 1, 3))


def test_close_at_keeps_survivors_on_partial_failure():
    out = _source().close_at(["AAPL.US", "MISSING.US"], dt.date(2024, 1, 3))
    assert list(out.index) == ["AAPL.US"]


def test_latest_returns_series_by_ticker():
    out = _source().latest(["AAPL.US", "MSFT.US"])
    assert out["MSFT.US"] == 55.5


def test_sovereign_yields_builds_bond_tickers():
    out = _source().sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert list(out.columns) == ["Rates"]
    assert list(out.index) == [
        "US5Y.GBOND",
        "US10Y.GBOND",
        "FR5Y.GBOND",
        "FR10Y.GBOND",
    ]
    assert out.loc["US5Y.GBOND", "Rates"] == 4.25
