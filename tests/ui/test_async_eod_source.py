"""AsyncEodSource flattens the parallel client into the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.async_eod_source import CLIENT_SIDE_SLICE_NOTE, AsyncEodSource
from portfolio_ui.sources.base import Capability, SourceUnavailable, TickerNotFound


class FakeAsyncEodClient:
    """Stands in for portfolio_construction.async_eod."""

    def __init__(self):
        self.calls = []

    def get_full_history(self, tickers):
        self.calls.append(("get_full_history", tuple(tickers)))
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        out = []
        for ticker in tickers:
            if ticker == "MISSING.US":
                out.append(None)
            else:
                out.append(pd.Series([100.0, 101.0, 102.0], index=index, name=ticker))
        return out

    def get_historical(self, tickers, date):
        self.calls.append(("get_historical", tuple(tickers), date))
        return [
            None if t == "MISSING.US" else {"code": t, "close": 123.75} for t in tickers
        ]

    def get_realtime(self, tickers):
        self.calls.append(("get_realtime", tuple(tickers)))
        return [{"code": t, "close": 55.5} for t in tickers]

    def sovereign_bonds(self, countries, tenors, strDate):
        self.calls.append(("sovereign_bonds", tuple(countries), tuple(tenors), strDate))
        tickers = [f"{c}{t}Y.GBOND" for c in countries for t in tenors]
        return pd.DataFrame({"Rates": [4.25] * len(tickers)}, index=tickers)


def _source():
    return AsyncEodSource(token="unit-test-key", client=FakeAsyncEodClient())


def test_declares_all_four_capabilities():
    assert _source().capabilities == frozenset(Capability)


def test_is_unavailable_without_a_token():
    source = AsyncEodSource(token="", client=FakeAsyncEodClient())
    assert not source.is_available()
    assert "EOD_API_KEY" in source.unavailable_reason()


def test_price_history_raises_when_token_missing():
    source = AsyncEodSource(token="", client=FakeAsyncEodClient())
    with pytest.raises(SourceUnavailable):
        source.price_history(["AAPL.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_price_history_concatenates_series_into_one_frame():
    out = _source().price_history(
        ["AAPL.US", "MSFT.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US", "MSFT.US"]
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_slices_client_side():
    out = _source().price_history(
        ["AAPL.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 3)
    )
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03")


def test_price_history_drops_tickers_with_no_history():
    out = _source().price_history(
        ["AAPL.US", "MISSING.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US"]


def test_price_history_raises_when_nothing_came_back():
    with pytest.raises(TickerNotFound):
        _source().price_history(
            ["MISSING.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
        )


def test_client_side_slice_note_is_exposed():
    assert "start" in CLIENT_SIDE_SLICE_NOTE and "async_eod" in CLIENT_SIDE_SLICE_NOTE


def test_close_at_flattens_code_close_dicts():
    out = _source().close_at(["AAPL.US", "MSFT.US"], dt.date(2024, 1, 3))
    assert out["AAPL.US"] == 123.75
    assert out.dtype == "float64"


def test_close_at_skips_none_results():
    out = _source().close_at(["AAPL.US", "MISSING.US"], dt.date(2024, 1, 3))
    assert list(out.index) == ["AAPL.US"]


def test_latest_flattens_code_close_dicts():
    out = _source().latest(["AAPL.US", "MSFT.US"])
    assert out["MSFT.US"] == 55.5


def test_sovereign_yields_returns_rates_frame():
    out = _source().sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert list(out.columns) == ["Rates"]
    assert out.loc["US10Y.GBOND", "Rates"] == 4.25
