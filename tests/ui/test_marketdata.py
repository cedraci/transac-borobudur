"""The eod_api-only surface, driven entirely through a fake client.

Every function here is a network call in production, so the client is injected
and no test touches the network or EOD_API_KEY.
"""

import pandas as pd
import pytest

from portfolio_ui.marketdata import MarketDataError, MarketDataService


class FakeEodApiClient:
    """Stands in for portfolio_construction.eod_api."""

    def __init__(self):
        self.calls = []

    def search_query(self, token, research, as_df=False):
        self.calls.append(("search_query", research))
        return [
            {"Code": "AAPL", "Exchange": "US", "Name": "Apple Inc", "Type": "Common Stock"},
            {"Code": "AAPL34", "Exchange": "SA", "Name": "Apple BDR", "Type": "Common Stock"},
        ]

    def fundamentals(self, symbol, token, *args, **kwargs):
        self.calls.append(("fundamentals", symbol, kwargs))
        return {"General": {"Name": "Apple Inc", "Sector": "Technology", "ISIN": "US0378331005"}}

    def index_constituents(self, token, ticker, return_tickers=False):
        self.calls.append(("index_constituents", ticker, return_tickers))
        return ["AAPL.US", "MSFT.US"] if return_tickers else {"Components": {}}

    def ohlcv(self, token, ticker, as_df=True):
        self.calls.append(("ohlcv", ticker))
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0],
                "close": [101.0, 102.0],
                "adjusted_close": [101.0, 102.0],
                "volume": [1000, 1100],
            }
        )

    def dividends(self, token, ticker, as_df=True):
        self.calls.append(("dividends", ticker))
        return pd.DataFrame({"date": ["2024-02-09"], "value": [0.24], "currency": ["USD"]})

    def splits(self, token, ticker, as_df=True):
        self.calls.append(("splits", ticker))
        return pd.DataFrame({"date": ["2020-08-31"], "split": ["4.000000/1.000000"]})

    def earnings_calendar(self, token, start_dt=None, end_dt=None):
        self.calls.append(("earnings_calendar", start_dt, end_dt))
        return pd.DataFrame({"code": ["AAPL.US"], "report_date": ["2024-02-01"]})

    def macro_events(self, tok, start, end, cntry):
        self.calls.append(("macro_events", start, end, cntry))
        return pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-05"]), "type": ["CPI"], "actual": [3.1]}
        ).set_index("date")

    def macro_indicators(self, token, country_code, indicator):
        self.calls.append(("macro_indicators", country_code, indicator))
        return pd.DataFrame({"Date": ["2023-12-31"], "Value": [2.5]})

    def stock_historical_dividend_yield(self, tok, ticker):
        self.calls.append(("dividend_yield", ticker))
        return pd.Series([0.006, 0.005], index=[2022, 2023], name="dividend_yield")

    def fixed_income_etf(self, tok, ticker):
        self.calls.append(("fixed_income_etf", ticker))
        return pd.DataFrame({ticker: [5.2, 3.1]}, index=["Duration", "YtM"])


def _service():
    return MarketDataService(token="unit-test-key", client=FakeEodApiClient())


def test_is_unavailable_without_a_token():
    service = MarketDataService(token="", client=FakeEodApiClient())
    assert not service.is_available()
    assert "EOD_API_KEY" in service.unavailable_reason()


def test_every_call_raises_when_the_token_is_missing():
    service = MarketDataService(token="", client=FakeEodApiClient())
    with pytest.raises(MarketDataError, match="EOD_API_KEY"):
        service.search("apple")


def test_search_returns_a_frame_with_a_usable_ticker_column():
    out = _service().search("apple")
    assert isinstance(out, pd.DataFrame)
    assert "Ticker" in out.columns
    assert out["Ticker"].iloc[0] == "AAPL.US"


def test_search_rejects_a_blank_query():
    with pytest.raises(MarketDataError, match="empty"):
        _service().search("   ")


def test_fundamentals_flattens_a_nested_section_to_rows():
    out = _service().fundamentals("AAPL.US", section="General")
    assert isinstance(out, pd.DataFrame)
    assert "Value" in out.columns
    assert out.loc["Sector", "Value"] == "Technology"


def test_index_constituents_returns_a_ticker_list():
    out = _service().index_constituents("GSPC.INDX")
    assert out == ["AAPL.US", "MSFT.US"]


def test_ohlcv_is_indexed_by_date():
    out = _service().ohlcv("AAPL.US")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "adjusted_close" in out.columns
    assert out.index.is_monotonic_increasing


def test_dividends_are_indexed_by_date():
    out = _service().dividends("AAPL.US")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out["value"].iloc[0] == 0.24


def test_splits_are_indexed_by_date():
    out = _service().splits("AAPL.US")
    assert isinstance(out.index, pd.DatetimeIndex)


def test_earnings_calendar_passes_the_date_range_through():
    client = FakeEodApiClient()
    service = MarketDataService(token="k", client=client)
    service.earnings_calendar("2024-01-01", "2024-03-31")
    assert ("earnings_calendar", "2024-01-01", "2024-03-31") in client.calls


def test_macro_events_returns_a_frame():
    out = _service().macro_events("2024-01-01", "2024-12-31", "US")
    assert isinstance(out, pd.DataFrame)
    assert not out.empty


def test_macro_indicators_returns_a_frame():
    out = _service().macro_indicators("USA", "gdp_current_usd")
    assert isinstance(out, pd.DataFrame)


def test_dividend_yield_history_returns_a_series():
    out = _service().dividend_yield_history("AAPL.US")
    assert isinstance(out, pd.Series)
    assert out.index.tolist() == [2022, 2023]


def test_fixed_income_etf_returns_a_frame():
    out = _service().fixed_income_etf("AGG.US")
    assert isinstance(out, pd.DataFrame)
    assert "Duration" in out.index


def test_upstream_failures_become_market_data_errors():
    class Broken(FakeEodApiClient):
        def ohlcv(self, token, ticker, as_df=True):
            raise ConnectionError("upstream is down")

    service = MarketDataService(token="k", client=Broken())
    with pytest.raises(MarketDataError, match="upstream is down"):
        service.ohlcv("AAPL.US")


def test_an_empty_response_is_reported_not_returned_blank():
    class Empty(FakeEodApiClient):
        def dividends(self, token, ticker, as_df=True):
            return pd.DataFrame()

    service = MarketDataService(token="k", client=Empty())
    with pytest.raises(MarketDataError, match="no dividends"):
        service.dividends("AAPL.US")
