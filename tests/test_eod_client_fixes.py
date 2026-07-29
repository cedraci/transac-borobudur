"""Regression tests for the four broken client code paths (spec section 7.1)."""

import json
import re

import pandas as pd

import portfolio_construction.async_eod as async_eod
import portfolio_construction.eod_api as eod_api


class _FakeResponse:
    """Stands in for the object urllib.request.urlopen returns."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


def test_full_history_url_uses_four_digit_year(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "unit-test-key")
    url = async_eod.full_history_url("AAPL.US")
    to_value = url.split("&to=")[1].split("&")[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", to_value), to_value


def test_assemble_sovereign_rates_skips_none_entries():
    results = [{"US10Y.GBOND": 4.25}, None, {"FR10Y.GBOND": 3.10}]
    df = async_eod._assemble_sovereign_rates(results)
    assert list(df.columns) == ["Rates"]
    assert df.loc["US10Y.GBOND", "Rates"] == 4.25
    assert df.loc["FR10Y.GBOND", "Rates"] == 3.10
    assert "None" not in df.index


def test_assemble_sovereign_rates_all_none_returns_empty_frame():
    df = async_eod._assemble_sovereign_rates([None, None])
    assert list(df.columns) == ["Rates"]
    assert df.empty


def test_last_prices_universe_returns_series_indexed_by_code(monkeypatch):
    payload = [
        {"code": "AAPL.US", "close": 190.5},
        {"code": "MSFT.US", "close": 410.25},
    ]
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    result = eod_api.last_prices_universe("dummy-token", ["AAPL.US", "MSFT.US"])
    assert isinstance(result, pd.Series)
    assert result["AAPL.US"] == 190.5
    assert result["MSFT.US"] == 410.25


def test_last_prices_universe_handles_single_ticker_dict_payload(monkeypatch):
    payload = {"code": "AAPL.US", "close": 190.5}
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    result = eod_api.last_prices_universe("dummy-token", ["AAPL.US"])
    assert result["AAPL.US"] == 190.5


def test_adjusted_prices_returns_a_datetime_indexed_frame(monkeypatch):
    # infer_datetime_format was removed in pandas 3, so this call raised a
    # TypeError for every payload regardless of its contents.
    payload = [
        {"date": "2024-01-02", "adjusted_close": 100.0},
        {"date": "2024-01-03", "adjusted_close": 101.0},
    ]
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    out = eod_api.adjusted_prices("dummy-token", ["AAPL.US"])
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == ["AAPL.US"]
    assert len(out) == 2
    assert out.index[0] == pd.Timestamp("2024-01-02")


def test_download_universe_returns_a_datetime_indexed_frame(monkeypatch):
    payload = [
        {"date": "2024-01-02", "adjusted_close": 100.0},
        {"date": "2024-01-03", "adjusted_close": 101.0},
    ]
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    out = eod_api.download_universe("dummy-token", ["AAPL.US"])
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == ["AAPL.US"]
    assert len(out) == 2


def test_last_prices_passes_its_own_token_argument(monkeypatch):
    captured = {}

    def fake_recursive(token, ticker, strDate):
        captured["token"] = token
        return 123.0

    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse([])
    )
    monkeypatch.setattr(eod_api, "recursive_adjClose_atDate", fake_recursive)
    eod_api.last_prices("token-from-argument", "AAPL.US")
    assert captured["token"] == "token-from-argument"
