"""The AlphaVantage key must come from the environment, never from source."""

import inspect

import pytest

import portfolio_construction.market_access as ma


def test_no_hardcoded_key_in_source():
    source = inspect.getsource(ma)
    assert "KO6RI8AXUX0HDGC2" not in source


def test_require_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        ma._require_alphavantage_key()


def test_require_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test-key")
    assert ma._require_alphavantage_key() == "av-test-key"


def test_historical_fx_url_carries_env_key(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test-key")
    captured = {}

    class _FakeReq:
        def json(self):
            return {
                "Time Series FX (Daily)": {
                    "2024-01-02": {
                        "1. open": "1.10",
                        "2. high": "1.20",
                        "3. low": "1.00",
                        "4. close": "1.15",
                    }
                }
            }

    def fake_get(url):
        captured["url"] = url
        return _FakeReq()

    monkeypatch.setattr(ma.requests, "get", fake_get)
    frame = ma.historical_fx("EUR", "USD")

    assert "apikey=av-test-key" in captured["url"]
    assert list(frame.columns) == ["Open", "High", "Low", "Close"]
    assert frame["Close"].iloc[0] == 1.15
