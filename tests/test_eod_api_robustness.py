"""Two crashes reachable from the Market Data page with ordinary payloads."""

import pytest

import portfolio_construction.eod_api as eod_api


def test_fixed_income_etf_ignores_fields_it_does_not_know(monkeypatch):
    """It mapped every returned key through a hardcoded dict.

    EOD returns whatever fields an ETF has; one outside the dict raised
    KeyError and lost the whole response.
    """
    payload = {
        "EffectiveDuration": {"Relative_to_Category": "5.2"},
        "YieldToMaturity": {"Relative_to_Category": "3.1"},
        "SomeNewFieldEodAdded": {"Relative_to_Category": "9.9"},
    }
    monkeypatch.setattr(eod_api, "fundamentals", lambda *a, **k: payload)

    out = eod_api.fixed_income_etf("dummy-token", "AGG.US")
    assert "Duration" in out.index
    assert "YtM" in out.index
    assert "SomeNewFieldEodAdded" not in out.index


def test_fixed_income_etf_skips_a_field_with_no_relative_value(monkeypatch):
    payload = {
        "EffectiveDuration": {"Relative_to_Category": "5.2"},
        "Coupon": {},  # present but empty
    }
    monkeypatch.setattr(eod_api, "fundamentals", lambda *a, **k: payload)

    out = eod_api.fixed_income_etf("dummy-token", "AGG.US")
    assert "Duration" in out.index
    assert "Coupon" not in out.index


def test_fixed_income_etf_reports_a_payload_with_nothing_usable(monkeypatch):
    monkeypatch.setattr(eod_api, "fundamentals", lambda *a, **k: {})
    with pytest.raises(ValueError, match="no fixed income"):
        eod_api.fixed_income_etf("dummy-token", "AGG.US")


def test_index_constituents_reports_a_payload_without_components(monkeypatch):
    """A non-index ticker returns fundamentals with no Components key."""

    class _Response:
        def read(self):
            return b'{"General": {"Name": "Apple Inc"}}'

    monkeypatch.setattr(eod_api.urllib.request, "urlopen", lambda url: _Response())

    with pytest.raises(ValueError, match="no index constituents"):
        eod_api.index_constituents("dummy-token", "AAPL.US", return_tickers=True)
