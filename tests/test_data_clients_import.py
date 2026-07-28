"""Smoke tests: importing eod_api.py and async_eod.py must not blow up, and no
real network calls are made here. Key handling itself is covered by
tests/test_eod_key_handling.py.
"""

import importlib


def test_eod_api_imports(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "dummy-token")
    module = importlib.import_module("portfolio_construction.eod_api")
    importlib.reload(module)
    assert hasattr(module, "fundamentals")
    assert hasattr(module, "adjusted_prices")


def test_async_eod_imports(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "dummy-token")
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    assert hasattr(module, "get_full_history")
    assert hasattr(module, "sovereign_bonds_tickers")
    assert module.sovereign_bonds_tickers(["US", "FR"], [5, 10]) == [
        "US5Y.GBOND", "US10Y.GBOND", "FR5Y.GBOND", "FR10Y.GBOND",
    ]
