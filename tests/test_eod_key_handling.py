"""Both EOD clients must import without EOD_API_KEY set, and only fail when a
call actually needs the key (spec section 6.5)."""

import importlib

import pytest


def test_async_eod_imports_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    assert hasattr(module, "get_full_history")


def test_eod_api_imports_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.eod_api")
    importlib.reload(module)
    assert hasattr(module, "adjusted_prices")


def test_async_eod_url_builder_raises_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    with pytest.raises(RuntimeError, match="EOD_API_KEY"):
        module.full_history_url("AAPL.US")


def test_async_eod_url_builder_uses_key_when_present(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "unit-test-key")
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    assert "api_token=unit-test-key" in module.full_history_url("AAPL.US")


def test_api_key_read_at_call_time_not_import_time(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    monkeypatch.setenv("EOD_API_KEY", "set-after-import")
    assert module._api_key() == "set-after-import"
