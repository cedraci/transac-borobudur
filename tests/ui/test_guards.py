"""Input parsing and the disabled-action reasons shown as tooltips."""

import pandas as pd

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.guards import (
    capability_blocked_reason,
    parse_tickers,
    require_active_dataset,
)
from portfolio_ui.sources.base import Capability
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource
from portfolio_ui.state import init_state, set_active_dataset


def test_parse_tickers_splits_on_commas_and_whitespace():
    assert parse_tickers("AAPL.US, MSFT.US\nGOOG.US  TSLA.US") == [
        "AAPL.US",
        "MSFT.US",
        "GOOG.US",
        "TSLA.US",
    ]


def test_parse_tickers_uppercases_and_strips():
    assert parse_tickers("  aapl.us ") == ["AAPL.US"]


def test_parse_tickers_deduplicates_preserving_order():
    assert parse_tickers("AAPL.US, MSFT.US, AAPL.US") == ["AAPL.US", "MSFT.US"]


def test_parse_tickers_of_empty_input_is_empty():
    assert parse_tickers("   ") == []


def test_capability_blocked_reason_is_none_when_supported():
    source = EodApiSource(token="k")
    assert capability_blocked_reason(source, Capability.LATEST) is None


def test_capability_blocked_reason_names_source_and_capability():
    source = MarketAccessSource()
    reason = capability_blocked_reason(source, Capability.LATEST)
    assert "market_access" in reason
    assert "latest" in reason


def test_capability_blocked_reason_reports_missing_key_first():
    source = EodApiSource(token="")
    reason = capability_blocked_reason(source, Capability.PRICE_HISTORY)
    assert "EOD_API_KEY" in reason


def test_require_active_dataset_returns_none_when_absent():
    store = {}
    init_state(store)
    assert require_active_dataset(store) is None


def test_require_active_dataset_returns_the_dataset():
    store = {}
    init_state(store)
    frame = pd.DataFrame(
        {"AAPL.US": [1.0, 2.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    ds = dataset_from_frame(frame, "sample", "local")
    set_active_dataset(store, ds)
    assert require_active_dataset(store) is ds
