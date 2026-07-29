"""The registry maps selector names to source instances."""

import pytest

from portfolio_ui.sources.base import Capability
from portfolio_ui.sources.registry import (
    SELECTABLE_SOURCES,
    build_source,
    describe_sources,
)


def test_selectable_sources_excludes_local():
    assert "local" not in SELECTABLE_SOURCES


def test_selectable_sources_lists_the_three_clients():
    assert set(SELECTABLE_SOURCES) == {"async_eod", "eod_api", "market_access"}


def test_build_source_returns_named_instance():
    assert build_source("eod_api", token="k").name == "eod_api"
    assert build_source("async_eod", token="k").name == "async_eod"
    assert build_source("market_access").name == "market_access"


def test_build_source_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_source("nope")


def test_describe_sources_reports_availability_without_a_key():
    infos = {info.name: info for info in describe_sources(token="")}
    assert not infos["eod_api"].available
    assert "EOD_API_KEY" in infos["eod_api"].reason
    assert infos["market_access"].available


def test_describe_sources_reports_capabilities():
    infos = {info.name: info for info in describe_sources(token="k")}
    assert infos["async_eod"].capabilities == frozenset(Capability)
    assert infos["market_access"].capabilities == frozenset(
        {Capability.PRICE_HISTORY}
    )
