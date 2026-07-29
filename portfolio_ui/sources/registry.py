"""Selector names to source instances, plus availability reporting."""

from __future__ import annotations

import os
from dataclasses import dataclass

from portfolio_ui.sources.async_eod_source import AsyncEodSource
from portfolio_ui.sources.base import Capability, PriceSource
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource

# Display order in the sidebar. "local" is intentionally absent: uploads and
# saved datasets are reached from the Data page, not the source selector.
SELECTABLE_SOURCES: tuple[str, ...] = ("async_eod", "eod_api", "market_access")


@dataclass(frozen=True)
class SourceInfo:
    """What the sidebar needs to render one source option."""

    name: str
    capabilities: frozenset[Capability]
    available: bool
    reason: str | None


def build_source(name: str, token: str | None = None) -> PriceSource:
    """Instantiate a source by its selector name."""
    if token is None:
        token = os.environ.get("EOD_API_KEY", "")

    if name == "async_eod":
        return AsyncEodSource(token=token)
    if name == "eod_api":
        return EodApiSource(token=token)
    if name == "market_access":
        return MarketAccessSource()
    raise KeyError(f"unknown source '{name}'")


def describe_sources(token: str | None = None) -> list[SourceInfo]:
    """Describe every selectable source for the sidebar."""
    infos = []
    for name in SELECTABLE_SOURCES:
        source = build_source(name, token=token)
        infos.append(
            SourceInfo(
                name=name,
                capabilities=source.capabilities,
                available=source.is_available(),
                reason=source.unavailable_reason(),
            )
        )
    return infos
