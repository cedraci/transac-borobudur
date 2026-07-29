"""The source protocol every market-data client is adapted to.

Nothing in this module may import streamlit - it is plain, testable Python.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Iterable, Mapping, Protocol, runtime_checkable

import pandas as pd


class Capability(str, Enum):
    """What a source can do. Values are stable identifiers used in the UI."""

    PRICE_HISTORY = "price_history"
    CLOSE_AT = "close_at"
    LATEST = "latest"
    SOVEREIGN = "sovereign_yields"


class SourceError(RuntimeError):
    """Base class for every failure a source can report."""


class SourceUnavailable(SourceError):
    """The source cannot be used at all, typically a missing API key."""


class TickerNotFound(SourceError):
    """Upstream returned no data for one or more tickers."""


class UpstreamError(SourceError):
    """The upstream API failed or returned something unparseable."""


class CapabilityNotSupported(SourceError):
    """This source does not implement the requested capability."""


@runtime_checkable
class PriceSource(Protocol):
    """The four capabilities the UI can route to any source."""

    name: str
    capabilities: frozenset[Capability]

    def price_history(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame: ...

    def close_at(self, tickers: list[str], on: dt.date) -> pd.Series: ...

    def latest(self, tickers: list[str]) -> pd.Series: ...

    def sovereign_yields(
        self, countries: list[str], tenors: list[int], on: dt.date
    ) -> pd.DataFrame: ...


class BaseSource:
    """Concrete base so subclasses only implement what they support."""

    name: str = "base"
    capabilities: frozenset[Capability] = frozenset()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def is_available(self) -> bool:
        """Whether the source is usable right now (e.g. its API key is set)."""
        return True

    def unavailable_reason(self) -> str | None:
        """Human-readable reason the source cannot be used, or None."""
        return None

    def _unsupported(self, capability: Capability):
        raise CapabilityNotSupported(
            f"{self.name} does not support {capability.value}"
        )

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        self._unsupported(Capability.PRICE_HISTORY)

    def close_at(self, tickers, on) -> pd.Series:
        self._unsupported(Capability.CLOSE_AT)

    def latest(self, tickers) -> pd.Series:
        self._unsupported(Capability.LATEST)

    def sovereign_yields(self, countries, tenors, on) -> pd.DataFrame:
        self._unsupported(Capability.SOVEREIGN)


def normalize_price_frame(
    frame: pd.DataFrame,
    tickers: Iterable[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Coerce any client's price output into the shared contract.

    Ascending DatetimeIndex named "Date", float64 values, one column per
    requested ticker in the requested order. Tickers with no data are dropped.
    """
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    present = [t for t in tickers if t in out.columns]
    out = out[present]
    out = out.apply(pd.to_numeric, errors="coerce").astype("float64")

    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]

    return out


def normalize_price_series(
    values: Mapping[str, object], tickers: Iterable[str]
) -> pd.Series:
    """Coerce point-in-time / latest prices into a float Series by ticker."""
    ordered = {t: values[t] for t in tickers if t in values}
    return pd.Series(ordered, dtype="float64")
