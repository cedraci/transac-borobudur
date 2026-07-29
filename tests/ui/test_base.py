"""The normalization contract every source must satisfy (spec section 4.2)."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    CapabilityNotSupported,
    normalize_price_frame,
    normalize_price_series,
)


def _raw_frame():
    # deliberately unsorted, string index, object dtype, extra column
    return pd.DataFrame(
        {
            "AAPL.US": ["191.0", "190.0", "192.0"],
            "MSFT.US": ["410.0", "409.0", "411.0"],
            "IGNORED.US": ["1.0", "2.0", "3.0"],
        },
        index=["2024-01-03", "2024-01-02", "2024-01-04"],
    )


def test_normalize_price_frame_sorts_ascending():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert out.index.is_monotonic_increasing


def test_normalize_price_frame_uses_datetime_index_named_date():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"


def test_normalize_price_frame_casts_to_float():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert all(dtype == "float64" for dtype in out.dtypes)


def test_normalize_price_frame_keeps_requested_columns_in_order():
    out = normalize_price_frame(_raw_frame(), ["MSFT.US", "AAPL.US"])
    assert list(out.columns) == ["MSFT.US", "AAPL.US"]


def test_normalize_price_frame_drops_missing_tickers_silently():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "NOPE.US"])
    assert list(out.columns) == ["AAPL.US"]


def test_normalize_price_frame_slices_date_range_inclusively():
    out = normalize_price_frame(
        _raw_frame(),
        ["AAPL.US"],
        start=dt.date(2024, 1, 3),
        end=dt.date(2024, 1, 4),
    )
    assert out.index.min() == pd.Timestamp("2024-01-03")
    assert out.index.max() == pd.Timestamp("2024-01-04")
    assert len(out) == 2


def test_normalize_price_frame_drops_duplicate_dates_keeping_last():
    frame = pd.DataFrame(
        {"AAPL.US": [1.0, 2.0]}, index=["2024-01-02", "2024-01-02"]
    )
    out = normalize_price_frame(frame, ["AAPL.US"])
    assert len(out) == 1
    assert out.iloc[0, 0] == 2.0


def test_normalize_price_series_indexed_by_requested_tickers():
    out = normalize_price_series({"AAPL.US": "190.5"}, ["AAPL.US", "MSFT.US"])
    assert out["AAPL.US"] == 190.5
    assert "MSFT.US" not in out.index
    assert out.dtype == "float64"


def test_base_source_reports_capabilities():
    class Sample(BaseSource):
        name = "sample"
        capabilities = frozenset({Capability.PRICE_HISTORY})

    source = Sample()
    assert source.supports(Capability.PRICE_HISTORY)
    assert not source.supports(Capability.LATEST)
    assert source.is_available()


def test_base_source_raises_for_unsupported_capability():
    class Sample(BaseSource):
        name = "sample"
        capabilities = frozenset({Capability.PRICE_HISTORY})

    with pytest.raises(CapabilityNotSupported, match="sample"):
        Sample().latest(["AAPL.US"])
