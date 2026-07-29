"""LocalSource serves an already-materialized price frame."""

import datetime as dt
import io

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, CapabilityNotSupported
from portfolio_ui.sources.local_source import LocalSource


def _frame():
    return pd.DataFrame(
        {"AAPL.US": [100.0, 101.0, 102.0], "MSFT.US": [400.0, 401.0, 402.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_declares_only_price_history():
    source = LocalSource(_frame(), label="upload:prices.csv")
    assert source.capabilities == frozenset({Capability.PRICE_HISTORY})


def test_name_reports_the_label():
    source = LocalSource(_frame(), label="upload:prices.csv")
    assert source.name == "upload:prices.csv"


def test_latest_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        LocalSource(_frame(), label="local").latest(["AAPL.US"])


def test_price_history_slices_and_selects():
    source = LocalSource(_frame(), label="local")
    out = source.price_history(
        ["MSFT.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 4)
    )
    assert list(out.columns) == ["MSFT.US"]
    assert len(out) == 2


def test_available_tickers_lists_frame_columns():
    source = LocalSource(_frame(), label="local")
    assert source.available_tickers() == ["AAPL.US", "MSFT.US"]


def test_from_upload_reads_csv():
    csv = b"Date,AAPL.US\n2024-01-02,100.0\n2024-01-03,101.0\n"
    source = LocalSource.from_upload(io.BytesIO(csv), "prices.csv")
    assert source.name == "upload:prices.csv"
    assert source.available_tickers() == ["AAPL.US"]


def test_from_upload_rejects_unknown_extension():
    with pytest.raises(ValueError, match="unsupported"):
        LocalSource.from_upload(io.BytesIO(b""), "prices.txt")
