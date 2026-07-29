"""The shared dataset: building, validation and round-tripping to disk."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.dataset import (
    ActiveDataset,
    build_dataset,
    dataset_from_frame,
    default_directory,
    infer_frequency,
    list_saved,
    load_dataset,
    notes_for_fetch,
    save_dataset,
    validate_prices,
)
from portfolio_ui.sources.local_source import LocalSource

TICKERS = ["AAPL.US", "MSFT.US"]


def _frame():
    return pd.DataFrame(
        {t: [100.0, 101.0, 102.0] for t in TICKERS},
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="Date"
        ),
    )


def test_validate_rejects_non_datetime_index():
    frame = _frame().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_prices(frame)


def test_validate_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        validate_prices(pd.DataFrame())


def test_validate_rejects_non_numeric_columns():
    frame = _frame()
    frame["TEXT"] = "abc"
    with pytest.raises(ValueError, match="numeric"):
        validate_prices(frame)


def test_validate_accepts_a_good_frame():
    validate_prices(_frame())  # must not raise


def test_build_dataset_records_provenance_and_metadata():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(source, "sample", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(ds, ActiveDataset)
    assert ds.name == "sample"
    assert ds.source == "local"
    assert ds.tickers == tuple(TICKERS)
    assert ds.frequency  # inferred, non-empty
    assert isinstance(ds.fetched_at, dt.datetime)


def test_build_dataset_narrows_range_to_what_came_back():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(source, "sample", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert ds.start == dt.date(2024, 1, 2)
    assert ds.end == dt.date(2024, 1, 4)


def test_build_dataset_notes_client_side_slicing_for_async_eod():
    # LocalSource takes its name from the label, so this stands in for a fetch
    # that actually ran through async_eod.
    source = LocalSource(_frame(), label="async_eod")
    ds = build_dataset(source, "s", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert any("client-side" in note for note in ds.notes)


def test_infer_frequency_of_a_single_row_is_unknown():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]), name="Date"),
    )
    assert infer_frequency(frame.index) == "Unknown"


def test_infer_frequency_of_daily_data():
    assert infer_frequency(_frame().index) == "Daily"


def test_notes_for_fetch_is_empty_when_everything_came_back():
    assert notes_for_fetch("eod_api", TICKERS, TICKERS) == ()


def test_build_dataset_notes_dropped_tickers():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(
        source, "s", TICKERS + ["GONE.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert any("GONE.US" in note for note in ds.notes)
    assert ds.tickers == tuple(TICKERS)


def test_dataset_is_frozen():
    ds = dataset_from_frame(_frame(), "sample", "upload:x.csv")
    with pytest.raises(Exception):
        ds.name = "other"


def test_save_and_load_round_trip(tmp_path):
    ds = dataset_from_frame(_frame(), "sample", "upload:x.csv")
    path = save_dataset(ds, directory=tmp_path)
    assert path.exists()

    loaded = load_dataset("sample", directory=tmp_path)
    pd.testing.assert_frame_equal(loaded.prices, ds.prices)
    assert loaded.source == "upload:x.csv"
    assert loaded.tickers == ds.tickers
    assert loaded.frequency == ds.frequency
    assert loaded.notes == ds.notes


def test_list_saved_returns_names(tmp_path):
    save_dataset(dataset_from_frame(_frame(), "one", "local"), directory=tmp_path)
    save_dataset(dataset_from_frame(_frame(), "two", "local"), directory=tmp_path)
    assert sorted(list_saved(directory=tmp_path)) == ["one", "two"]


def test_load_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("nope", directory=tmp_path)


def test_default_directory_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_UI_DATA_DIR", str(tmp_path))
    assert default_directory() == tmp_path
