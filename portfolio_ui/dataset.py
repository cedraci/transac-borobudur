"""The active dataset every downstream page consumes.

No streamlit import here - session wiring lives in state.py.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from portfolio_construction.stats import time_series_frequence_inference
from portfolio_ui.sources.async_eod_source import CLIENT_SIDE_SLICE_NOTE


@dataclass(frozen=True)
class ActiveDataset:
    """A price matrix plus the provenance needed to reproduce it."""

    name: str
    prices: pd.DataFrame
    source: str
    tickers: tuple[str, ...]
    start: dt.date
    end: dt.date
    fetched_at: dt.datetime
    frequency: str
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{len(self.tickers)} cols - {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} "
            f"- via {self.source}"
        )


def validate_prices(frame: pd.DataFrame) -> None:
    """Reject frames that would break downstream analysis."""
    if frame is None or frame.empty:
        raise ValueError("price frame is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("price frame must have a DatetimeIndex")
    non_numeric = [
        col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        raise ValueError(f"non-numeric columns are not allowed: {non_numeric}")


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Frequency of the index, or "Unknown" when there is too little to infer.

    time_series_frequence_inference needs at least two observations to take a
    diff; a single-day fetch would otherwise raise.
    """
    if len(index) < 2:
        return "Unknown"
    return time_series_frequence_inference(index)


# None of the remote sources pass the requested range upstream: async_eod asks
# for 1990-to-today, eod_api's ohlcv sends no from/to at all, and market_access
# asks yfinance for period="max". They all slice locally, so the provenance note
# belongs on all three - not just async_eod.
_CLIENT_SIDE_SLICE_NOTES = {
    "async_eod": CLIENT_SIDE_SLICE_NOTE,
    "eod_api": (
        "eod_api requests the full available history and ignores start/end; "
        "the range was applied client-side"
    ),
    "market_access": (
        "market_access requests period='max' and ignores start/end; "
        "the range was applied client-side"
    ),
}


def notes_for_fetch(source_name: str, requested, returned_columns) -> tuple[str, ...]:
    """Caveats worth surfacing about a fetch that just completed.

    Shared by build_dataset and the Data page, which caches the frame itself.
    """
    notes: list[str] = []
    slice_note = _CLIENT_SIDE_SLICE_NOTES.get(source_name)
    if slice_note:
        notes.append(slice_note)

    dropped = [t for t in requested if t not in returned_columns]
    if dropped:
        notes.append(f"no data returned for: {', '.join(dropped)}")
    return tuple(notes)


def dataset_from_frame(
    frame: pd.DataFrame,
    name: str,
    source_label: str,
    notes: tuple[str, ...] = (),
) -> ActiveDataset:
    """Wrap an already-normalized frame as a dataset."""
    validate_prices(frame)
    return ActiveDataset(
        name=name,
        prices=frame,
        source=source_label,
        tickers=tuple(frame.columns),
        start=frame.index.min().date(),
        end=frame.index.max().date(),
        fetched_at=dt.datetime.now(),
        frequency=infer_frequency(frame.index),
        notes=notes,
    )


def build_dataset(source, name: str, tickers, start: dt.date, end: dt.date) -> ActiveDataset:
    """Fetch from a source and record what actually came back."""
    requested = list(tickers)
    frame = source.price_history(requested, start, end)
    validate_prices(frame)
    notes = notes_for_fetch(source.name, requested, frame.columns)
    return dataset_from_frame(frame, name, source.name, notes=notes)


def default_directory() -> Path:
    """Where saved datasets live. Override with PORTFOLIO_UI_DATA_DIR."""
    override = os.environ.get("PORTFOLIO_UI_DATA_DIR")
    if override:
        return Path(override)
    return Path("data") / "datasets"


def _paths(name: str, directory: Path | None) -> tuple[Path, Path]:
    base = Path(directory) if directory is not None else default_directory()
    return base / f"{name}.parquet", base / f"{name}.json"


def save_dataset(dataset: ActiveDataset, directory: Path | None = None) -> Path:
    """Write the frame as parquet plus a JSON metadata sidecar."""
    parquet_path, meta_path = _paths(dataset.name, directory)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.prices.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "name": dataset.name,
                "source": dataset.source,
                "tickers": list(dataset.tickers),
                "start": dataset.start.isoformat(),
                "end": dataset.end.isoformat(),
                "fetched_at": dataset.fetched_at.isoformat(),
                "frequency": dataset.frequency,
                "notes": list(dataset.notes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parquet_path


def load_dataset(name: str, directory: Path | None = None) -> ActiveDataset:
    """Read back a saved dataset with its recorded provenance."""
    parquet_path, meta_path = _paths(name, directory)
    if not parquet_path.exists():
        raise FileNotFoundError(f"no saved dataset named '{name}' in {parquet_path.parent}")

    frame = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_prices(frame)

    return ActiveDataset(
        name=meta["name"],
        prices=frame,
        source=meta["source"],
        tickers=tuple(meta["tickers"]),
        start=dt.date.fromisoformat(meta["start"]),
        end=dt.date.fromisoformat(meta["end"]),
        fetched_at=dt.datetime.fromisoformat(meta["fetched_at"]),
        frequency=meta["frequency"],
        notes=tuple(meta["notes"]),
    )


def list_saved(directory: Path | None = None) -> list[str]:
    """Names of every dataset saved in the directory."""
    base = Path(directory) if directory is not None else default_directory()
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.parquet"))
