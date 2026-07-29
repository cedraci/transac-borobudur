"""A price frame the user supplied by upload or reloaded from disk."""

from __future__ import annotations

from pathlib import PurePath

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    TickerNotFound,
    normalize_price_frame,
)


class LocalSource(BaseSource):
    """Wraps an in-memory frame so uploads satisfy the same contract.

    Not offered in the source selector (spec section 5.1) - it is reached
    through upload and saved-dataset loading - but it implements PriceSource so
    the cross-source contract test covers it.
    """

    capabilities = frozenset({Capability.PRICE_HISTORY})

    def __init__(self, frame: pd.DataFrame, label: str):
        self._frame = frame
        self.name = label

    @classmethod
    def from_upload(cls, file_like, filename: str) -> "LocalSource":
        suffix = PurePath(filename).suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(file_like, index_col=0, parse_dates=True)
        elif suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(file_like, index_col=0, parse_dates=True)
        elif suffix == ".parquet":
            frame = pd.read_parquet(file_like)
        else:
            raise ValueError(
                f"unsupported file type '{suffix}' - use .csv, .xlsx or .parquet"
            )
        return cls(frame, label=f"upload:{filename}")

    def available_tickers(self) -> list[str]:
        return list(self._frame.columns)

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        frame = normalize_price_frame(self._frame, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"the loaded file has no data for {list(tickers)} in that range"
            )
        return frame
