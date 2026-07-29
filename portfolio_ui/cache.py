"""Streamlit caching around the pure source calls.

Kept separate from sources/ so that package stays importable and testable
without streamlit. This module is thin wiring and is not unit-tested.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from portfolio_ui.sources.registry import build_source

CACHE_TTL_SECONDS = 3600


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_price_history(
    source_name: str,
    token: str,
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Fetch price history, memoized on every argument.

    tickers is a tuple because cache keys must be hashable.
    """
    source = build_source(source_name, token=token)
    return source.price_history(list(tickers), start, end)
