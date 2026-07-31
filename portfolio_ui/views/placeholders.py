"""Stubs for the pages built in later plans.

Each states what it will do so the navigation shell is complete and honest.
"""

import streamlit as st

_COMING = {
    "Market Data": "Fundamentals, dividends, earnings and macro series (eod_api only).",
}


def _stub(title: str) -> None:
    st.title(title)
    st.info(f"Not built yet. Planned: {_COMING[title]}")


def market_data_page():
    _stub("Market Data")
