"""Stubs for the pages built in later plans.

Each states what it will do so the navigation shell is complete and honest.
"""

import streamlit as st

_COMING = {
    "Risk": "VaR, expected shortfall and Monte Carlo simulation.",
    "Optimization": "Eight optimization objectives, bounds and covariance estimators.",
    "Backtest": "Rebalanced backtests with momentum universe selection.",
    "Market Data": "Fundamentals, dividends, earnings and macro series (eod_api only).",
}


def _stub(title: str) -> None:
    st.title(title)
    st.info(f"Not built yet. Planned: {_COMING[title]}")


def risk_page():
    _stub("Risk")


def optimization_page():
    _stub("Optimization")


def backtest_page():
    _stub("Backtest")


def market_data_page():
    _stub("Market Data")
