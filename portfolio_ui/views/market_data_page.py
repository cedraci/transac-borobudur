"""Fundamentals, corporate actions, calendars and macro series.

Everything here comes from the synchronous EOD client, which is the only
source that offers these endpoints - so unlike the other pages this one does
not consume the active dataset, and it works whichever source is selected.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from portfolio_ui.marketdata import MarketDataError, MarketDataService


def _run(label: str, fn, *args, **kwargs):
    """Call the service and render its error rather than raising."""
    try:
        return fn(*args, **kwargs)
    except MarketDataError as exc:
        st.error(str(exc))
        return None


def _discovery_tab(service, blocked):
    st.subheader("Ticker search")
    query = st.text_input("Company or code", value="apple", key="md_search")
    if st.button("Search", disabled=bool(blocked), help=blocked):
        found = _run("search", service.search, query)
        if found is not None:
            st.caption("Use the Ticker column on the Data page.")
            st.dataframe(found, width="stretch")

    st.divider()
    st.subheader("Index constituents")
    index_ticker = st.text_input(
        "Index ticker", value="GSPC.INDX",
        help="e.g. GSPC.INDX for the S&P 500, FCHI.INDX for the CAC 40",
    )
    if st.button("List constituents", disabled=bool(blocked), help=blocked):
        tickers = _run("constituents", service.index_constituents, index_ticker)
        if tickers is not None:
            st.success(f"{len(tickers)} constituents")
            st.code(", ".join(tickers))

    st.divider()
    st.subheader("Fundamentals")
    col_ticker, col_section = st.columns(2)
    ticker = col_ticker.text_input("Ticker", value="AAPL.US", key="md_fundamentals")
    section = col_section.selectbox(
        "Section",
        ["General", "Highlights", "Valuation", "SharesStats", "Technicals", "ETF_Data"],
    )
    if st.button("Fetch fundamentals", disabled=bool(blocked), help=blocked):
        frame = _run("fundamentals", service.fundamentals, ticker, section=section)
        if frame is not None:
            st.dataframe(frame, width="stretch")


def _corporate_actions_tab(service, blocked):
    ticker = st.text_input("Ticker", value="AAPL.US", key="md_actions")

    prices, dividends, splits, yields = st.tabs(
        ["Price history", "Dividends", "Splits", "Dividend yield"]
    )

    with prices:
        if st.button("Fetch OHLCV", disabled=bool(blocked), help=blocked):
            frame = _run("ohlcv", service.ohlcv, ticker)
            if frame is not None:
                st.dataframe(frame.tail(250), width="stretch")

    with dividends:
        if st.button("Fetch dividends", disabled=bool(blocked), help=blocked):
            frame = _run("dividends", service.dividends, ticker)
            if frame is not None:
                st.dataframe(frame, width="stretch")

    with splits:
        if st.button("Fetch splits", disabled=bool(blocked), help=blocked):
            frame = _run("splits", service.splits, ticker)
            if frame is not None:
                st.dataframe(frame, width="stretch")

    with yields:
        st.caption("Dividends paid in a year over the price at each payment date.")
        if st.button("Fetch dividend yield history", disabled=bool(blocked), help=blocked):
            series = _run("dividend yield", service.dividend_yield_history, ticker)
            if series is not None:
                st.bar_chart(series)
                st.dataframe(series.to_frame("Dividend yield"), width="stretch")


def _calendars_tab(service, blocked):
    st.subheader("Earnings calendar")
    col_start, col_end = st.columns(2)
    start = col_start.date_input("From", value=dt.date.today(), key="md_earn_start")
    end = col_end.date_input(
        "To", value=dt.date.today() + dt.timedelta(days=30), key="md_earn_end"
    )
    if st.button("Fetch earnings", disabled=bool(blocked), help=blocked):
        frame = _run("earnings", service.earnings_calendar, str(start), str(end))
        if frame is not None:
            st.dataframe(frame, width="stretch")

    st.divider()
    st.subheader("Macro events")
    col_country, col_from, col_to = st.columns(3)
    country = col_country.text_input("Country (ISO)", value="US", key="md_macro_country")
    macro_start = col_from.date_input(
        "From", value=dt.date.today() - dt.timedelta(days=90), key="md_macro_start"
    )
    macro_end = col_to.date_input("To", value=dt.date.today(), key="md_macro_end")
    if st.button("Fetch macro events", disabled=bool(blocked), help=blocked):
        frame = _run(
            "macro events", service.macro_events, str(macro_start), str(macro_end), country
        )
        if frame is not None:
            st.dataframe(frame, width="stretch")

    st.divider()
    st.subheader("Macro indicator")
    col_code, col_indicator = st.columns(2)
    code = col_code.text_input("Country code (3 letter)", value="USA")
    indicator = col_indicator.text_input("Indicator", value="gdp_current_usd")
    if st.button("Fetch indicator", disabled=bool(blocked), help=blocked):
        frame = _run("macro indicator", service.macro_indicators, code, indicator)
        if frame is not None:
            st.dataframe(frame, width="stretch")


def _fixed_income_tab(service, blocked):
    st.caption("Duration, maturity, rating and yield for a bond ETF.")
    ticker = st.text_input("ETF ticker", value="AGG.US", key="md_etf")
    if st.button("Fetch fixed-income data", disabled=bool(blocked), help=blocked):
        frame = _run("fixed income", service.fixed_income_etf, ticker)
        if frame is not None:
            st.dataframe(frame, width="stretch")


def market_data_page() -> None:
    st.title("Market Data")

    service = MarketDataService()
    blocked = service.unavailable_reason()

    st.caption(
        "These endpoints exist only on the synchronous EOD client, so this page "
        "works regardless of which source is selected in the sidebar."
    )
    if blocked:
        st.warning(f"{blocked} - every action on this page is disabled.")

    discovery, actions, calendars, fixed_income = st.tabs(
        ["Discovery", "Corporate actions", "Calendars & macro", "Fixed income"]
    )
    with discovery:
        _discovery_tab(service, blocked)
    with actions:
        _corporate_actions_tab(service, blocked)
    with calendars:
        _calendars_tab(service, blocked)
    with fixed_income:
        _fixed_income_tab(service, blocked)
