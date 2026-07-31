"""Performance statistics, drawdown episodes and calendar returns."""

from __future__ import annotations

import streamlit as st

from portfolio_ui.analytics import (
    AnalyticsError,
    calendar_table,
    drawdown_episodes,
    drawdown_series,
    monthly_returns_table,
    performance_table,
    rebased_prices,
    rolling_cagr,
    weighted_nav,
)
from portfolio_ui.charts import calendar_bar_figure, drawdown_figure, nav_figure
from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.state import add_derived


def _weights_editor(dataset):
    """Let the user weight the basket, defaulting to equal weight."""
    equal = 1.0 / len(dataset.tickers)
    with st.expander("Weights", expanded=False):
        st.caption("Fixed weights, buy and hold. Rebalancing lives on the Backtest page.")
        weights = {
            ticker: st.number_input(
                ticker, min_value=0.0, max_value=1.0, value=equal, step=0.05,
                key=f"analysis_w_{ticker}",
            )
            for ticker in dataset.tickers
        }
        total = sum(weights.values())
        st.caption(f"Total: {total:.4f}")
    return weights


def analysis_page() -> None:
    st.title("Analysis")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    rf = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.25, value=0.0, step=0.005
    )
    weights = _weights_editor(dataset)

    try:
        nav = weighted_nav(dataset.prices, weights)
    except AnalyticsError as exc:
        st.error(str(exc))
        return

    per_asset, portfolio = st.tabs(["Per asset", "Portfolio"])

    with per_asset:
        try:
            st.dataframe(performance_table(dataset.prices, rf=rf), width="stretch")
        except AnalyticsError as exc:
            st.error(str(exc))

        st.subheader("Rebase to a chosen date")
        anchor = st.date_input(
            "Rebase from", value=dataset.start,
            min_value=dataset.start, max_value=dataset.end,
        )
        try:
            st.dataframe(rebased_prices(dataset.prices, anchor).tail(20), width="stretch")
        except AnalyticsError as exc:
            # Non-trading days are the normal case here, not an error state.
            st.info(str(exc))

    with portfolio:
        st.plotly_chart(nav_figure(nav, f"{dataset.name} - weighted NAV"), width="stretch")

        try:
            st.plotly_chart(drawdown_figure(drawdown_series(nav)), width="stretch")
        except AnalyticsError as exc:
            st.error(str(exc))

        st.subheader("Worst drawdown episodes")
        top_n = st.slider("How many", min_value=3, max_value=20, value=10)
        try:
            st.dataframe(drawdown_episodes(nav, top_n=top_n), width="stretch")
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute drawdown episodes: {exc}")

        st.subheader("Calendar performance")
        try:
            calendar = calendar_table(nav)
            st.plotly_chart(calendar_bar_figure(calendar), width="stretch")
            st.dataframe(calendar, width="stretch")
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute calendar performance: {exc}")

        st.subheader("Monthly returns")
        try:
            st.dataframe(monthly_returns_table(nav), width="stretch")
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute monthly returns: {exc}")

        st.subheader("Rolling CAGR")
        horizon = st.slider("Max holding period (years)", 2, 20, 10)
        if st.button("Compute rolling CAGR"):
            try:
                st.dataframe(rolling_cagr(nav, max_holding_period=horizon), width="stretch")
            except (AnalyticsError, ValueError, KeyError) as exc:
                st.error(f"Could not compute rolling CAGR: {exc}")

        if st.button("Save this NAV as a derived dataset"):
            derived = dataset_from_frame(
                nav.to_frame(), f"{dataset.name}-nav", dataset.source,
                notes=("weighted NAV derived on the Analysis page",),
            )
            add_derived(store, derived)
            st.success(f"Registered derived dataset '{derived.name}'")
