"""Rebalanced backtests: equity curve, allocation drift and momentum scores."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from portfolio_ui.backtesting import (
    REBALANCING_METHODS,
    BacktestError,
    momentum_scores,
    run_backtest,
)
from portfolio_ui.charts import nav_figure, weights_over_time_figure
from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.optimize import OBJECTIVES
from portfolio_ui.state import add_derived

RESULT_KEY = "backtest_result"


def _configure(dataset):
    """Collect the run's parameters. Returns a kwargs dict for run_backtest."""
    objective = st.selectbox(
        "Objective",
        options=sorted(OBJECTIVES),
        index=sorted(OBJECTIVES).index("minimum_variance"),
    )
    st.caption(OBJECTIVES[objective])

    # The default start must leave room for the lookback, or the very first run
    # fails: the estimator needs history BEFORE the backtest window opens.
    default_lookback = 250
    earliest_usable = dataset.prices.index[
        min(default_lookback, len(dataset.prices) - 1)
    ].date()

    col_start, col_end = st.columns(2)
    start = col_start.date_input(
        "From", value=earliest_usable, min_value=dataset.start, max_value=dataset.end,
        help="Needs at least `lookback` observations of history before it",
    )
    end = col_end.date_input(
        "To", value=dataset.end, min_value=dataset.start, max_value=dataset.end
    )

    col_method, col_lookback = st.columns(2)
    method = col_method.selectbox(
        "Rebalancing",
        options=list(REBALANCING_METHODS),
        format_func=lambda m: f"{REBALANCING_METHODS[m]} ({m})",
    )
    lookback = col_lookback.number_input(
        "Lookback (observations)", min_value=20, max_value=1000, value=250,
        help="History required before the start date, used to estimate the covariance",
    )

    robust = st.checkbox(
        "Shrunk covariance", value=False,
        help="Tempers the sample covariance estimate at each rebalancing",
    )
    stock_picking = st.checkbox(
        "Momentum universe selection", value=False,
        help="Rank by momentum at each rebalancing and hold only the leaders",
    )
    nb_securities = None
    if stock_picking:
        nb_securities = st.number_input(
            "Names to hold", min_value=1, max_value=max(1, len(dataset.tickers)),
            value=min(6, len(dataset.tickers)),
        )

    return {
        "objective": objective,
        "start": str(start),
        "end": str(end),
        "lookback": int(lookback),
        "method": method,
        "robust": robust,
        "stock_picking": stock_picking,
        "nb_securities": int(nb_securities) if nb_securities else None,
    }


def _show_result(store, dataset, result):
    st.success(f"{result.summary()} - {result.duration_seconds:.1f}s")

    if result.start != str(dataset.start):
        st.caption(
            f"Start snapped to {result.start}, the first trading day on or after "
            "the date you chose."
        )

    st.plotly_chart(
        nav_figure(result.equity_curve, f"{result.objective} - equity curve"),
        width="stretch",
    )

    total = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total return", f"{total:.2%}")
    c2.metric("Rebalancings", len(result.rebalancing_dates))
    c3.metric("Observations", len(result.equity_curve))

    st.subheader("Allocation over time")
    st.caption("Weights drift between rebalancings and reset at each one.")
    st.plotly_chart(weights_over_time_figure(result.weights.fillna(0.0)), width="stretch")

    with st.expander("Weights at each rebalancing"):
        st.dataframe(
            result.weights.loc[list(result.rebalancing_dates)].style.format("{:.2%}"),
            width="stretch",
        )

    if st.button("Send this equity curve to Analysis"):
        derived = dataset_from_frame(
            result.equity_curve.to_frame(),
            f"{dataset.name}-{result.objective}-{result.method}",
            dataset.source,
            notes=(result.summary(),),
        )
        add_derived(store, derived)
        st.success(
            f"Registered '{derived.name}'. Analysis and Risk can now use it."
        )


def _momentum_tab(dataset):
    st.caption(
        "The scores the momentum selector ranks by: an annualised regression "
        "slope weighted by the fit's R-squared."
    )
    col_date, col_window = st.columns(2)
    as_of = col_date.date_input(
        "As of", value=dataset.end, min_value=dataset.start, max_value=dataset.end,
        key="momentum_date",
    )
    window = col_window.number_input(
        "Window (observations)", min_value=20, max_value=1000, value=250
    )

    if st.button("Score the universe"):
        try:
            scores = momentum_scores(dataset.prices, as_of, window=int(window))
        except BacktestError as exc:
            st.error(str(exc))
            return
        st.dataframe(scores.rename("Momentum score").to_frame(), width="stretch")


def backtest_page() -> None:
    st.title("Backtest")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    if len(dataset.tickers) < 2:
        st.warning("A backtest needs at least two assets; this dataset has one.")
        return

    run_tab, momentum_tab = st.tabs(["Run", "Momentum scores"])

    with run_tab:
        params = _configure(dataset)

        if st.button("Run backtest"):
            with st.spinner("Rebalancing through history..."):
                try:
                    store[RESULT_KEY] = run_backtest(dataset.prices, **params)
                except BacktestError as exc:
                    store.pop(RESULT_KEY, None)
                    st.error(str(exc))

        result = store.get(RESULT_KEY)
        if result is not None:
            _show_result(store, dataset, result)

    with momentum_tab:
        _momentum_tab(dataset)
