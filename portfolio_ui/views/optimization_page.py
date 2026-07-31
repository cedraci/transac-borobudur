"""Optimal weights across the eight objectives, plus the estimator tools."""

from __future__ import annotations

import numpy as np
import streamlit as st

from portfolio_ui.charts import (
    correlation_heatmap_figure,
    frontier_figure,
    weights_bar_figure,
)
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.optimize import (
    OBJECTIVES,
    OptimizeError,
    bayes_stein_returns,
    capital_protection,
    correlation_matrix,
    efficient_frontier,
    optimize,
    risk_contributions,
)


def _allocate_tab(dataset):
    objective = st.selectbox(
        "Objective", options=sorted(OBJECTIVES), index=sorted(OBJECTIVES).index("minimum_variance")
    )
    st.caption(OBJECTIVES[objective])

    col_cov, col_target = st.columns(2)
    cov_mat = col_cov.selectbox(
        "Covariance estimator", ["sample", "shrunked", "gerber"],
        help="shrunked tempers the sample estimate; gerber is robust to outliers",
    )
    target_return = col_target.number_input(
        "Target return (annual)", min_value=0.0, max_value=1.0, value=0.10, step=0.01,
        help="Only used by mean_variance",
        disabled=objective != "mean_variance",
    )

    constrain = st.checkbox("Constrain each weight", value=False)
    bounds = None
    if constrain:
        low, high = st.slider(
            "Per-asset weight bounds", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05
        )
        bounds = (low, high)

    if not st.button("Optimize"):
        return

    try:
        weights = optimize(
            dataset.prices, objective, bounds=bounds, cov_mat=cov_mat,
            target_return=target_return,
        )
        contributions = risk_contributions(dataset.prices, weights)
    except OptimizeError as exc:
        st.error(str(exc))
        return

    st.plotly_chart(
        weights_bar_figure(weights, f"{objective} weights"), width="stretch"
    )

    returns = dataset.prices.pct_change().dropna()
    portfolio = returns.mul(weights.fillna(0.0)).sum(axis=1)
    col_ret, col_vol, col_sharpe = st.columns(3)
    annual_return = float(portfolio.mean() * 252)
    annual_vol = float(portfolio.std() * np.sqrt(252))
    col_ret.metric("Expected return (annual)", f"{annual_return:.2%}")
    col_vol.metric("Volatility (annual)", f"{annual_vol:.2%}")
    col_sharpe.metric(
        "Sharpe", f"{annual_return / annual_vol:.2f}" if annual_vol else "n/a"
    )

    st.subheader("Risk decomposition")
    st.caption(
        "Each asset's share of total portfolio risk. Equal risk contribution "
        "aims to make these identical."
    )
    st.dataframe(
        contributions.rename("Share of risk").to_frame().style.format("{:.2%}"),
        width="stretch",
    )


def _frontier_tab(dataset):
    points = st.slider("Points on the frontier", min_value=5, max_value=40, value=15)
    st.caption(
        "Each point is the lowest-volatility portfolio achieving a given return. "
        "One solve per point, so this is behind a button."
    )

    if not st.button("Compute frontier"):
        return

    with st.spinner(f"Solving {points} portfolios..."):
        try:
            frontier = efficient_frontier(dataset.prices, points=points)
        except OptimizeError as exc:
            st.error(str(exc))
            return

        marks = {}
        returns = dataset.prices.pct_change().dropna()
        for objective in ("minimum_variance", "maximum_sharpe"):
            try:
                weights = optimize(dataset.prices, objective)
            except OptimizeError:
                continue
            portfolio = returns.mul(weights.fillna(0.0)).sum(axis=1)
            marks[objective] = (
                float(portfolio.std() * np.sqrt(252)),
                float(portfolio.mean() * 252),
            )

    st.plotly_chart(frontier_figure(frontier, points=marks), width="stretch")
    st.dataframe(frontier.style.format("{:.2%}"), width="stretch")


def _estimators_tab(dataset):
    st.subheader("Correlations")
    method = st.radio(
        "Estimator", ["sample", "gerber"], horizontal=True,
        help="Gerber counts co-movements beyond a threshold, ignoring outlier magnitudes",
    )
    try:
        st.plotly_chart(
            correlation_heatmap_figure(correlation_matrix(dataset.prices, method=method)),
            width="stretch",
        )
    except OptimizeError as exc:
        st.error(str(exc))

    st.subheader("Bayes-Stein expected returns")
    st.caption(
        "Sample means are noisy optimizer inputs; shrinkage pulls extreme "
        "estimates toward the cross-sectional average."
    )
    try:
        raw = dataset.prices.pct_change().dropna().mean().mul(252).rename("Sample")
        shrunk = bayes_stein_returns(dataset.prices).mul(252).rename("Bayes-Stein")
        st.dataframe(
            raw.to_frame().join(shrunk).style.format("{:.2%}"), width="stretch"
        )
    except OptimizeError as exc:
        st.error(str(exc))

    st.subheader("Capital protection")
    col_duration, col_confidence = st.columns(2)
    duration = col_duration.number_input(
        "Horizon (years)", min_value=1, max_value=30, value=5
    )
    confidence = col_confidence.slider(
        "Confidence", min_value=0.80, max_value=0.99, value=0.95, step=0.01
    )
    if st.button("Maximise return subject to protecting capital"):
        try:
            result = capital_protection(
                dataset.prices, duration=int(duration), confidence=confidence
            )
        except OptimizeError as exc:
            st.error(str(exc))
            return

        if not result.success:
            st.warning(
                f"The solver did not converge: {result.message}. "
                "The weights below are its best attempt, not a solution."
            )
        st.plotly_chart(
            weights_bar_figure(result.weights, "Capital-protection weights"),
            width="stretch",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected return (annual)", f"{result.expected_return_annual:.2%}")
        c2.metric("Volatility (annual)", f"{result.volatility_annual:.2%}")
        c3.metric("Worst case over horizon", f"{result.worst_case_total_return:.2%}")


def optimization_page() -> None:
    st.title("Optimization")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    if len(dataset.tickers) < 2:
        st.warning("Optimization needs at least two assets; this dataset has one.")
        return

    allocate, frontier, estimators = st.tabs(
        ["Allocate", "Efficient frontier", "Estimators"]
    )
    with allocate:
        _allocate_tab(dataset)
    with frontier:
        _frontier_tab(dataset)
    with estimators:
        _estimators_tab(dataset)
