"""Value-at-Risk, expected shortfall, Monte Carlo simulation and stress."""

from __future__ import annotations

import streamlit as st

from portfolio_ui.charts import simulation_fan_figure
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.risk import (
    RiskError,
    covariance_var,
    simulate_paths,
    simulate_portfolio_paths,
    stressed_covariance,
    var_es_table,
)


def risk_page() -> None:
    st.title("Risk")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    col_alpha, col_duration, col_dist = st.columns(3)
    alpha = col_alpha.select_slider(
        "Tail probability", options=[0.01, 0.025, 0.05, 0.10], value=0.05
    )
    duration = col_duration.number_input(
        "Horizon (periods)", min_value=1, max_value=250, value=1
    )
    distrib = col_dist.selectbox("Distribution", ["normal", "student"])

    tail, simulation, stress = st.tabs(["VaR / ES", "Monte Carlo", "Stressed covariance"])

    with tail:
        st.caption(
            f"Loss not exceeded on {(1 - alpha):.0%} of {duration}-period horizons, "
            "equal-weighted across the dataset."
        )
        if st.button("Estimate VaR and ES"):
            with st.spinner("Estimating..."):
                try:
                    table = var_es_table(
                        dataset.prices, None, alpha=alpha, duration=int(duration),
                        distrib=distrib,
                    )
                except (RiskError, ValueError, KeyError) as exc:
                    st.error(str(exc))
                else:
                    st.dataframe(
                        table.style.format("{:.2%}"), use_container_width=True
                    )

    with simulation:
        col_sims, col_days = st.columns(2)
        n_sims = col_sims.number_input(
            "Paths", min_value=100, max_value=20000, value=2000, step=100
        )
        days = col_days.number_input(
            "Days ahead", min_value=5, max_value=500, value=60
        )
        correlated = st.checkbox(
            "Multi-asset with correlations (Cholesky)", value=True,
            help="Unchecked simulates the equal-weighted NAV as a single series",
        )

        if st.button("Run simulation"):
            with st.spinner(f"Simulating {n_sims} paths..."):
                try:
                    if correlated:
                        paths = simulate_portfolio_paths(
                            dataset.prices, None, int(n_sims), int(days), distrib
                        )
                    else:
                        nav = dataset.prices.mean(axis=1)
                        paths = simulate_paths(nav, int(n_sims), int(days), distrib)
                except (RiskError, ValueError, KeyError) as exc:
                    st.error(str(exc))
                else:
                    st.plotly_chart(
                        simulation_fan_figure(paths, "Simulated forward paths"),
                        use_container_width=True,
                    )
                    final = paths.iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median outcome", f"{final.median():,.1f}")
                    c2.metric("5th percentile", f"{final.quantile(0.05):,.1f}")
                    c3.metric("95th percentile", f"{final.quantile(0.95):,.1f}")

    with stress:
        factor = st.slider(
            "Stress factor", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="How far correlations are pushed toward 1",
        )
        if st.button("Compute stressed covariance"):
            try:
                stressed = stressed_covariance(dataset.prices, stress_factor=factor)
                base_var = covariance_var(dataset.prices, alpha=alpha, distrib=distrib)
                stressed_var = covariance_var(
                    dataset.prices, alpha=alpha, distrib=distrib, cov=stressed
                )
            except (RiskError, ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                # Same weights priced under both matrices - that comparison is
                # the whole point of stressing.
                left, right = st.columns(2)
                left.metric("VaR, base covariance", f"{base_var:.2%}")
                right.metric(
                    "VaR, stressed", f"{stressed_var:.2%}",
                    delta=f"{stressed_var - base_var:.2%}", delta_color="inverse",
                )
                st.dataframe(stressed, use_container_width=True)
