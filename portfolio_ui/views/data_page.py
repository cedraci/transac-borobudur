"""Build the active dataset: fetch from a source, upload a file, or reload."""

from __future__ import annotations

import datetime as dt
import os

import streamlit as st

from portfolio_ui.cache import cached_price_history
from portfolio_ui.charts import latest_prices_figure, price_history_figure
from portfolio_construction.stats import annualization_factor
from portfolio_ui.dataset import (
    dataset_from_frame,
    list_saved,
    load_dataset,
    notes_for_fetch,
    save_dataset,
)
from portfolio_ui.guards import capability_blocked_reason, parse_tickers
from portfolio_ui.sources.base import Capability, SourceError
from portfolio_ui.sources.local_source import LocalSource
from portfolio_ui.sources.registry import build_source
from portfolio_ui.state import get_active_dataset, get_source_name, set_active_dataset


def _fetch_tab(store, source):
    raw = st.text_area(
        "Tickers", value="AAPL.US, MSFT.US, GOOG.US", help="Comma or newline separated"
    )
    tickers = parse_tickers(raw)

    col_start, col_end = st.columns(2)
    start = col_start.date_input("From", value=dt.date(2015, 1, 1))
    end = col_end.date_input("To", value=dt.date.today())

    name = st.text_input("Dataset name", value="working-set")
    blocked = capability_blocked_reason(source, Capability.PRICE_HISTORY)

    if st.button("Fetch price history", disabled=bool(blocked) or not tickers, help=blocked):
        with st.spinner(f"Fetching {len(tickers)} tickers via {source.name}..."):
            try:
                frame = cached_price_history(
                    source.name,
                    os.environ.get("EOD_API_KEY", ""),
                    tuple(tickers),
                    start,
                    end,
                )
            except SourceError as exc:
                st.error(str(exc))
                return

        # The cache returns the frame, so the dataset is assembled here rather
        # than through build_dataset - same notes, no second fetch.
        dataset = dataset_from_frame(
            frame,
            name,
            source.name,
            notes=notes_for_fetch(source.name, tickers, frame.columns),
        )
        set_active_dataset(store, dataset)
        st.success(f"Loaded {dataset.summary()}")
        for note in dataset.notes:
            st.warning(note)


def _point_in_time_tab(source):
    raw = st.text_area("Tickers", value="AAPL.US, MSFT.US", key="pit_tickers")
    tickers = parse_tickers(raw)
    on = st.date_input("As of", value=dt.date.today() - dt.timedelta(days=1))

    close_blocked = capability_blocked_reason(source, Capability.CLOSE_AT)
    latest_blocked = capability_blocked_reason(source, Capability.LATEST)

    col_close, col_latest = st.columns(2)
    if col_close.button("Close on date", disabled=bool(close_blocked), help=close_blocked):
        try:
            series = source.close_at(tickers, on)
            st.plotly_chart(
                latest_prices_figure(series, f"Close on {on:%Y-%m-%d}"),
                use_container_width=True,
            )
            st.dataframe(series.rename("close"))
        except SourceError as exc:
            st.error(str(exc))

    if col_latest.button("Latest prices", disabled=bool(latest_blocked), help=latest_blocked):
        try:
            series = source.latest(tickers)
            st.plotly_chart(
                latest_prices_figure(series, "Latest prices"), use_container_width=True
            )
            st.dataframe(series.rename("close"))
        except SourceError as exc:
            st.error(str(exc))


def _sovereign_tab(source):
    countries = parse_tickers(st.text_input("Countries (ISO codes)", value="US, FR, DE"))
    tenors_raw = st.text_input("Tenors (years)", value="2, 5, 10")
    tenors = [int(t) for t in parse_tickers(tenors_raw) if t.isdigit()]
    on = st.date_input("As of", value=dt.date.today() - dt.timedelta(days=1), key="sov_date")

    blocked = capability_blocked_reason(source, Capability.SOVEREIGN)
    if st.button("Fetch yields", disabled=bool(blocked), help=blocked):
        try:
            st.dataframe(source.sovereign_yields(countries, tenors, on))
        except SourceError as exc:
            st.error(str(exc))


def _upload_tab(store):
    uploaded = st.file_uploader("Price file", type=["csv", "xlsx", "parquet"])
    name = st.text_input("Dataset name", value="uploaded", key="upload_name")
    if uploaded is not None and st.button("Load file"):
        try:
            local = LocalSource.from_upload(uploaded, uploaded.name)
            frame = local.price_history(
                local.available_tickers(), dt.date(1900, 1, 1), dt.date.today()
            )
            dataset = dataset_from_frame(frame, name, local.name)
        except (ValueError, SourceError) as exc:
            st.error(str(exc))
            return
        set_active_dataset(store, dataset)
        st.success(f"Loaded {dataset.summary()}")


def _saved_tab(store):
    active = get_active_dataset(store)
    if active is not None and st.button("Save active dataset"):
        # ImportError covers a missing pyarrow (declared only in the ui extra),
        # OSError an unwritable PORTFOLIO_UI_DATA_DIR or a parquet whose json
        # sidecar was lost, KeyError/ValueError a sidecar that is incomplete.
        try:
            path = save_dataset(active)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Saved to {path}")

    names = list_saved()
    if not names:
        st.caption("No saved datasets yet.")
        return

    chosen = st.selectbox("Saved datasets", options=names)
    if st.button("Load saved dataset"):
        try:
            dataset = load_dataset(chosen)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            st.error(str(exc))
        else:
            set_active_dataset(store, dataset)
            st.success(f"Loaded {chosen}")


def data_page() -> None:
    st.title("Data")
    store = st.session_state
    source_name = get_source_name(store)

    if source_name is None:
        st.info("Choose a data source in the sidebar to begin.")
        return

    source = build_source(source_name)
    st.caption(f"Selected source: **{source.name}**")

    fetch, point_in_time, sovereign, upload, saved = st.tabs(
        ["Price history", "Point in time", "Sovereign yields", "Upload", "Saved"]
    )
    with fetch:
        _fetch_tab(store, source)
    with point_in_time:
        _point_in_time_tab(source)
    with sovereign:
        _sovereign_tab(source)
    with upload:
        _upload_tab(store)
    with saved:
        _saved_tab(store)

    active = get_active_dataset(store)
    if active is not None:
        st.divider()
        st.subheader("Active dataset")

        # Inferred metadata (spec section 5): frequency drives the annualization
        # factor every downstream statistic uses.
        meta_cols = st.columns(4)
        meta_cols[0].metric("Tickers", len(active.tickers))
        meta_cols[1].metric("Observations", len(active.prices))
        meta_cols[2].metric("Frequency", active.frequency)
        meta_cols[3].metric("Annualization", annualization_factor(active.frequency))

        rebased = st.checkbox("Rebase to 100", value=False)
        st.plotly_chart(price_history_figure(active, rebased=rebased), use_container_width=True)
        st.dataframe(active.prices.tail(20))
