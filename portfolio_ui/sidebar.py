"""The persistent sidebar: source selection and active dataset summary."""

from __future__ import annotations

from typing import MutableMapping

import streamlit as st

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.registry import SourceInfo, describe_sources
from portfolio_ui.state import get_active_dataset, get_source_name, set_source_name


def source_option_label(info: SourceInfo) -> str:
    """Name the source, and say why it cannot be used when that applies."""
    if info.available:
        return info.name
    return f"{info.name} - unavailable: {info.reason}"


def dataset_summary_lines(dataset: ActiveDataset | None) -> list[str]:
    """The few lines describing what is currently loaded."""
    if dataset is None:
        return ["No active dataset"]

    lines = [
        f"**{dataset.name}**",
        f"{len(dataset.tickers)} cols - {dataset.start:%Y-%m-%d} to {dataset.end:%Y-%m-%d}",
        f"via {dataset.source} - {dataset.frequency}",
    ]
    lines.extend(f"note: {note}" for note in dataset.notes)
    return lines


def render(store: MutableMapping) -> None:
    """Draw the sidebar. Streamlit wiring only."""
    with st.sidebar:
        st.subheader("Data source")

        infos = describe_sources()
        by_label = {source_option_label(info): info for info in infos}
        current = get_source_name(store)
        current_label = next(
            (label for label, info in by_label.items() if info.name == current), None
        )
        options = list(by_label)

        chosen_label = st.selectbox(
            "Choose a client before fetching",
            options=options,
            index=options.index(current_label) if current_label else None,
            placeholder="Select a source...",
            key="source_selectbox",
        )

        if chosen_label:
            info = by_label[chosen_label]
            if info.available:
                set_source_name(store, info.name)
            else:
                st.warning(info.reason)

        st.divider()
        st.subheader("Active dataset")
        for line in dataset_summary_lines(get_active_dataset(store)):
            st.caption(line)
