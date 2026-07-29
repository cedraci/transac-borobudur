"""Typed access to the session store.

Every function takes the store explicitly (st.session_state in the app, a dict
in tests) so this module never imports streamlit and stays unit-testable.
"""

from __future__ import annotations

from typing import MutableMapping

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.registry import SELECTABLE_SOURCES

SOURCE_KEY = "source_name"
ACTIVE_KEY = "active_dataset"
DERIVED_KEY = "derived_datasets"


def init_state(store: MutableMapping) -> None:
    """Seed missing keys without disturbing existing ones."""
    store.setdefault(SOURCE_KEY, None)
    store.setdefault(ACTIVE_KEY, None)
    store.setdefault(DERIVED_KEY, {})


def get_source_name(store: MutableMapping) -> str | None:
    return store.get(SOURCE_KEY)


def set_source_name(store: MutableMapping, name: str) -> None:
    """Select the source used for the *next* fetch.

    Deliberately does not touch the active dataset: spec section 5.1 keeps a
    fetched dataset valid, with its own provenance, after the selector moves.
    """
    if name not in SELECTABLE_SOURCES:
        raise KeyError(f"unknown source '{name}'")
    store[SOURCE_KEY] = name


def get_active_dataset(store: MutableMapping) -> ActiveDataset | None:
    return store.get(ACTIVE_KEY)


def set_active_dataset(store: MutableMapping, dataset: ActiveDataset) -> None:
    store[ACTIVE_KEY] = dataset


def has_active_dataset(store: MutableMapping) -> bool:
    return store.get(ACTIVE_KEY) is not None


def add_derived(store: MutableMapping, dataset: ActiveDataset) -> None:
    """Register a computed dataset (e.g. a backtest equity curve)."""
    store.setdefault(DERIVED_KEY, {})[dataset.name] = dataset


def list_derived(store: MutableMapping) -> list[ActiveDataset]:
    return list(store.get(DERIVED_KEY, {}).values())
