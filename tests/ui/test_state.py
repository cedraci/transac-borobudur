"""Session state accessors, exercised with a plain dict.

state.py takes the store as an argument precisely so it can be tested without
a running Streamlit script.
"""

import pandas as pd
import pytest

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.state import (
    add_derived,
    get_active_dataset,
    get_source_name,
    has_active_dataset,
    init_state,
    list_derived,
    set_active_dataset,
    set_source_name,
)


def _dataset(name="sample"):
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 101.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, name, "local")


def test_init_state_starts_with_no_source_selected():
    store = {}
    init_state(store)
    assert get_source_name(store) is None


def test_init_state_starts_with_no_active_dataset():
    store = {}
    init_state(store)
    assert not has_active_dataset(store)
    assert get_active_dataset(store) is None


def test_init_state_is_idempotent():
    store = {}
    init_state(store)
    set_source_name(store, "eod_api")
    init_state(store)
    assert get_source_name(store) == "eod_api"


def test_set_and_get_source_name():
    store = {}
    init_state(store)
    set_source_name(store, "async_eod")
    assert get_source_name(store) == "async_eod"


def test_set_source_rejects_unknown_name():
    store = {}
    init_state(store)
    with pytest.raises(KeyError):
        set_source_name(store, "nope")


def test_active_dataset_round_trip():
    store = {}
    init_state(store)
    ds = _dataset()
    set_active_dataset(store, ds)
    assert has_active_dataset(store)
    assert get_active_dataset(store) is ds


def test_changing_source_keeps_the_active_dataset():
    """Spec section 5.1: switching source does not invalidate the dataset."""
    store = {}
    init_state(store)
    set_source_name(store, "eod_api")
    ds = _dataset()
    set_active_dataset(store, ds)

    set_source_name(store, "async_eod")

    assert get_active_dataset(store) is ds
    assert get_active_dataset(store).source == "local"


def test_derived_datasets_are_listed_in_insertion_order():
    store = {}
    init_state(store)
    add_derived(store, _dataset("first"))
    add_derived(store, _dataset("second"))
    assert [d.name for d in list_derived(store)] == ["first", "second"]


def test_adding_a_derived_dataset_with_an_existing_name_replaces_it():
    store = {}
    init_state(store)
    add_derived(store, _dataset("same"))
    add_derived(store, _dataset("same"))
    assert len(list_derived(store)) == 1
