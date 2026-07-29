"""Pure formatting helpers behind the sidebar."""

import pandas as pd

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.sidebar import dataset_summary_lines, source_option_label
from portfolio_ui.sources.registry import describe_sources


def _dataset():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 101.0], "MSFT.US": [400.0, 401.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, "sample", "eod_api")


def test_available_source_label_is_just_the_name():
    info = {i.name: i for i in describe_sources(token="k")}["eod_api"]
    assert source_option_label(info) == "eod_api"


def test_unavailable_source_label_states_the_reason():
    info = {i.name: i for i in describe_sources(token="")}["eod_api"]
    label = source_option_label(info)
    assert "eod_api" in label
    assert "EOD_API_KEY is not set" in label


def test_dataset_summary_lines_report_shape_and_provenance():
    lines = dataset_summary_lines(_dataset())
    assert any("sample" in line for line in lines)
    assert any("2 cols" in line for line in lines)
    assert any("eod_api" in line for line in lines)


def test_dataset_summary_lines_include_notes():
    frame = _dataset().prices
    ds = dataset_from_frame(frame, "sample", "async_eod", notes=("sliced client-side",))
    assert any("sliced client-side" in line for line in dataset_summary_lines(ds))


def test_dataset_summary_lines_handle_no_dataset():
    assert dataset_summary_lines(None) == ["No active dataset"]
