"""Chart builders are asserted on structure, never rendered."""

import pandas as pd

from portfolio_ui.charts import latest_prices_figure, price_history_figure
from portfolio_ui.dataset import dataset_from_frame


def _dataset():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 110.0], "MSFT.US": [400.0, 420.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, "sample", "local")


def test_price_history_figure_has_one_trace_per_ticker():
    fig = price_history_figure(_dataset())
    assert len(fig.data) == 2
    assert {trace.name for trace in fig.data} == {"AAPL.US", "MSFT.US"}


def test_price_history_figure_titles_the_axes():
    fig = price_history_figure(_dataset())
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Price"


def test_price_history_figure_rebases_to_100():
    fig = price_history_figure(_dataset(), rebased=True)
    for trace in fig.data:
        assert trace.y[0] == 100.0
    assert fig.layout.yaxis.title.text == "Rebased to 100"


def test_latest_prices_figure_is_a_bar_per_ticker():
    series = pd.Series({"AAPL.US": 190.0, "MSFT.US": 410.0}, dtype="float64")
    fig = latest_prices_figure(series, title="Latest")
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["AAPL.US", "MSFT.US"]
    assert fig.layout.title.text == "Latest"
