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


def _nav():
    return pd.Series(
        [100.0, 105.0, 98.0, 110.0],
        index=pd.bdate_range("2020-01-01", periods=4, name="Date"),
        name="Portfolio",
    )


def test_nav_figure_has_one_trace_titled_as_asked():
    from portfolio_ui.charts import nav_figure

    fig = nav_figure(_nav(), title="Equity curve")
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Equity curve"
    assert fig.layout.yaxis.title.text == "NAV"


def test_drawdown_figure_is_a_filled_area_and_never_positive():
    from portfolio_ui.charts import drawdown_figure

    nav = _nav()
    drawdown = nav.div(nav.cummax()).sub(1.0)
    fig = drawdown_figure(drawdown)
    assert len(fig.data) == 1
    assert fig.data[0].fill == "tozeroy"
    assert max(fig.data[0].y) <= 0


def test_calendar_bar_figure_has_a_bar_per_period():
    from portfolio_ui.charts import calendar_bar_figure

    calendar = pd.DataFrame({"Performance": [0.10, -0.05]}, index=[2020, 2021])
    fig = calendar_bar_figure(calendar)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [2020, 2021]
