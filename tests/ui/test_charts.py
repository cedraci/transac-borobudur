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


def test_calendar_bar_figure_uses_the_last_column_for_bar_heights():
    from portfolio_ui.charts import calendar_bar_figure

    # calendar_performances returns a wide frame whose LAST column is the
    # period total; earlier columns are the individual months.
    calendar = pd.DataFrame(
        {"Jan": [0.01, 0.02], "Feb": [0.03, 0.04], "Total": [0.10, -0.05]},
        index=[2020, 2021],
    )
    fig = calendar_bar_figure(calendar)

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [2020, 2021]
    # would fail against .iloc[:, 0], which would give the Jan column
    assert list(fig.data[0].y) == [0.10, -0.05]


def test_simulation_fan_figure_summarizes_rather_than_drawing_every_path():
    from portfolio_ui.charts import simulation_fan_figure

    paths = pd.DataFrame(
        {f"path_{i}": [100.0, 100.0 + i, 100.0 + 2 * i] for i in range(500)}
    )
    fig = simulation_fan_figure(paths, title="Simulated NAV")
    # median + two band edges, never 500 traces
    assert len(fig.data) <= 3
    assert fig.layout.title.text == "Simulated NAV"


def test_weights_bar_figure_has_a_bar_per_ticker():
    from portfolio_ui.charts import weights_bar_figure

    weights = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    fig = weights_bar_figure(weights, title="Optimal weights")
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["AAA", "BBB", "CCC"]
    assert list(fig.data[0].y) == [0.5, 0.3, 0.2]
    assert fig.layout.title.text == "Optimal weights"


def test_frontier_figure_plots_volatility_against_return():
    from portfolio_ui.charts import frontier_figure

    frontier = pd.DataFrame({"Return": [0.05, 0.08], "Volatility": [0.10, 0.15]})
    fig = frontier_figure(frontier)
    assert len(fig.data) >= 1
    assert fig.layout.xaxis.title.text == "Volatility"
    assert fig.layout.yaxis.title.text == "Return"


def test_frontier_figure_can_mark_named_portfolios():
    from portfolio_ui.charts import frontier_figure

    frontier = pd.DataFrame({"Return": [0.05, 0.08], "Volatility": [0.10, 0.15]})
    points = {"minimum_variance": (0.09, 0.04)}
    fig = frontier_figure(frontier, points=points)
    names = {trace.name for trace in fig.data}
    assert "minimum_variance" in names


def test_correlation_heatmap_is_a_single_heatmap_trace():
    from portfolio_ui.charts import correlation_heatmap_figure

    corr = pd.DataFrame(
        [[1.0, 0.4], [0.4, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    fig = correlation_heatmap_figure(corr)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["AAA", "BBB"]


def test_weights_over_time_figure_stacks_one_trace_per_ticker():
    from portfolio_ui.charts import weights_over_time_figure

    idx = pd.bdate_range("2020-01-01", periods=3, name="Date")
    weights = pd.DataFrame({"AAA": [0.5, 0.6, 0.4], "BBB": [0.5, 0.4, 0.6]}, index=idx)
    fig = weights_over_time_figure(weights)
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"AAA", "BBB"}
    # stacked area, so the reader can see the mix rather than crossing lines
    assert all(t.stackgroup for t in fig.data)
