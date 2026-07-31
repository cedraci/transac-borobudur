"""Plotly figure builders. No streamlit import - pages do the rendering."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from portfolio_ui.dataset import ActiveDataset


def price_history_figure(dataset: ActiveDataset, rebased: bool = False) -> go.Figure:
    """One line per ticker, optionally rebased to 100 at the first observation."""
    frame = dataset.prices
    if rebased:
        frame = frame.div(frame.iloc[0]).mul(100.0)

    fig = go.Figure()
    for ticker in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame[ticker], mode="lines", name=ticker)
        )

    fig.update_layout(
        title=f"{dataset.name} - {dataset.summary()}",
        xaxis_title="Date",
        yaxis_title="Rebased to 100" if rebased else "Price",
        hovermode="x unified",
        legend_title="Ticker",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def latest_prices_figure(series: pd.Series, title: str) -> go.Figure:
    """A bar per ticker for point-in-time or latest prices."""
    fig = go.Figure(
        data=[go.Bar(x=list(series.index), y=list(series.values), name=title)]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Price",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def nav_figure(nav: pd.Series, title: str) -> go.Figure:
    """A single equity curve."""
    fig = go.Figure(
        data=[go.Scatter(x=nav.index, y=nav.values, mode="lines", name=nav.name or "NAV")]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def drawdown_figure(drawdown: pd.Series) -> go.Figure:
    """Underwater plot - the drop from the running peak, filled to zero."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
            )
        ]
    )
    fig.update_layout(
        title="Drawdown from running peak",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def calendar_bar_figure(calendar: pd.DataFrame) -> go.Figure:
    """One bar per calendar period, using the frame's last column."""
    values = calendar.iloc[:, -1]
    fig = go.Figure(data=[go.Bar(x=list(calendar.index), y=list(values), name="Return")])
    fig.update_layout(
        title="Calendar performance",
        xaxis_title="Period",
        yaxis_title="Return",
        yaxis_tickformat=".1%",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def simulation_fan_figure(paths: pd.DataFrame, title: str) -> go.Figure:
    """Median path with a 5th-95th percentile band.

    Deliberately summarizes: drawing 10 000 individual traces would make the
    page unusable and tell the reader less, not more.
    """
    steps = list(range(len(paths)))
    low = paths.quantile(0.05, axis=1)
    high = paths.quantile(0.95, axis=1)
    median = paths.median(axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=steps, y=high, mode="lines", name="95th percentile",
                   line=dict(width=0), showlegend=False)
    )
    fig.add_trace(
        go.Scatter(x=steps, y=low, mode="lines", name="5th-95th percentile",
                   line=dict(width=0), fill="tonexty")
    )
    fig.add_trace(go.Scatter(x=steps, y=median, mode="lines", name="Median"))
    fig.update_layout(
        title=title,
        xaxis_title="Days ahead",
        yaxis_title="Simulated value",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def weights_bar_figure(weights: pd.Series, title: str) -> go.Figure:
    """One bar per ticker, for a single allocation."""
    fig = go.Figure(
        data=[go.Bar(x=list(weights.index), y=list(weights.values), name="Weight")]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Weight",
        yaxis_tickformat=".1%",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def frontier_figure(frontier: pd.DataFrame, points: dict | None = None) -> go.Figure:
    """Risk on the x-axis, return on the y-axis - the conventional orientation.

    `points` marks named portfolios as (volatility, return) so a chosen
    objective can be seen against the frontier rather than in isolation.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatility"],
            y=frontier["Return"],
            mode="lines+markers",
            name="Efficient frontier",
        )
    )

    for name, (volatility, expected) in (points or {}).items():
        fig.add_trace(
            go.Scatter(
                x=[volatility], y=[expected], mode="markers",
                marker=dict(size=12, symbol="star"), name=name,
            )
        )

    fig.update_layout(
        title="Efficient frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        hovermode="closest",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def correlation_heatmap_figure(corr: pd.DataFrame) -> go.Figure:
    """Correlation matrix as a heatmap on a fixed -1..1 scale.

    Fixing the range matters: an auto-scaled correlation heatmap makes weakly
    correlated assets look strongly correlated.
    """
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                zmin=-1.0,
                zmax=1.0,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Correlation"),
            )
        ]
    )
    fig.update_layout(
        title="Correlation matrix",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def weights_over_time_figure(weights: pd.DataFrame) -> go.Figure:
    """Stacked area of the allocation through a backtest."""
    fig = go.Figure()
    for ticker in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[ticker],
                mode="lines",
                name=ticker,
                stackgroup="allocation",
            )
        )

    fig.update_layout(
        title="Allocation over time",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
