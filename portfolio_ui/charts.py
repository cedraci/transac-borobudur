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
