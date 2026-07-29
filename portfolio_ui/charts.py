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
