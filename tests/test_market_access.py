import numpy as np
import pandas as pd

import portfolio_construction.market_access as ma


def test_yf_dividend_adjustment_factor():
    factor = ma.yf_dividend_ajustment_factor(div=1.0, price=101.0)
    assert factor == (101.0 - 1.0) / 101.0


def test_dividend_adjustment_factor():
    factor = ma.dividend_ajustment_factor(div=1.0, price=100.0)
    assert factor == 1.0 / (1.0 + 1.0 / 100.0)


def test_split_adjustment_factor():
    assert ma.split_adjustement_factor(2.0) == 0.5


def test_adjust_for_corporates_actions_split_only():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 10.0, 5.0, 5.0, 5.0],
            "High": [11.0, 11.0, 5.5, 5.5, 5.5],
            "Low": [9.0, 9.0, 4.5, 4.5, 4.5],
            "Close": [10.0, 10.0, 5.0, 5.0, 5.0],
            "Dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 2.0, 0.0, 0.0],
        },
        index=dates,
    )
    adjusted = ma.adjust_for_corporates_actions(df, method="Yahoo", forModel=True, trace=False)
    # prices before the 2-for-1 split should be halved to be comparable post-split
    assert adjusted["Close"].iloc[0] == 5.0
    assert adjusted["Close"].iloc[1] == 5.0
    assert adjusted["Close"].iloc[2] == 5.0
    assert "Dividends" not in adjusted.columns
    assert "Stock Splits" not in adjusted.columns
