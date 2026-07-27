from portfolio_construction.utils import Portfolio


def test_portfolio_save_current_portfolio():
    p = Portfolio()
    p.save_current_portfolio("2024-01-31", {"A": 0.6, "B": 0.4})

    assert p.portfolio_date == "2024-01-31"
    assert p.portfolio_weights == {"A": 0.6, "B": 0.4}
    assert p.historical_portfolios == [{"date": "2024-01-31", "weights": {"A": 0.6, "B": 0.4}}]

    p.save_current_portfolio("2024-02-29", {"A": 0.5, "B": 0.5})
    assert len(p.historical_portfolios) == 2
    assert p.portfolio_weights == {"A": 0.5, "B": 0.5}
