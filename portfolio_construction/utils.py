class Portfolio:
    def __init__(self):
        self.portfolio_weights = []
        self.portfolio_date = None
        self.historical_portfolios = []

    def save_current_portfolio(self, date, weights):
        self.historical_portfolios.append({"date":date, "weights":weights}) # append daily portfolio with equity drift with date
        self.portfolio_weights = weights
        self.portfolio_date = date