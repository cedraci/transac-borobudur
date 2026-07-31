import pandas as pd
import numpy as np
import datetime
import portfolio_construction.portfolio_optimization as ptf_opt
import portfolio_construction.utils as ptf
from scipy.stats import linregress

"""
Rebalancing frequency codes used by `RebalancingCalendar.method`:

eom : end of month
eoq : end of quarter
eos : end of semester (Q2 and Q4)
eow : end of week
bim : bi-monthly (every other week)

Any other value falls back to end of year.
"""


def snap_to_trading_day(index, target, direction="forward"):
    """Map a calendar date onto a date that actually exists in `index`.

    A date picker happily returns a Saturday; the index only holds trading
    days. Looking such a date up directly used to raise IndexError.

    Args:
        index (pd.DatetimeIndex): the available trading dates
        target: the requested date, anything pd.Timestamp accepts
        direction (str): "forward" for the next available date, "backward" for
            the previous one

    Returns:
        pd.Timestamp: a date present in `index`
    """
    target = pd.Timestamp(target)

    if target in index:
        return target

    if direction == "backward":
        candidates = index[index <= target]
        if len(candidates) == 0:
            raise ValueError(
                f"{target:%Y-%m-%d} is outside the available history "
                f"({index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d})"
            )
        return candidates[-1]

    candidates = index[index >= target]
    if len(candidates) == 0:
        raise ValueError(
            f"{target:%Y-%m-%d} is outside the available history "
            f"({index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d})"
        )
    return candidates[0]


class RebalancingCalendar:
    def __init__(self, data, start_dt, end_dt, method, lookback=None):
        self.rebalCalendar = None
        self.method = method
        self.start_date = start_dt
        self.end_date = end_dt
        self.lookback = lookback
        self.all_dates = data.index

        self.define_calendar()

    def define_calendar(self):
        idx_start = np.where(self.all_dates == self.start_date)[0][0]

        if isinstance(self.lookback,(float, int)) and idx_start - self.lookback < 0.:
            print("not sufficient historical for the given lookback")
            return
        
        df_dates = self.all_dates.to_frame()
        df_dates['year'] = df_dates.index.year
        df_dates['quarter'] = df_dates.index.quarter
        df_dates['month'] = df_dates.index.month
        df_dates['week'] = df_dates.index.isocalendar().week
        df_dates['dow'] =  df_dates.index.weekday # 0 is Monday 7 is Sunday
        df_dates['bi_month'] = df_dates['week'].apply(lambda x: x % 2)
        
        if self.method == "eos":
            tmp = df_dates.groupby(["year","quarter"]).tail(1)
            self.rebalCalendar = tmp[(tmp['quarter'] == 2) | (tmp['quarter'] == 4)].index
        elif self.method == "eoq":
            self.rebalCalendar = df_dates.groupby(["year","quarter"]).tail(1).index
        elif self.method == "eom":
            self.rebalCalendar = df_dates.groupby(["year","month"]).tail(1).index
        elif self.method == "eow":
            self.rebalCalendar = df_dates.groupby(["year","month","week"]).tail(1).index
        elif self.method == "bim": # last day of the second week of each months
            tmp = df_dates.groupby(["year","month","week"]).tail(1)
            self.rebalCalendar = tmp[tmp['bi_month'] == 0].index
        else: # the last day of the year
            self.rebalCalendar = df_dates.groupby('year').tail().index

        self.rebalCalendar = self.rebalCalendar[(self.rebalCalendar >= self.start_date) \
            & (self.rebalCalendar <= self.end_date)]





class Backtest:
    def __init__(self):
        self.backtest_start_date = None
        self.backtest_end_date = None
        self.backtest_duration = None
        self.sequence_rebal = None
        self.dates_series = None
        self.strat = None
        self.start_date = None
        self.end_date = None
        self.daily_returns = None
        self.daily_prices = None
        self.lookback = None
        self.historical_portfolios = None
        self.selection_window = 250
        self.nb_securities = 6

    def initialize_parameters(self, data, start_date, end_date, lookback, method="eom"):
        """Prepare the backtest window, returns and rebalancing calendar.

        Args:
            data (pd.DataFrame): price history, one column per security
            start_date (str): "YYYY-MM-DD"; snapped forward to a trading day
            end_date (str): "YYYY-MM-DD"; snapped backward to a trading day
            lookback (int): observations required before the start date
            method (str): rebalancing frequency, see the module docstring
        """
        # Il faut rajouter un test sur l'existence de NA avant les traitements car il n'en faut pas.
        # Dates are snapped: a date picker returns calendar days, not trading days.
        start_ts = snap_to_trading_day(data.index, start_date, "forward")
        end_ts = snap_to_trading_day(data.index, end_date, "backward")

        if start_ts > end_ts:
            raise ValueError(
                f"start date {start_ts:%Y-%m-%d} is after end date {end_ts:%Y-%m-%d}"
            )

        idx_start_date = np.where(data.index == start_ts)[0][0]
        idx_end_date = np.where(data.index == end_ts)[0][0]

        if int(idx_start_date - lookback) < 0:
            raise ValueError(
                f"not enough history before {start_ts:%Y-%m-%d} for a lookback of "
                f"{lookback} observations (only {idx_start_date} available)"
            )

        daily_ret = data.pct_change().dropna()

        new = daily_ret.copy()
        new['Year'] = new.index.year
        new["Month"] = new.index.month
        new["Day"] = new.index.day

        # The rebalancing calendar drives the frequency; it used to be hardcoded
        # to month-ends here, leaving RebalancingCalendar unused.
        calendar = RebalancingCalendar(data, start_ts, end_ts, method)
        self.sequence_rebal = calendar.rebalCalendar
        if len(self.sequence_rebal) == 0 or self.sequence_rebal[0] != start_ts:
            self.sequence_rebal = self.sequence_rebal.insert(0, start_ts)

        self.rebalancing_method = method
        self.dates_series = data.iloc[idx_start_date:(idx_end_date + 1)].index
        self.start_date = start_ts.strftime("%Y-%m-%d")
        self.end_date = end_ts.strftime("%Y-%m-%d")
        self.daily_returns = new.drop(columns = ["Year","Month","Day"])
        self.daily_prices = data
        self.lookback = lookback

    def is_rebal_date(self, current_date):
        if current_date in self.sequence_rebal:
            return True
        else:
            return False

    def simulations(self, typeOpt, robust=False, bayes=False, stock_picking=False,
                    selection_window=None, nb_securities=None):
        self.backtest_start_date = datetime.datetime.now()
        # Do things
        portfolio = ptf.Portfolio()
        strat = []

        for i, dAy in enumerate(self.dates_series):
            if i == 0:
                w_star = self.target_weights(dAy, typeOpt, robust, bayes, stock_picking,
                                             selection_window, nb_securities)
                portfolio.save_current_portfolio(dAy, w_star)
                strat.append(np.sum(list(portfolio.portfolio_weights.values())))
            elif self.is_rebal_date(dAy):
                daily_perf = list(portfolio.portfolio_weights.values()) * (1 + self.daily_returns.loc[dAy, portfolio.portfolio_weights.keys()]) - list(portfolio.portfolio_weights.values())
                strat.append(strat[-1] * (1 + np.sum(daily_perf)))
                w_star = self.target_weights(dAy, typeOpt, robust, bayes, stock_picking,
                                             selection_window, nb_securities)
                portfolio.save_current_portfolio(dAy, w_star)
            else:
                new_weights = list(portfolio.portfolio_weights.values()) * (1 + self.daily_returns.loc[dAy, portfolio.portfolio_weights.keys()])
                daily_perf = new_weights - list(portfolio.portfolio_weights.values())
                portfolio.save_current_portfolio(dAy, dict(zip(w_star.keys(), np.round(new_weights, 5))))
                strat.append(strat[-1] * (1 + np.sum(daily_perf)))

        self.strat = pd.DataFrame(strat, index=self.dates_series, columns=["Strategy"])
        self.backtest_end_date = datetime.datetime.now()
        self.backtest_duration = self.backtest_end_date - self.backtest_start_date
        self.historical_portfolios = portfolio.historical_portfolios

    def target_weights(self, current_date, typeOpt, robust, bayes, stock_picking=False,
                       selection_window=None, nb_securities=None):
        idx_day = np.where(self.daily_returns.index == current_date)[0][0]
        #sub_data = self.daily_returns.iloc[(idx_day - self.lookback):idx_day]

        if stock_picking == True:
            # Both were hardcoded to 250/6, so a caller asking for a different
            # universe size was silently ignored.
            window = self.selection_window if selection_window is None else selection_window
            count = self.nb_securities if nb_securities is None else nb_securities
            count = min(count, self.daily_prices.shape[1])
            univ_idx, univ_names, mom_scores = universe_selection(
                self.daily_prices, current_date, window, count)
            sub_data = self.daily_returns.iloc[0:idx_day,univ_idx]
        else:
            sub_data = self.daily_returns.iloc[0:idx_day]

        # `robust` selects a shrunk covariance estimate; `bayes` (Bayes-Stein
        # shrinkage of expected returns) isn't wired into portfolio_optimization
        # yet, so it's accepted here but currently has no effect.
        w = ptf_opt.portfolio_optimization(
            sub_data, typeOpt, cov_mat="shrunked" if robust else "sample"
        )
        #w = ptf_opt.max_momentum_optimization(sub_data, mom_scores)
        #w = ptf_opt.longshort_momentum_optimization(sub_data, mom_scores)
        # portfolio_optimization already returns {column_name: weight}, keyed
        # exactly by univ_names - no need to re-zip.
        return w

def universe_selection(data, current_date, window, nb_securities):
    idx_day = np.where(data.index == current_date)[0][0]
    sub_data = data.iloc[(idx_day - window):idx_day]
    mom_list = []
    sec_score = dict()

    #if criteria == "Momentum":
    for sec in data.columns:
        sec_score = {
            'Security':sec,
            'Score':float(sub_data[sec].tail(window).rolling(window).apply(momentum, raw=False).dropna().values.item())
        }
        mom_list.append(sec_score)
    df_scores = pd.DataFrame(mom_list)
    selected_idx = df_scores.sort_values(by="Score", ascending=False).index[:nb_securities]
    selected_names = df_scores['Security'].iloc[selected_idx].values
    selected_scores = df_scores.sort_values(by="Score", ascending=False)["Score"][:nb_securities].values
    return selected_idx, selected_names, selected_scores

def universe_selection_v2(data, current_date, window, nb_securities):
    idx_day = np.where(data.index == current_date)[0][0]
    sub_data = data.iloc[(idx_day - window - 1):idx_day]
    all_mom = naive_momentum(sub_data, window, int(np.ceil(window * 0.2)))
    mom_list = []
    sec_score = dict()
    
    for sec in all_mom.columns:
        sec_score = {
            'Security':sec,
            'Score':all_mom[sec].tail(1).values
        }
        mom_list.append(sec_score)
    df_scores = pd.DataFrame(mom_list)
    selected_idx = df_scores.sort_values(by="Score", ascending=False).index[:nb_securities]
    selected_names = df_scores['Security'].iloc[selected_idx].values
    selected_scores = df_scores["Score"].iloc[selected_idx]
    return selected_idx, selected_names, selected_scores

def momentum(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    #return ((1 + slope) ** 252) * (rvalue ** 2)  # annualize slope and multiply by R^2
    return (np.power(np.exp(slope), 250) -1) * 100 *  (rvalue ** 2)

def naive_momentum(closes, long_period, short_period):
    long_term_mom = closes / closes.shift(long_period) - 1
    short_term_mom = closes / closes.shift(short_period) - 1
    long_term_mom = long_term_mom.dropna()
    short_term_mom = short_term_mom.dropna()
    mom = long_term_mom - short_term_mom[np.isin(short_term_mom.index, long_term_mom.index)]
    return mom