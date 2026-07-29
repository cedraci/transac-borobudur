import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os


def _require_alphavantage_key() -> str:
    """Read the AlphaVantage key at call time.

    Reading lazily keeps the yfinance-backed helpers in this module usable
    without an AlphaVantage account.
    """
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not set - required for AlphaVantage endpoints"
        )
    return key


def historical_adj_for_symbol(ticker):
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        "&symbol=" + ticker + "&outputsize=full&apikey=" + _require_alphavantage_key()
    )
    req = requests.get(url)
    brut = req.json()
    return brut

def historical_fx(currency1, currency2):
    url = (
        "https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=" + currency1
        + "&to_symbol=" + currency2
        + "&outputsize=full&apikey=" + _require_alphavantage_key()
    )
    req = requests.get(url)
    brut = req.json()
    df_data = pd.DataFrame(brut["Time Series FX (Daily)"])
    df_data = df_data.T
    df_data = df_data.astype('float64')
    df_data = df_data[::-1]
    dates = pd.to_datetime(df_data.index)
    df_data.index = dates
    df_data.columns = ["Open","High","Low","Close"]
    return df_data

def alphaVantage_historical_data(ticker):
    # IMPORTANT : seul le close est ajusté, il faut le traitement pour les autres colonnes.
    # --> il suffit de reprendre le code que j'ai déjà créé et l'adapter.
    json_data = historical_adj_for_symbol(ticker)
    df_data = pd.DataFrame(json_data["Time Series (Daily)"])
    df_data = df_data.T
    df_data = df_data.astype('float64')
    df_data = df_data[::-1]
    dates = pd.to_datetime(df_data.index)
    df_data.index = dates
    df_data.columns = ["Open","High","Low","Close","Adj Close","Volume","Dividends","Stock Splits"]
    return df_data

# Semble fonctionner uniquement pour les valeurs cotees aux US.
def alphaVantage_company_snapshot(ticker):
    url = (
        "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + ticker
        + "&apikey=" + _require_alphavantage_key()
    )
    req = requests.get(url)
    brut = req.json()
    return brut 

def yahooFinance_company_snapshot(ticker):
    tic = yf.Ticker(ticker)

    infos = tic.info
    snapshot = {"Sector":infos["sector"],
                "Industry":infos["industry"],
                "Country":infos["country"],
                "Short Name":infos["shortName"],
                "Long Name":infos["longName"],
                "Currency":infos["currency"],
                "5yrAvgDivYield":infos["fiveYearAvgDividendYield"],
                "Symbol":infos["symbol"]
                }

    return pd.DataFrame(snapshot.values(), index=snapshot.keys(), columns=[snapshot["Symbol"]])

def yahooFinance_historical_data(ticker):
    tic = yf.Ticker(ticker)
    corporates_actions = tic.actions # Get corporates actions

    if corporates_actions.shape[0] > 0: # if corporate actions exist, more things to do
        print("some corporates actions to endle") # But capitalized etf shouldn't have corp actions

    # Download data from yahoo finance web site
    histo = tic.history(period="max", interval="1d", auto_adjust=False)
    return histo

def yf_dividend_ajustment_factor(div, price):
    return (price - div)/price

def dividend_ajustment_factor(div, price):
    return 1. / (1. + div / price)

def split_adjustement_factor(split):
    return 1. / split

def clean_data(data):
    new = data.copy()
    
    nb_dividends = new.Dividends[new.Dividends != 0.].shape[0]
    nb_splits = new[new['Stock Splits'] != 0.].shape[0]
    nb_corporates_actions = nb_dividends + nb_splits

    if nb_corporates_actions > 0.:
        print("corporate actions treatment")
        new = adjust_for_corporates_actions(new, "Yahoo", forModel=False, trace=True)

    new = new.drop(columns=["Adj Close","Dividends","Stock Splits"]) # adapted since end of 2021 !
    #new = new.drop(columns=["Dividends","Stock Splits"])
    return new

def adjust_for_corporates_actions(df, method, forModel=False, trace=False):
    new_df = df.copy()
    dividends_series = new_df.Dividends[new_df.Dividends != 0.]
    splits_series = new_df[new_df['Stock Splits'] != 0.]['Stock Splits'] # updated end of 2021
    coef_vector = np.ones(new_df.shape[0])
    
    # Dividends treatment
    if dividends_series.shape[0] > 0:
        for i in range(0, dividends_series.shape[0]):
            idx = np.where(new_df.index == dividends_series.index[i])[0][0] # re use index for coef_vector
            
            if method == "Yahoo":
                coef = np.asarray(yf_dividend_ajustment_factor(dividends_series.iloc[i], new_df.Close.iloc[idx-1]))
                coef_vector = [float(val * coef) if a < (idx-1)  else val for a, val in enumerate(coef_vector)]
            else:
                coef = np.asarray(dividend_ajustment_factor(dividends_series.iloc[i], new_df.Close.iloc[idx-1]))
                coef_vector = [float(val * coef) if a < idx  else val for a, val in enumerate(coef_vector)]

    # Split treatments
    if splits_series.shape[0] > 0:
        for i in range(0, splits_series.shape[0]):
            idx = np.where(new_df.index == splits_series.index[i])[0][0]
            coef = split_adjustement_factor(splits_series.iloc[i])
            coef_vector = [float(val * coef) if a < idx  else val for a, val in enumerate(coef_vector)]

    adjusted = new_df.copy()
    adjusted['Open'] = new_df['Open'] * coef_vector
    adjusted['High'] = new_df['High'] * coef_vector
    adjusted['Low'] = new_df['Low'] * coef_vector
    adjusted['Close'] = new_df['Close'] * coef_vector
    
    if forModel == True:
        adjusted = adjusted.drop(['Dividends', 'Stock Splits'], axis=1)

    if trace == True:
        if dividends_series.shape[0] > 0.:
            print('%s dividends treated'%(dividends_series.shape[0]))
        if splits_series.shape[0] > 0.:
            print('%s stock splits treated'%(splits_series.shape[0]))
    return adjusted