import json
import urllib.request 
import pandas as pd
import os
import calendar
import datetime

tok = os.environ.get('EOD_API_KEY', '')

"""
i.e fundamentals("MSFT.US", tok, filter="General::CurrencyName,General::ISIN")
"""
def fundamentals(symbol, token, *args, **kwargs):
    if kwargs:
        url = "https://eodhistoricaldata.com/api/fundamentals/%s?api_token=%s&order=d&fmt=json&%s=%s" % (symbol, token, list(kwargs.keys())[0], list(kwargs.values())[0])  
        #print(url)
    else:
         url = "https://eodhistoricaldata.com/api/fundamentals/" + symbol + "?api_token=" + token + "&order=d&fmt=json"
    
    response = urllib.request.urlopen(url) 
    data = json.loads(response.read())
    return data
    

def exchanges_list(token):
    url = "https://eodhistoricaldata.com/api/exchanges-list/?api_token=%s&fmt=json" % (token)
    response = urllib.request.urlopen(url) 
    data = json.loads(response.read())
    return data

"""
i.e index_constituents(tok, "FCHI.INDX")
"""
def index_constituents(token, ticker, return_tickers=False):
    url = "https://eodhistoricaldata.com/api/fundamentals/%s?api_token=%s" % (ticker, token)
    response = urllib.request.urlopen(url) 
    data = json.loads(response.read())

    if return_tickers == False:
        return data
    else:
        tickers = []
        for key in data['Components'].values():
            tickers.append(key['Code']+"."+key['Exchange'])
        
        return tickers

def ohlcv(token, ticker, as_df=True):
    url = "https://eodhistoricaldata.com/api/eod/%s?api_token=%s&fmt=json" % (ticker, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    if as_df:
        df = pd.DataFrame().from_dict(data)
        data = df
    return data

def dividends(token, ticker, as_df=True):
    url = "https://eodhistoricaldata.com/api/div/%s?api_token=%s&fmt=json" % (ticker, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    
    if as_df:
        df = pd.DataFrame().from_dict(data)
        data = df
    return data

def splits(token, ticker, as_df=True):
    url = "https://eodhistoricaldata.com/api/splits/%s?api_token=%s&fmt=json" % (ticker, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    
    if as_df:
        df = pd.DataFrame().from_dict(data)
        data = df
    return data


def adjusted_prices(tok, tickers):
    """ request historical adjusted prices from eod provider """
    tmp = []
    col_titles = []
    sec_nb = len(tickers) # needed for logging purpose.
    all = pd.DataFrame()

    for tic in tickers: 
        try:
            sec = ohlcv(tok, tic) # historical data extraction.
            sec_ts = sec["adjusted_close"] # keep only adjusted prices.
            sec_ts.index = pd.to_datetime(sec["date"])
            tmp.append(sec_ts)
            col_titles.append(tic) # append columns names when extraction succed.
        except:
            print('unable to extract {}'.format(tic))
            continue  

        if len(tmp) == 1:
            all = pd.DataFrame(tmp[0])
        else:
            all = pd.merge_asof(all, sec_ts, left_index=True, right_index=True, tolerance=pd.Timedelta('5 day'))

    all.columns = col_titles
    all.index = pd.to_datetime(all.index)
    #print('{}/{} extracted'.format(len(tmp), sec_nb))
    return all

def search_query(token, research, as_df=False):
    """ research for ticker """
    url = "https://eodhistoricaldata.com/api/search/%s?api_token=%s&limit=50" % (research, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    if as_df:
        df = pd.DataFrame().from_dict(data)
        data = df
    return data

def download_universe(tok, tickers):
    tmp = []
    for tic in tickers:
        sec = ohlcv(tok, tic)
        sec_ts = sec["adjusted_close"]
        sec_ts.index = sec["date"]
        tmp.append(sec_ts)
        
    all = pd.concat(tmp, axis=1, join="inner")
    all.columns = tickers
    all.index = pd.to_datetime(all.index)
    return all

def adjClose_atDate(token, ticker, strDate):
    """ request ticker's price for one date """
    url = "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&period=d&api_token=%s&fmt=json" % (ticker, strDate, strDate, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())[0]
    return data['adjusted_close']


def recursive_adjClose_atDate(token, ticker, strDate):
    """ request ticker's price for one date and fill unkwown dates with previous dates """
    # amélioration : loop until you get price  -> last known price before given date
    url = "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&period=d&api_token=%s&fmt=json" % (ticker, strDate, strDate, token)
    response = urllib.request.urlopen(url)

    try:
        data = json.loads(response.read())[0]
    except(IndexError):
        new_strDate = pd.to_datetime(strDate) + pd.offsets.Day(-1) 
        new_strDate = new_strDate.strftime("%Y-%m-%d")
        url = "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&period=d&api_token=%s&fmt=json" % (ticker, new_strDate, new_strDate, token)
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        print(data)
    return data['adjusted_close']

def last_prices(token, ticker):
    """ extract current / last price from eod historical data """
    date = datetime.datetime.utcnow()
    strDate = date.strftime("%Y-%m-%d")
    cur_unix_time = calendar.timegm(date.utctimetuple())

    # First try to extract price from current time (intraday)
    url = "https://eodhistoricaldata.com/api/intraday/%s?from=%s&to=%s&api_token=%s&interval=1m&fmt=json" % (ticker, cur_unix_time, cur_unix_time, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    # If the first try doesn't work, extract the last knwon price
    if not isinstance(data, float):
        data = recursive_adjClose_atDate(token, ticker, strDate)

    return data


def get_SovBond(token, ticker, strDate):
    """ extract close yield for a sovereign bond given iso country code and tenor """
    url = "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&api_token=%s&fmt=json" % (ticker, strDate, strDate, token)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())[0]
    return data['adjusted_close']

def last_prices_universe(token, tickers):
    """ extract real time prices (delayed 15 min) for a list of tickers """
    tmp = ("&s=" + ','.join(tickers[1:])) if len(tickers) > 1 else ""
    url = f"https://eodhistoricaldata.com/api/real-time/{tickers[0]}?api_token={token}&fmt=json{tmp}"

    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    if isinstance(data, dict):  # single-ticker requests return a bare object
        data = [data]

    df = pd.DataFrame(data)
    return df.set_index("code")["close"].astype(float)


def fixed_income_etf(tok, ticker):
    """ extract fixed income information from ETF_Data """
    fields = {
        "EffectiveDuration":"Duration",
        "ModifiedDuration":"Modified Duration",
        "EffectiveMaturity":"Maturity",
        "CreditQuality":"Rating",
        "Coupon":"Coupon",
        "Price":"Price",
        "YieldToMaturity":"YtM"
        }
    
    extract = dict()
    infos = fundamentals(ticker, tok, filter="ETF_Data::Fixed_Income")

    for val in infos.keys():
        extract.update({fields[val]:float(infos[val]["Relative_to_Category"])})

    df = pd.DataFrame().from_dict(extract, orient="index")
    df.columns = [ticker]
    return df


def macro_events(tok, start, end, cntry):
    """ function that allow to extract macro publication between two dates (limited to 1000 entries) """
    # macro_events(tok, "2015-01-01", "2022-12-31", "FR", "S&P Global Composite PMI")

    url = "https://eodhd.com/api/economic-events?&api_token=%s&country=%s&from=%s&to=%s&fmt=json&limit=1000" % (tok, cntry, start, end)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    df = pd.DataFrame().from_dict(data)
    df.index = pd.to_datetime(df.date)
    return df


def earnings_calendar(token: str, start_dt = None, end_dt = None) -> pd.DataFrame:
    """ function to extract all earnings calendar. Dates can be given in order to reduce the size of the request """
    url = f"https://eodhistoricaldata.com/api/calendar/earnings?api_token={token}&fmt=json"

    if start_dt:
        url += f"&from={start_dt}"

    if end_dt:
        url +=  f"&to={end_dt}"

    response = urllib.request.urlopen(url)
    data = json.loads(response.read())["earnings"]
    df = pd.DataFrame().from_dict(data)
    return df


def stock_historical_dividend_yield(tok, ticker):
    """Function for estimate the historical dividend yield for stock through years
    """
    historical_prices = ohlcv(tok, ticker, True)
    historical_prices["date"] = pd.to_datetime(historical_prices["date"])
    historical_dividends = dividends(tok, ticker, as_df=True)
    historical_dividends["date"] = pd.to_datetime(historical_dividends["date"])

    merged = pd.merge_asof(historical_dividends, historical_prices, left_on='date', right_on='date', tolerance=pd.Timedelta('5 day'))
    if historical_dividends["currency"].iloc[-1] == "GBP":
        merged["dividend_yield"] = (merged["value"]*100.)/merged["close"]
    else:
        merged["dividend_yield"] = merged["value"]/merged["close"]

    merged["year"] = merged["date"].dt.year

    return merged.groupby(["year"]).sum()["dividend_yield"]


def macro_indicators(token, country_code, indicator):
    url = f"https://eodhd.com/api/macro-indicator/{country_code}?indicator={indicator}&api_token={token}&fmt=json"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    df = pd.DataFrame().from_dict(data)
    return df

