import asyncio
import aiohttp
from aiolimiter import AsyncLimiter

import datetime
import time
import math
import itertools
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import platform

EOD_API_KEY = os.environ.get("EOD_API_KEY", "")


def _api_key() -> str:
    """Read the EOD API key at call time.

    Reading lazily (rather than at import) lets the UI import this module and
    report a missing key as a disabled action instead of crashing at startup.
    """
    key = os.environ.get("EOD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "EOD_API_KEY is not set - required to call EOD Historical Data"
        )
    return key

if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# from config import EOD_API_KEY

# EOD API limits : 1000 calls / min
# Intraday, technical, news : 1 requete = 5 calls, 200/min
# Source : https://eodhistoricaldata.com/financial-apis/api-limits/

# Parallelization settings
concurrent_count = 40
limiter = AsyncLimiter(20, 0.01)


def main():
    pass


### JSON url request
async def async_req_json(url, semaphore):

    async with aiohttp.ClientSession() as session:
        await semaphore.acquire()
        async with limiter:
            # print("Requesting ", url)
            async with session.get(url) as resp:
                content = await resp.json()
                semaphore.release()
                return content


###
### EOD REALTIME (DELAYED) MULTI TICKERS REQUEST
###
def multi_tickers_delayed_url(tickers):
    return f"https://eodhistoricaldata.com/api/real-time/{tickers[0]}?api_token={_api_key()}&fmt=json{('&s=' + ','.join(tickers[1:])) if len(tickers) > 1 else ''}"


async def async_get_realtime(tickers, semaphore):

    url = multi_tickers_delayed_url(tickers)
    data = await async_req_json(url, semaphore)

    if data:
        temp = []
        if isinstance(data, list):
            for elt in data:
                temp.append({"code": elt["code"], "close": elt["close"]})
        else:
            temp.append({"code": data["code"], "close": data["close"]})
        return temp
    else:
        print("No data found")
        return None


async def async_realtime_multi(tickers):
    semaphore = asyncio.Semaphore(value=concurrent_count)
    ticker_groups = [tickers[x : x + 15] for x in range(0, len(tickers), 15)]
    result = await asyncio.gather(
        *[async_get_realtime(group, semaphore) for group in ticker_groups]
    )

    return result


def get_realtime(tickers):
    """Get real time data for a list of tickers

    Args:
        tickers (list): list of tickers to fetch

    Returns:
        list of dictionaries {ticker:close}
    """

    result = asyncio.run(async_realtime_multi(tickers))
    result = list(itertools.chain.from_iterable(result))

    return result


###
### EOD END OF DAY SINGLE TICKER REQUEST (INCLUDING OFFSET TO GET LAST KNOWN PRICE)
###
def eod_url(ticker, query_date):
    return f'https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from={query_date.strftime("%Y-%m-%d")}&to={query_date.strftime("%Y-%m-%d")}&period=d'


def eod_url_offset(ticker, query_date, offset):
    offset_date = query_date - relativedelta(days=offset)
    return f'https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from={offset_date.strftime("%Y-%m-%d")}&to={query_date.strftime("%Y-%m-%d")}&period=d'


async def async_get_eod(ticker, date, semaphore):

    url = eod_url(ticker, date)
    data = await async_req_json(url, semaphore)
    if data:
        return {"code": ticker, "close": data[0]["close"]}
    else:
        url = eod_url_offset(ticker, date, 10)
        data = await async_req_json(url, semaphore)
        if data:
            return {"code": ticker, "close": data[-1]["close"]}
        else:
            print("No data found")
            return None


async def async_historical_multi(tickers, date):
    semaphore = asyncio.Semaphore(value=concurrent_count)
    if isinstance(tickers, (dict)):
        result = await asyncio.gather(
            *[
                async_get_eod(ticker, date, semaphore)
                for isin, ticker in tickers.items()
            ]
        )
    elif isinstance(tickers, (list)):
        result = await asyncio.gather(
            *[async_get_eod(ticker, date, semaphore) for ticker in tickers]
        )
    elif isinstance(tickers, (str)):
        result = await asyncio.gather(*[async_get_eod(tickers, date, semaphore)])
    return result


def get_historical(tickers, date):
    """Get close price for tickers, up to 10 days delay if no data

    Args:
        tickers (list): list of tickers (or a single ticker / dict of tickers in values)
        date (datetime): datetime for query

    Returns:
        list: list of dictionaries {ticker: close}
    """
    result = asyncio.run(async_historical_multi(tickers, date))

    return result


###
### EOD END OF DAY (FULL PRICE HISTORY) REQUEST
###
def full_history_url(ticker):
    return f"https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from=1990-01-01&to={datetime.datetime.today().strftime('%y-%m-%d')}&period=d"


async def async_get_full_history(ticker, semaphore):
    url = full_history_url(ticker)
    data = await async_req_json(url, semaphore)
    if data:
        tmp = pd.DataFrame(data)
        tmp.index = tmp["date"]
        tmp = tmp.drop("date", axis=1)
        tmp.index.names = ["Date"]
        tmp.index = pd.to_datetime(tmp.index)
        tmp.astype(float)
        return tmp.rename(columns={"adjusted_close": ticker})[ticker]
    else:
        print("no history for ", ticker)
        return None


async def async_full_history_multi(tickers):
    semaphore = asyncio.Semaphore(value=concurrent_count)

    result = await asyncio.gather(
        *[async_get_full_history(ticker, semaphore) for ticker in tickers]
    )

    return result


def get_full_history(tickers):
    """Get full end of day adjusted close history for tickers

    Args:
        tickers (list): list of tickers

    Returns:
        list: DataFrames
    """
    result = asyncio.run(async_full_history_multi(tickers))

    return result


###
### EOD SOVEREIGN BONDS SINGLE TICKER REQUEST
###
def sovereign_bonds_url(ticker, strDate):
    return (
        "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&api_token=%s&fmt=json"
        % (ticker, strDate, strDate, _api_key())
    )


async def async_sovereign_bond(ticker, strDate, semaphore):
    url = sovereign_bonds_url(ticker, strDate)
    data = await async_req_json(url, semaphore)
    if data:
        return {ticker: data[0]["adjusted_close"]}
    else:
        return None


def sovereign_bonds_tickers(countries, tenors):
    tickers = []
    for country in countries:
        for tenor in tenors:
            tickers.append(country + str(tenor) + "Y.GBOND")
    return tickers


async def async_sovereign_bonds_multi(countries, tenors, date):
    semaphore = asyncio.Semaphore(value=concurrent_count)
    tickers = sovereign_bonds_tickers(countries, tenors)
    tmp = dict()
    result = await asyncio.gather(
        *[async_sovereign_bond(ticker, date, semaphore) for ticker in tickers]
    )
    for elt in result:
        tmp.update(elt)
    return tmp


def sovereign_bonds(countries, tenors, strDate):
    """_summary_

    Args:
        countries (_type_): _description_
        tenors (_type_): _description_
        strDate (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = asyncio.run(async_sovereign_bonds_multi(countries, tenors, strDate))
    df_rates = pd.DataFrame().from_dict(result, orient="index")
    df_rates.columns = ["Rates"]
    # result = list(itertools.chain.from_iterable(result))
    return df_rates


if __name__ == "__main__":
    main()
