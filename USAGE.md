# Portfolio Construction Toolkit — Usage Guide

This is a personal quant-finance toolkit for building, testing, and analyzing
investment portfolios in Python. It covers the full loop: **get market data →
estimate risk/return → optimize weights → backtest → analyze performance**.

This guide assumes you're comfortable running Python scripts but may be new
to portfolio theory or to this specific codebase. Every section explains the
"why", not just the "how".

---

## 1. Project layout

```
.
├── pyproject.toml                     packaging/dependency metadata
├── USAGE.md                           this file
├── portfolio_construction/            the library — import this in your own code
│   ├── stats.py                       small shared math helpers (frequency, VaR)
│   ├── portfolio_analysis.py          performance & risk analytics (returns, vol, VaR, drawdowns...)
│   ├── portfolio_optimization.py      portfolio optimizers (min-variance, risk parity, etc.)
│   ├── portfolio_backtest.py          rebalancing calendars + a simple backtest engine
│   ├── utils.py                       the `Portfolio` bookkeeping class
│   ├── market_access.py               Yahoo Finance / AlphaVantage data + dividend/split adjustments
│   ├── eod_api.py                     EOD Historical Data client — fundamentals, one ticker at a time
│   └── async_eod.py                   EOD Historical Data client — prices, many tickers in parallel
└── examples/
    ├── asset_allocation_plot.py       runnable example: efficient frontier sweep + plot
    └── *.xlsx                         sample data used by the example
```

Everything under `portfolio_construction/` is meant to be **imported**, e.g.:

```python
import portfolio_construction.portfolio_analysis as pa
import portfolio_construction.portfolio_optimization as po
```

Everything under `examples/` is meant to be **run directly** as a script.

---

## 2. Installation

You need Python 3.10+.

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate        # on Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate    # on macOS/Linux

pip install -e .
```

`pip install -e .` reads `pyproject.toml` and installs the package in
"editable" mode — code changes in `portfolio_construction/` take effect
immediately without reinstalling, and `import portfolio_construction...`
works from anywhere on your machine, not just this folder.

If you'd rather not install the package, you can also just run scripts
**from the project root** — Python will find the `portfolio_construction`
folder as a local package automatically as long as your working directory
is this one.

### API key

`eod_api.py` and `async_eod.py` both talk to
[EOD Historical Data](https://eodhistoricaldata.com/) and require an API
token, read from the `EOD_API_KEY` environment variable:

```powershell
# PowerShell, current session only
$env:EOD_API_KEY = "your-token-here"

# persist across sessions (Windows)
setx EOD_API_KEY "your-token-here"
```

```bash
# bash
export EOD_API_KEY="your-token-here"
```

Both modules import fine without the key set. The key is read when a request is
actually built, so a missing key raises a `RuntimeError` naming `EOD_API_KEY`
at call time rather than at import.

---

## 3. Key concepts, explained simply

If you're new to portfolio construction, here's the vocabulary used
throughout the codebase:

- **Weights (`w`)**: what fraction of the portfolio's money is invested in
  each asset. Weights for a fully-invested, long-only portfolio sum to 1.0
  (e.g. `[0.6, 0.4]` = 60% in asset A, 40% in asset B).
- **Expected return (`mu` / `mean_ret`)**: the return you expect an asset to
  generate, usually estimated as the historical average return, annualized.
- **Covariance matrix (`S` / `cov`)**: a table of how every pair of assets'
  returns move together. The diagonal holds each asset's own variance
  (volatility squared). This matrix drives everything about *risk*.
- **Portfolio variance / volatility**: `w · S · w` (variance), or its square
  root (volatility) — how much the whole portfolio's value swings around.
  Lower is "safer" (less uncertain), not necessarily "better returns".
- **Sharpe ratio**: return earned per unit of risk taken —
  `(return - risk_free_rate) / volatility`. Higher is better; it lets you
  compare strategies with different risk levels on equal footing.
- **VaR (Value-at-Risk)**: "how much could I lose, at worst, X% of the
  time?" e.g. a 1-day 95% VaR of -2% means: on 95% of days, you won't lose
  more than 2%. Estimated here three ways: `historical` (from the empirical
  return distribution), `parametric` (assuming a normal/Student-t
  distribution), and `Monte Carlo` (simulating thousands of price paths).
- **Expected Shortfall / ES** (a.k.a. CVaR): the *average* loss in the worst
  cases beyond the VaR threshold — a more conservative risk measure than
  VaR because it looks past the cutoff instead of stopping at it.
- **Drawdown**: the drop from a portfolio's running peak value to its
  current value. "Max drawdown" is the worst peak-to-trough decline in the
  whole history — a common gut-check for "how bad could this get".
  `compute_drawdown_periods` in `portfolio_analysis.py` finds and ranks the
  worst drawdown *episodes* (not just the single worst number), including
  how long each took to recover.
- **Rebalancing**: periodically resetting portfolio weights back to target
  (e.g. every month-end) since they drift as prices move.
- **Optimization objective**: the mathematical goal an optimizer chases —
  e.g. "minimum variance" (safest possible), "maximum Sharpe" (best
  risk-adjusted return), "equal risk contribution" (each asset contributes
  equally to total risk, a.k.a. risk parity).

---

## 4. Module-by-module guide

### `portfolio_construction.stats`

Tiny shared helpers used by both the analysis and optimization modules —
you'll rarely call these directly, but they're worth knowing about:

- `time_series_frequence_inference(dates)` — looks at the gaps between
  dates in your index and guesses whether your data is Daily/Weekly/Monthly.
- `annualization_factor(freq)` — the multiplier used to annualize a
  volatility computed from that frequency (252 for daily, 12 for monthly...).
- `covariance_parametric_var(w, S, alpha, distrib)` — parametric VaR from a
  covariance matrix and weights alone (no return history needed).

### `portfolio_construction.portfolio_analysis`

The main analytics module — everything about **measuring** a strategy's
performance and risk, given a time series of prices or NAV.

```python
import pandas as pd
import portfolio_construction.portfolio_analysis as pa

prices = pd.read_excel("my_strategy.xlsx", index_col=0, parse_dates=True)["NAV"]

pa.annualized_return(prices)       # e.g. 0.072  (7.2% / year)
pa.annualized_volatility(prices)   # e.g. 0.114
pa.annualized_sharpe_ratio(prices, rf=0.02)  # risk-free rate = 2%
pa.maximum_drawdown(prices)        # e.g. -0.183 (-18.3%)
pa.calendar_performances(prices)   # month x year table of returns

# a full one-shot report across several strategies (columns of a DataFrame)
pa.stats_report(df_of_strategies, rf=0.0)

# top-10 worst drawdown *episodes*, with recovery dates
pa.drawdowns_table(prices, top_n=10, min_drawdown=0.02)

# Monte Carlo simulation of a single asset's future price paths
paths = pa.gbm_multiple_path(num=10_000, startprice=100, mu=0.07, sigma=0.15, days=252)

# Monte Carlo simulation of a *multi-asset portfolio* (correlated assets via Cholesky)
paths = pa.portfolio_path_cholesky(
    n_sims=10_000, start_price=100, weights=[0.6, 0.4],
    mu=[0.08, 0.03], cov_matrix=cov_matrix, days=252,
)
```

### `portfolio_construction.portfolio_optimization`

Given historical returns (or a covariance matrix), compute the optimal
weights for a chosen objective. The main entry point is
`portfolio_optimization(ret, typeOpt, bounds, cov_mat, target_return)`:

```python
import portfolio_construction.portfolio_optimization as po

# ret: DataFrame of daily returns, one column per asset
weights = po.portfolio_optimization(ret, typeOpt="minimum_variance")
weights = po.portfolio_optimization(ret, typeOpt="maximum_sharpe")
weights = po.portfolio_optimization(ret, typeOpt="equal_risk_contribution")
weights = po.portfolio_optimization(ret, typeOpt="most_diversified")
weights = po.portfolio_optimization(ret, typeOpt="minimum_VaR")
weights = po.portfolio_optimization(ret, typeOpt="maximum_return")
weights = po.portfolio_optimization(ret, typeOpt="inverse_volatility")
weights = po.portfolio_optimization(
    ret, typeOpt="mean_variance", target_return=0.06
)
```

`typeOpt` options:

| `typeOpt`                  | Goal |
|-----------------------------|------|
| `minimum_variance`           | Lowest possible portfolio volatility |
| `most_diversified`           | Maximize the diversification ratio |
| `maximum_sharpe`              | Best return per unit of risk |
| `minimum_VaR`                 | Smallest downside (parametric VaR) |
| `equal_risk_contribution`     | Risk parity — every asset contributes equally to total risk |
| `maximum_return`              | Highest expected return (ignores risk beyond the bounds) |
| `inverse_volatility`          | Simple heuristic: weight ∝ 1 / volatility, no optimizer needed |
| `mean_variance`               | Lowest variance that still hits `target_return` |

`bounds` is a list of `(min_weight, max_weight)` tuples, one per asset — use
it to cap or floor any position (e.g. no more than 40% in one asset).

`cov_mat` controls how the covariance matrix is estimated:
- `"sample"` (default) — plain historical covariance
- `"shrunked"` — `sklearn`'s `ShrunkCovariance`, pulls extreme estimates
  toward the average (more stable with limited history / many assets)
- `"gerber"` — the Gerber statistic, a robust co-movement measure less
  sensitive to outliers than plain correlation (see `gerber_correlation_matrix`)

Other standalone tools in this module:

```python
# Bayes-Stein shrinkage of expected returns (pulls extreme return estimates
# toward the cross-sectional average — reduces estimation error)
adjusted_mu = po.estimate_bayes_stein(returns_matrix)

# Find the max-return allocation that keeps a >=95% chance of not losing
# money over a given horizon (assumes lognormal/GBM returns)
result = po.optimize_capital_protection(mu, cov_matrix, duration=5.0, confidence=0.95)

# Stress-test a covariance matrix by pushing all correlations toward 1
# (simulates a "everything crashes together" panic scenario)
stressed_cov = po.get_stressed_covariance(cov_matrix, stress_factor=0.4)
```

### `portfolio_construction.portfolio_backtest`

Simulates how a strategy would have performed historically, rebalancing on
a schedule you choose.

```python
import pandas as pd
import portfolio_construction.portfolio_backtest as pb

prices = pd.read_excel("prices.xlsx", index_col=0, parse_dates=True)

bt = pb.Backtest()
bt.initialize_parameters(prices, start_date="2015-01-01", end_date="2023-12-31", lookback=250)
bt.simulations(typeOpt="minimum_variance")

bt.strat                    # daily NAV series of the simulated strategy
bt.historical_portfolios    # list of {"date": ..., "weights": ...} snapshots at each rebalance
```

`RebalancingCalendar` computes *when* rebalances happen:

```python
cal = pb.RebalancingCalendar(prices, "2015-01-01", "2023-12-31", method="eom", lookback=250)
cal.rebalCalendar   # DatetimeIndex of rebalance dates
```

`method` values: `"eom"` (end of month), `"eoq"` (end of quarter), `"eow"`
(end of week), `"eos"` (end of semester), `"bim"` (bi-monthly), or anything
else for end-of-year.

### `portfolio_construction.utils`

Just the `Portfolio` bookkeeping class — tracks the current weights and a
full history of every rebalance. `Backtest.simulations()` uses this
internally; you'd only touch it directly if writing your own backtest loop.

### `portfolio_construction.market_access`

Price/fundamental data from **Yahoo Finance** (via `yfinance`) and
**AlphaVantage** (free tier, rate-limited), plus dividend/split adjustment
logic:

```python
import portfolio_construction.market_access as ma

histo = ma.yahooFinance_historical_data("AAPL")     # full daily OHLCV history
snap = ma.yahooFinance_company_snapshot("AAPL")     # sector, country, currency...
clean = ma.clean_data(histo)                        # adjust for dividends/splits, drop raw columns
```

⚠️ The AlphaVantage functions have a hardcoded demo API key
(`KO6RI8AXUX0HDGC2`) — replace it with your own free key from
alphavantage.co before relying on this in anything real; the shared demo
key is heavily rate-limited.

### `portfolio_construction.eod_api` vs. `portfolio_construction.async_eod`

Both talk to the same **EOD Historical Data** API, but they cover different
use cases — use whichever matches what you're fetching:

| | `eod_api.py` | `async_eod.py` |
|---|---|---|
| Style | synchronous, one call at a time | asynchronous, many tickers in parallel (`asyncio` + `aiohttp`), rate-limited to the API's 1000 calls/min cap |
| Best for | **fundamentals & metadata**: dividends, splits, earnings calendar, macro events/indicators, index constituents, exchange lists, fixed-income ETF stats | **price time series at scale**: full history, single-date EOD (with automatic fallback to the last known price if a date has no data), real-time quotes, sovereign bond yields — for potentially hundreds of tickers at once |
| Example call | `eod_api.dividends(tok, "AAPL.US")` | `async_eod.get_full_history(["AAPL.US", "MSFT.US", ...])` |

Rule of thumb: reaching for a **single ticker's fundamentals/dividends/
earnings/macro data** → `eod_api.py`. Reaching for **prices across a large
universe of tickers, fast** → `async_eod.py`.

```python
import portfolio_construction.eod_api as eod
import os

tok = os.environ["EOD_API_KEY"]
divs = eod.dividends(tok, "AAPL.US")
earnings = eod.earnings_calendar(tok, "2024-01-01", "2024-12-31")
prices_df = eod.adjusted_prices(tok, ["AAPL.US", "MSFT.US", "GOOG.US"])
```

```python
import portfolio_construction.async_eod as async_eod
import datetime

# EOD_API_KEY is read automatically from the environment inside this module
histories = async_eod.get_full_history(["AAPL.US", "MSFT.US", "GOOG.US"])
today_prices = async_eod.get_historical(["AAPL.US", "MSFT.US"], datetime.datetime.today())
yields = async_eod.sovereign_bonds(["US", "FR", "DE"], [2, 5, 10], "2024-06-30")
```

---

## 5. Running the example

`examples/asset_allocation_plot.py` sweeps a range of target returns,
computes the minimum-variance allocation for each one (an "efficient
frontier"), and plots how the optimal asset mix shifts as you ask for more
return:

```bash
python examples/asset_allocation_plot.py
```

It reads `examples/Asset Class Historical Data.xlsx` (sheet `"rebased"`) —
swap in your own workbook by editing `DATA_FILE` at the top of the script.
A matplotlib window will pop up with a stacked-area chart of weights per
target return; the function also returns the two result DataFrames
(`df_weights`, `df_optim`) if you want to inspect or export them yourself
instead of just plotting.

---

## 6. Known limitations / things to double check before relying on this

- `market_access.py`'s AlphaVantage calls use a shared demo API key — get
  your own key for anything beyond a quick test.
- The optimizers use `scipy.optimize.minimize(method="SLSQP")`, which finds
  a *local* optimum — for objectives with many equivalent solutions (like
  `equal_risk_contribution`), reruns from different starting weights can
  land on slightly different but equally valid answers.
- `portfolio_optimization`'s `"gerber"` covariance option is
  computationally heavier (nested Python loops per asset pair) — expect it
  to be noticeably slower than `"sample"` or `"shrunked"` on large universes.
