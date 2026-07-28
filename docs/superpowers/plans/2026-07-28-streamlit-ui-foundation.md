# Streamlit UI Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation of the Streamlit research UI — core client fixes, a tested source adapter over `async_eod`/`eod_api`/`market_access`/local files, and a runnable app whose Data page can fetch, save and reload a shared price dataset.

**Architecture:** A new top-level `portfolio_ui` package depends one-way on `portfolio_construction`. All real logic lives in plain-Python modules with no Streamlit import (`sources/`, `dataset.py`, `state.py`, `charts.py`) and is unit-tested; Streamlit-touching code (`app.py`, `cache.py`, `pages/`) is thin wiring. A `PriceSource` protocol normalizes four capabilities across four sources so the rest of the app never knows which client ran.

**Tech Stack:** Python ≥3.10, pandas, Streamlit, Plotly, pyarrow (parquet), pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-streamlit-ui-design.md`. This plan implements build-order slices 1–3 of spec §10.

## Global Constraints

- Python floor is `>=3.10` (`pyproject.toml`). **Do not use `enum.StrEnum`** (3.11+) — use `class X(str, Enum)`.
- `portfolio_construction` must stay importable and usable with neither `streamlit` nor `plotly` installed. Nothing under `portfolio_construction/` may import either.
- `portfolio_ui/sources/`, `dataset.py`, `state.py`, `charts.py`, `guards.py` must **not** import `streamlit`. Only `app.py`, `sidebar.py`, `cache.py` and `views/` may.
- Page modules live in `portfolio_ui/views/`, **not** `portfolio_ui/views/`. Streamlit's legacy multipage discovery treats a `pages/` directory beside the entrypoint script as auto-registered pages, which would conflict with the explicit `st.navigation` list.
- Tests must never make network calls. Stub `urllib.request.urlopen` / `aiohttp` or inject fake clients.
- Run tests with the project venv: `.venv/Scripts/python.exe -m pytest`.
- Normalized price frame contract (spec §4.2), enforced everywhere: wide `DataFrame`, ascending `DatetimeIndex` named `Date`, `float64` values, one column per ticker named exactly as requested.
- Environment variables: `EOD_API_KEY` (both EOD sources), `ALPHAVANTAGE_API_KEY` (market_access AlphaVantage endpoints).
- Existing test suite must stay green after every task.

---

### Task 1: Make the EOD clients import-safe without a key

Spec §6.5 requires a missing key to disable a source's actions, not crash the app. Today both modules read the key at import and raise `KeyError`.

**Files:**
- Modify: `portfolio_construction/eod_api.py:8`
- Modify: `portfolio_construction/async_eod.py:14`, and the five URL builders at lines 51, 102, 107, 163, 212
- Modify: `USAGE.md` (the "API key" section, ~lines 86-88)
- Modify: `tests/test_data_clients_import.py:1-4` (docstring)
- Test: `tests/test_eod_key_handling.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `portfolio_construction.async_eod._api_key() -> str` (raises `RuntimeError` when unset). Module attributes `async_eod.EOD_API_KEY: str` and `eod_api.tok: str` remain, defaulting to `""`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eod_key_handling.py`:

```python
"""Both EOD clients must import without EOD_API_KEY set, and only fail when a
call actually needs the key (spec section 6.5)."""

import importlib

import pytest


def test_async_eod_imports_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    assert hasattr(module, "get_full_history")


def test_eod_api_imports_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.eod_api")
    importlib.reload(module)
    assert hasattr(module, "adjusted_prices")


def test_async_eod_url_builder_raises_without_key(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    with pytest.raises(RuntimeError, match="EOD_API_KEY"):
        module.full_history_url("AAPL.US")


def test_async_eod_url_builder_uses_key_when_present(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "unit-test-key")
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    assert "api_token=unit-test-key" in module.full_history_url("AAPL.US")


def test_api_key_read_at_call_time_not_import_time(monkeypatch):
    monkeypatch.delenv("EOD_API_KEY", raising=False)
    module = importlib.import_module("portfolio_construction.async_eod")
    importlib.reload(module)
    monkeypatch.setenv("EOD_API_KEY", "set-after-import")
    assert module._api_key() == "set-after-import"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eod_key_handling.py -v`
Expected: FAIL — `KeyError: 'EOD_API_KEY'` during reload, and `AttributeError: module has no attribute '_api_key'`.

- [ ] **Step 3: Make `async_eod` read the key lazily**

In `portfolio_construction/async_eod.py`, replace line 14:

```python
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
```

Then replace every use of the `EOD_API_KEY` constant inside the five URL builders with `_api_key()`:

```python
def multi_tickers_delayed_url(tickers):
    return f"https://eodhistoricaldata.com/api/real-time/{tickers[0]}?api_token={_api_key()}&fmt=json{('&s=' + ','.join(tickers[1:])) if len(tickers) > 1 else ''}"


def eod_url(ticker, query_date):
    return f'https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from={query_date.strftime("%Y-%m-%d")}&to={query_date.strftime("%Y-%m-%d")}&period=d'


def eod_url_offset(ticker, query_date, offset):
    offset_date = query_date - relativedelta(days=offset)
    return f'https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from={offset_date.strftime("%Y-%m-%d")}&to={query_date.strftime("%Y-%m-%d")}&period=d'


def full_history_url(ticker):
    return f"https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from=1990-01-01&to={datetime.datetime.today().strftime('%y-%m-%d')}&period=d"


def sovereign_bonds_url(ticker, strDate):
    return (
        "https://eodhistoricaldata.com/api/eod/%s?from=%s&to=%s&api_token=%s&fmt=json"
        % (ticker, strDate, strDate, _api_key())
    )
```

Leave the `%y-%m-%d` bug in `full_history_url` alone — Task 2 fixes it with its own regression test.

- [ ] **Step 4: Make `eod_api` read the key lazily**

In `portfolio_construction/eod_api.py`, replace line 8:

```python
tok = os.environ.get("EOD_API_KEY", "")
```

Every public function in this module already takes `token` as an explicit argument, so no other change is needed here.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eod_key_handling.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the full suite for regressions**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Update the documentation that states the old behaviour**

In `USAGE.md`, replace the paragraph beginning "If `EOD_API_KEY` isn't set, importing `async_eod.py` will raise a `KeyError`" with:

```markdown
Both modules import fine without the key set. The key is read when a request is
actually built, so a missing key raises a `RuntimeError` naming `EOD_API_KEY`
at call time rather than at import.
```

In `tests/test_data_clients_import.py`, replace the module docstring with:

```python
"""Smoke tests: importing eod_api.py and async_eod.py must not blow up, and no
real network calls are made here. Key handling itself is covered by
tests/test_eod_key_handling.py.
"""
```

- [ ] **Step 8: Commit**

```bash
git add portfolio_construction/eod_api.py portfolio_construction/async_eod.py USAGE.md tests/test_eod_key_handling.py tests/test_data_clients_import.py
git commit -m "fix: read EOD_API_KEY at call time so clients import without it"
```

---

### Task 2: Fix the four broken client code paths

Spec §7.1. Each fix gets a regression test.

**Files:**
- Modify: `portfolio_construction/async_eod.py` (`full_history_url`, `async_sovereign_bonds_multi`, `sovereign_bonds`)
- Modify: `portfolio_construction/eod_api.py` (`last_prices_universe`, `last_prices`)
- Test: `tests/test_eod_client_fixes.py`

**Interfaces:**
- Consumes: `async_eod._api_key()` from Task 1.
- Produces:
  - `async_eod._assemble_sovereign_rates(results: list[dict | None]) -> pd.DataFrame` — index = bond ticker, single `Rates` column, empty frame with a `Rates` column when nothing usable.
  - `eod_api.last_prices_universe(token, tickers) -> pd.Series` — **indexed by ticker code**, `float64`. (Spec §7.1 says "return the `close` column"; indexing by `code` is required so the adapter's `latest` capability can map values back to tickers.)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eod_client_fixes.py`:

```python
"""Regression tests for the four broken client code paths (spec section 7.1)."""

import json
import re

import pandas as pd
import pytest

import portfolio_construction.async_eod as async_eod
import portfolio_construction.eod_api as eod_api


class _FakeResponse:
    """Stands in for the object urllib.request.urlopen returns."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


def test_full_history_url_uses_four_digit_year(monkeypatch):
    monkeypatch.setenv("EOD_API_KEY", "unit-test-key")
    url = async_eod.full_history_url("AAPL.US")
    to_value = url.split("&to=")[1].split("&")[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", to_value), to_value


def test_assemble_sovereign_rates_skips_none_entries():
    results = [{"US10Y.GBOND": 4.25}, None, {"FR10Y.GBOND": 3.10}]
    df = async_eod._assemble_sovereign_rates(results)
    assert list(df.columns) == ["Rates"]
    assert df.loc["US10Y.GBOND", "Rates"] == 4.25
    assert df.loc["FR10Y.GBOND", "Rates"] == 3.10
    assert "None" not in df.index


def test_assemble_sovereign_rates_all_none_returns_empty_frame():
    df = async_eod._assemble_sovereign_rates([None, None])
    assert list(df.columns) == ["Rates"]
    assert df.empty


def test_last_prices_universe_returns_series_indexed_by_code(monkeypatch):
    payload = [
        {"code": "AAPL.US", "close": 190.5},
        {"code": "MSFT.US", "close": 410.25},
    ]
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    result = eod_api.last_prices_universe("dummy-token", ["AAPL.US", "MSFT.US"])
    assert isinstance(result, pd.Series)
    assert result["AAPL.US"] == 190.5
    assert result["MSFT.US"] == 410.25


def test_last_prices_universe_handles_single_ticker_dict_payload(monkeypatch):
    payload = {"code": "AAPL.US", "close": 190.5}
    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    result = eod_api.last_prices_universe("dummy-token", ["AAPL.US"])
    assert result["AAPL.US"] == 190.5


def test_last_prices_passes_its_own_token_argument(monkeypatch):
    captured = {}

    def fake_recursive(token, ticker, strDate):
        captured["token"] = token
        return 123.0

    monkeypatch.setattr(
        eod_api.urllib.request, "urlopen", lambda url: _FakeResponse([])
    )
    monkeypatch.setattr(eod_api, "recursive_adjClose_atDate", fake_recursive)
    eod_api.last_prices("token-from-argument", "AAPL.US")
    assert captured["token"] == "token-from-argument"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eod_client_fixes.py -v`
Expected: FAIL — `to=26-07-28` fails the year regex, `_assemble_sovereign_rates` does not exist, `last_prices_universe` raises `TypeError`, and the token assertion fails.

- [ ] **Step 3: Fix the `%y` year format**

In `portfolio_construction/async_eod.py`, `full_history_url`:

```python
def full_history_url(ticker):
    return f"https://eodhistoricaldata.com/api/eod/{ticker}?api_token={_api_key()}&fmt=json&from=1990-01-01&to={datetime.datetime.today().strftime('%Y-%m-%d')}&period=d"
```

- [ ] **Step 4: Extract and fix sovereign rate assembly**

In `portfolio_construction/async_eod.py`, replace `async_sovereign_bonds_multi` and `sovereign_bonds` with:

```python
def _assemble_sovereign_rates(results):
    """Build the rates frame from per-ticker results, skipping missing data.

    async_sovereign_bond returns None when the API has no data for a ticker,
    so the results list may contain None entries.
    """
    rates = {}
    for elt in results:
        if elt:
            rates.update(elt)

    if not rates:
        return pd.DataFrame(columns=["Rates"])

    df_rates = pd.DataFrame.from_dict(rates, orient="index")
    df_rates.columns = ["Rates"]
    return df_rates


async def async_sovereign_bonds_multi(countries, tenors, date):
    semaphore = asyncio.Semaphore(value=concurrent_count)
    tickers = sovereign_bonds_tickers(countries, tenors)
    return await asyncio.gather(
        *[async_sovereign_bond(ticker, date, semaphore) for ticker in tickers]
    )


def sovereign_bonds(countries, tenors, strDate):
    """Get close yields for sovereign bonds.

    Args:
        countries (list): ISO country codes, e.g. ["US", "FR"]
        tenors (list): tenors in years, e.g. [5, 10]
        strDate (str): query date as "YYYY-MM-DD"

    Returns:
        pd.DataFrame: indexed by bond ticker with a single "Rates" column.
    """
    results = asyncio.run(async_sovereign_bonds_multi(countries, tenors, strDate))
    return _assemble_sovereign_rates(results)
```

- [ ] **Step 5: Fix `last_prices_universe` and `last_prices`**

In `portfolio_construction/eod_api.py`:

```python
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
```

And in `last_prices`, line 169, pass the function's own parameter:

```python
        data = recursive_adjClose_atDate(token, ticker, strDate)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eod_client_fixes.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add portfolio_construction/async_eod.py portfolio_construction/eod_api.py tests/test_eod_client_fixes.py
git commit -m "fix: repair four broken EOD client code paths"
```

---

### Task 3: Remove the hardcoded AlphaVantage key and correct the backtest docstring

Spec §7.4 and §7.5.

**Files:**
- Modify: `portfolio_construction/market_access.py:6-13`, `:39-43`
- Modify: `portfolio_construction/portfolio_backtest.py:8-16` (module docstring)
- Modify: `USAGE.md` (API key section)
- Test: `tests/test_market_access_key.py`

**Interfaces:**
- Produces: `market_access._require_alphavantage_key() -> str`, raising `RuntimeError` when `ALPHAVANTAGE_API_KEY` is unset.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_market_access_key.py`:

```python
"""The AlphaVantage key must come from the environment, never from source."""

import inspect

import pytest

import portfolio_construction.market_access as ma


def test_no_hardcoded_key_in_source():
    source = inspect.getsource(ma)
    assert "KO6RI8AXUX0HDGC2" not in source


def test_require_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        ma._require_alphavantage_key()


def test_require_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test-key")
    assert ma._require_alphavantage_key() == "av-test-key"


def test_historical_fx_url_carries_env_key(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test-key")
    captured = {}

    class _FakeReq:
        def json(self):
            return {"Time Series FX (Daily)": {}}

    def fake_get(url):
        captured["url"] = url
        return _FakeReq()

    monkeypatch.setattr(ma.requests, "get", fake_get)
    with pytest.raises(Exception):
        # empty payload makes the downstream frame handling fail; we only care
        # that the key reached the URL
        ma.historical_fx("EUR", "USD")
    assert "apikey=av-test-key" in captured["url"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_market_access_key.py -v`
Expected: FAIL — hardcoded key still present, `_require_alphavantage_key` does not exist.

- [ ] **Step 3: Replace the hardcoded key**

In `portfolio_construction/market_access.py`, add after the imports:

```python
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
```

Then rewrite the three functions that embedded the key:

```python
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


def alphaVantage_company_snapshot(ticker):
    url = (
        "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + ticker
        + "&apikey=" + _require_alphavantage_key()
    )
    req = requests.get(url)
    brut = req.json()
    return brut
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_market_access_key.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Correct the rebalancing-code docstring**

In `portfolio_construction/portfolio_backtest.py`, replace the module docstring (lines 8-16) with the codes `define_calendar` actually implements:

```python
"""
Rebalancing frequency codes used by `RebalancingCalendar.method`:

eom : end of month
eoq : end of quarter
eos : end of semester (Q2 and Q4)
eow : end of week
bim : bi-monthly (every other week)

Any other value falls back to end of year.
"""
```

- [ ] **Step 6: Document the new environment variable**

In `USAGE.md`, in the "API key" section, add after the EOD block:

````markdown
`market_access.py` uses AlphaVantage for its `historical_fx`,
`alphaVantage_historical_data` and `alphaVantage_company_snapshot` helpers, and
reads `ALPHAVANTAGE_API_KEY` the same way:

```powershell
$env:ALPHAVANTAGE_API_KEY = "your-key-here"
```

The yfinance-backed helpers in that module need no key.
````

- [ ] **Step 7: Run the full suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass.

```bash
git add portfolio_construction/market_access.py portfolio_construction/portfolio_backtest.py USAGE.md tests/test_market_access_key.py
git commit -m "fix: read AlphaVantage key from env, correct rebalancing docstring"
```

**Note for the reviewer:** the key is still in git history. Rotating it at AlphaVantage is recommended but is the user's call and is out of scope here.

---

### Task 4: `portfolio_ui` package skeleton, packaging, and the source protocol

**Files:**
- Create: `portfolio_ui/__init__.py`
- Create: `portfolio_ui/sources/__init__.py`
- Create: `portfolio_ui/sources/base.py`
- Create: `tests/ui/__init__.py`
- Create: `tests/ui/test_base.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Produces (all importable from `portfolio_ui.sources.base`):
  - `Capability(str, Enum)` with members `PRICE_HISTORY`, `CLOSE_AT`, `LATEST`, `SOVEREIGN` (values `"price_history"`, `"close_at"`, `"latest"`, `"sovereign_yields"`).
  - Exceptions `SourceError(RuntimeError)`, `SourceUnavailable`, `TickerNotFound`, `UpstreamError`, `CapabilityNotSupported` — all subclasses of `SourceError`.
  - `PriceSource` Protocol with `name: str`, `capabilities: frozenset[Capability]`, and methods `price_history(tickers, start, end)`, `close_at(tickers, on)`, `latest(tickers)`, `sovereign_yields(countries, tenors, on)`.
  - `BaseSource` — concrete base whose four methods raise `CapabilityNotSupported`; also provides `supports(capability) -> bool` and `is_available() -> bool` (default `True`).
  - `normalize_price_frame(frame, tickers, start=None, end=None) -> pd.DataFrame`
  - `normalize_price_series(values, tickers) -> pd.Series`

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/__init__.py` (empty file), then `tests/ui/test_base.py`:

```python
"""The normalization contract every source must satisfy (spec section 4.2)."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    CapabilityNotSupported,
    normalize_price_frame,
    normalize_price_series,
)


def _raw_frame():
    # deliberately unsorted, string index, object dtype, extra column
    return pd.DataFrame(
        {
            "AAPL.US": ["191.0", "190.0", "192.0"],
            "MSFT.US": ["410.0", "409.0", "411.0"],
            "IGNORED.US": ["1.0", "2.0", "3.0"],
        },
        index=["2024-01-03", "2024-01-02", "2024-01-04"],
    )


def test_normalize_price_frame_sorts_ascending():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert out.index.is_monotonic_increasing


def test_normalize_price_frame_uses_datetime_index_named_date():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"


def test_normalize_price_frame_casts_to_float():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "MSFT.US"])
    assert all(dtype == "float64" for dtype in out.dtypes)


def test_normalize_price_frame_keeps_requested_columns_in_order():
    out = normalize_price_frame(_raw_frame(), ["MSFT.US", "AAPL.US"])
    assert list(out.columns) == ["MSFT.US", "AAPL.US"]


def test_normalize_price_frame_drops_missing_tickers_silently():
    out = normalize_price_frame(_raw_frame(), ["AAPL.US", "NOPE.US"])
    assert list(out.columns) == ["AAPL.US"]


def test_normalize_price_frame_slices_date_range_inclusively():
    out = normalize_price_frame(
        _raw_frame(),
        ["AAPL.US"],
        start=dt.date(2024, 1, 3),
        end=dt.date(2024, 1, 4),
    )
    assert out.index.min() == pd.Timestamp("2024-01-03")
    assert out.index.max() == pd.Timestamp("2024-01-04")
    assert len(out) == 2


def test_normalize_price_frame_drops_duplicate_dates_keeping_last():
    frame = pd.DataFrame(
        {"AAPL.US": [1.0, 2.0]}, index=["2024-01-02", "2024-01-02"]
    )
    out = normalize_price_frame(frame, ["AAPL.US"])
    assert len(out) == 1
    assert out.iloc[0, 0] == 2.0


def test_normalize_price_series_indexed_by_requested_tickers():
    out = normalize_price_series({"AAPL.US": "190.5"}, ["AAPL.US", "MSFT.US"])
    assert out["AAPL.US"] == 190.5
    assert "MSFT.US" not in out.index
    assert out.dtype == "float64"


def test_base_source_reports_capabilities():
    class Sample(BaseSource):
        name = "sample"
        capabilities = frozenset({Capability.PRICE_HISTORY})

    source = Sample()
    assert source.supports(Capability.PRICE_HISTORY)
    assert not source.supports(Capability.LATEST)
    assert source.is_available()


def test_base_source_raises_for_unsupported_capability():
    class Sample(BaseSource):
        name = "sample"
        capabilities = frozenset({Capability.PRICE_HISTORY})

    with pytest.raises(CapabilityNotSupported, match="sample"):
        Sample().latest(["AAPL.US"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui'`

- [ ] **Step 3: Create the package skeleton**

Create `portfolio_ui/__init__.py`:

```python
"""Streamlit research UI for the portfolio construction toolkit."""

__version__ = "0.1.0"
```

Create `portfolio_ui/sources/__init__.py`:

```python
"""Market data sources normalized behind one interface."""
```

- [ ] **Step 4: Write `portfolio_ui/sources/base.py`**

```python
"""The source protocol every market-data client is adapted to.

Nothing in this module may import streamlit - it is plain, testable Python.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Iterable, Mapping, Protocol, runtime_checkable

import pandas as pd


class Capability(str, Enum):
    """What a source can do. Values are stable identifiers used in the UI."""

    PRICE_HISTORY = "price_history"
    CLOSE_AT = "close_at"
    LATEST = "latest"
    SOVEREIGN = "sovereign_yields"


class SourceError(RuntimeError):
    """Base class for every failure a source can report."""


class SourceUnavailable(SourceError):
    """The source cannot be used at all, typically a missing API key."""


class TickerNotFound(SourceError):
    """Upstream returned no data for one or more tickers."""


class UpstreamError(SourceError):
    """The upstream API failed or returned something unparseable."""


class CapabilityNotSupported(SourceError):
    """This source does not implement the requested capability."""


@runtime_checkable
class PriceSource(Protocol):
    """The four capabilities the UI can route to any source."""

    name: str
    capabilities: frozenset[Capability]

    def price_history(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame: ...

    def close_at(self, tickers: list[str], on: dt.date) -> pd.Series: ...

    def latest(self, tickers: list[str]) -> pd.Series: ...

    def sovereign_yields(
        self, countries: list[str], tenors: list[int], on: dt.date
    ) -> pd.DataFrame: ...


class BaseSource:
    """Concrete base so subclasses only implement what they support."""

    name: str = "base"
    capabilities: frozenset[Capability] = frozenset()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def is_available(self) -> bool:
        """Whether the source is usable right now (e.g. its API key is set)."""
        return True

    def unavailable_reason(self) -> str | None:
        """Human-readable reason the source cannot be used, or None."""
        return None

    def _unsupported(self, capability: Capability):
        raise CapabilityNotSupported(
            f"{self.name} does not support {capability.value}"
        )

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        self._unsupported(Capability.PRICE_HISTORY)

    def close_at(self, tickers, on) -> pd.Series:
        self._unsupported(Capability.CLOSE_AT)

    def latest(self, tickers) -> pd.Series:
        self._unsupported(Capability.LATEST)

    def sovereign_yields(self, countries, tenors, on) -> pd.DataFrame:
        self._unsupported(Capability.SOVEREIGN)


def normalize_price_frame(
    frame: pd.DataFrame,
    tickers: Iterable[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Coerce any client's price output into the shared contract.

    Ascending DatetimeIndex named "Date", float64 values, one column per
    requested ticker in the requested order. Tickers with no data are dropped.
    """
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    present = [t for t in tickers if t in out.columns]
    out = out[present]
    out = out.apply(pd.to_numeric, errors="coerce").astype("float64")

    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]

    return out


def normalize_price_series(
    values: Mapping[str, object], tickers: Iterable[str]
) -> pd.Series:
    """Coerce point-in-time / latest prices into a float Series by ticker."""
    ordered = {t: values[t] for t in tickers if t in values}
    return pd.Series(ordered, dtype="float64")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_base.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Wire up packaging**

In `pyproject.toml`, replace the `[project.optional-dependencies]` and `[tool.setuptools.packages.find]` sections with:

```toml
[project.optional-dependencies]
test = ["pytest"]
ui = ["streamlit", "plotly", "pyarrow"]

[tool.setuptools.packages.find]
include = ["portfolio_construction*", "portfolio_ui*"]
```

In `.gitignore`, add:

```
data/
```

- [ ] **Step 7: Install the UI extra and confirm the package imports**

Run: `.venv/Scripts/python.exe -m pip install -e ".[ui,test]"`
Then: `.venv/Scripts/python.exe -c "import portfolio_ui, streamlit, plotly, pyarrow; print(portfolio_ui.__version__)"`
Expected: prints `0.1.0` with no import error.

- [ ] **Step 8: Commit**

```bash
git add portfolio_ui/ tests/ui/ pyproject.toml .gitignore
git commit -m "feat: add portfolio_ui package with source protocol and normalizers"
```

---

### Task 5: `EodApiSource`

**Files:**
- Create: `portfolio_ui/sources/eod_api_source.py`
- Test: `tests/ui/test_eod_api_source.py`

**Interfaces:**
- Consumes: everything from `portfolio_ui.sources.base` (Task 4).
- Produces: `EodApiSource(token: str | None = None, client=None)` with `name = "eod_api"` and all four capabilities. `client` defaults to the `portfolio_construction.eod_api` module and exists so tests can inject a fake.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_eod_api_source.py`:

```python
"""EodApiSource adapts the synchronous client to the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, SourceUnavailable, TickerNotFound
from portfolio_ui.sources.eod_api_source import EodApiSource


class FakeEodApiClient:
    """Stands in for portfolio_construction.eod_api."""

    def __init__(self):
        self.calls = []

    def adjusted_prices(self, tok, tickers):
        self.calls.append(("adjusted_prices", tok, tuple(tickers)))
        return pd.DataFrame(
            {t: [100.0, 101.0, 102.0] for t in tickers},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

    def recursive_adjClose_atDate(self, token, ticker, strDate):
        self.calls.append(("recursive_adjClose_atDate", token, ticker, strDate))
        if ticker == "MISSING.US":
            raise IndexError("no data")
        return 123.75

    def last_prices_universe(self, token, tickers):
        self.calls.append(("last_prices_universe", token, tuple(tickers)))
        return pd.Series({t: 55.5 for t in tickers}, dtype="float64")

    def get_SovBond(self, token, ticker, strDate):
        self.calls.append(("get_SovBond", token, ticker, strDate))
        return 4.25


def _source():
    return EodApiSource(token="unit-test-key", client=FakeEodApiClient())


def test_declares_all_four_capabilities():
    assert _source().capabilities == frozenset(Capability)


def test_is_unavailable_without_a_token():
    source = EodApiSource(token="", client=FakeEodApiClient())
    assert not source.is_available()
    assert "EOD_API_KEY" in source.unavailable_reason()


def test_price_history_raises_when_token_missing():
    source = EodApiSource(token="", client=FakeEodApiClient())
    with pytest.raises(SourceUnavailable):
        source.price_history(["AAPL.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_price_history_returns_normalized_frame():
    out = _source().price_history(
        ["AAPL.US", "MSFT.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US", "MSFT.US"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_slices_to_requested_range():
    out = _source().price_history(
        ["AAPL.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 3)
    )
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03")


def test_close_at_returns_series_by_ticker():
    out = _source().close_at(["AAPL.US", "MSFT.US"], dt.date(2024, 1, 3))
    assert out["AAPL.US"] == 123.75
    assert out.dtype == "float64"


def test_close_at_raises_when_every_ticker_is_missing():
    with pytest.raises(TickerNotFound):
        _source().close_at(["MISSING.US"], dt.date(2024, 1, 3))


def test_close_at_keeps_survivors_on_partial_failure():
    out = _source().close_at(["AAPL.US", "MISSING.US"], dt.date(2024, 1, 3))
    assert list(out.index) == ["AAPL.US"]


def test_latest_returns_series_by_ticker():
    out = _source().latest(["AAPL.US", "MSFT.US"])
    assert out["MSFT.US"] == 55.5


def test_sovereign_yields_builds_bond_tickers():
    out = _source().sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert list(out.columns) == ["Rates"]
    assert list(out.index) == [
        "US5Y.GBOND",
        "US10Y.GBOND",
        "FR5Y.GBOND",
        "FR10Y.GBOND",
    ]
    assert out.loc["US5Y.GBOND", "Rates"] == 4.25
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_eod_api_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.sources.eod_api_source'`

- [ ] **Step 3: Write `portfolio_ui/sources/eod_api_source.py`**

```python
"""Adapter for the synchronous portfolio_construction.eod_api client."""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    SourceUnavailable,
    TickerNotFound,
    UpstreamError,
    normalize_price_frame,
    normalize_price_series,
)


def sovereign_ticker(country: str, tenor: int) -> str:
    """EOD's sovereign bond ticker convention, e.g. US10Y.GBOND."""
    return f"{country}{tenor}Y.GBOND"


class EodApiSource(BaseSource):
    """The synchronous client: slower, but the widest feature set."""

    name = "eod_api"
    capabilities = frozenset(Capability)

    def __init__(self, token: str | None = None, client=None):
        if client is None:
            from portfolio_construction import eod_api

            client = eod_api
        self._client = client
        self._token = token if token is not None else os.environ.get("EOD_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._token)

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "EOD_API_KEY is not set"

    def _require_token(self) -> str:
        if not self._token:
            raise SourceUnavailable("EOD_API_KEY is not set")
        return self._token

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        token = self._require_token()
        try:
            raw = self._client.adjusted_prices(token, list(tickers))
        except Exception as exc:  # network / parse failures
            raise UpstreamError(f"eod_api price history failed: {exc}") from exc

        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(f"no price history for {list(tickers)}")
        return frame

    def close_at(self, tickers, on: dt.date) -> pd.Series:
        token = self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        values: dict[str, float] = {}
        for ticker in tickers:
            try:
                values[ticker] = self._client.recursive_adjClose_atDate(
                    token, ticker, str_date
                )
            except (IndexError, KeyError):
                continue
            except Exception as exc:
                raise UpstreamError(f"eod_api close_at failed for {ticker}: {exc}") from exc

        if not values:
            raise TickerNotFound(f"no close on {str_date} for {list(tickers)}")
        return normalize_price_series(values, tickers)

    def latest(self, tickers) -> pd.Series:
        token = self._require_token()
        try:
            raw = self._client.last_prices_universe(token, list(tickers))
        except Exception as exc:
            raise UpstreamError(f"eod_api latest failed: {exc}") from exc

        series = normalize_price_series(dict(raw), tickers)
        if series.empty:
            raise TickerNotFound(f"no latest price for {list(tickers)}")
        return series

    def sovereign_yields(self, countries, tenors, on: dt.date) -> pd.DataFrame:
        token = self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        rates: dict[str, float] = {}
        for country in countries:
            for tenor in tenors:
                ticker = sovereign_ticker(country, tenor)
                try:
                    rates[ticker] = self._client.get_SovBond(token, ticker, str_date)
                except (IndexError, KeyError):
                    continue
                except Exception as exc:
                    raise UpstreamError(
                        f"eod_api sovereign yield failed for {ticker}: {exc}"
                    ) from exc

        if not rates:
            raise TickerNotFound(f"no sovereign yields on {str_date}")

        frame = pd.DataFrame.from_dict(rates, orient="index")
        frame.columns = ["Rates"]
        return frame.astype("float64")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_eod_api_source.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/sources/eod_api_source.py tests/ui/test_eod_api_source.py
git commit -m "feat: add EodApiSource adapter"
```

---

### Task 6: `AsyncEodSource`

**Files:**
- Create: `portfolio_ui/sources/async_eod_source.py`
- Test: `tests/ui/test_async_eod_source.py`

**Interfaces:**
- Consumes: `portfolio_ui.sources.base` (Task 4).
- Produces: `AsyncEodSource(token=None, client=None)`, `name = "async_eod"`, all four capabilities. Exposes the module constant `CLIENT_SIDE_SLICE_NOTE: str` used as an `ActiveDataset.notes` entry in Task 9.

The client returns awkward shapes this adapter must flatten:
`get_full_history(tickers)` → list of `pd.Series` (or `None`), one per ticker, each named after its ticker; `get_realtime(tickers)` → flat list of `{"code", "close"}` dicts; `get_historical(tickers, date)` → list of `{"code", "close"}` dicts or `None`; `sovereign_bonds(...)` → DataFrame with a `Rates` column.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_async_eod_source.py`:

```python
"""AsyncEodSource flattens the parallel client into the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.async_eod_source import CLIENT_SIDE_SLICE_NOTE, AsyncEodSource
from portfolio_ui.sources.base import Capability, SourceUnavailable, TickerNotFound


class FakeAsyncEodClient:
    """Stands in for portfolio_construction.async_eod."""

    def __init__(self):
        self.calls = []

    def get_full_history(self, tickers):
        self.calls.append(("get_full_history", tuple(tickers)))
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        out = []
        for ticker in tickers:
            if ticker == "MISSING.US":
                out.append(None)
            else:
                out.append(pd.Series([100.0, 101.0, 102.0], index=index, name=ticker))
        return out

    def get_historical(self, tickers, date):
        self.calls.append(("get_historical", tuple(tickers), date))
        return [
            None if t == "MISSING.US" else {"code": t, "close": 123.75} for t in tickers
        ]

    def get_realtime(self, tickers):
        self.calls.append(("get_realtime", tuple(tickers)))
        return [{"code": t, "close": 55.5} for t in tickers]

    def sovereign_bonds(self, countries, tenors, strDate):
        self.calls.append(("sovereign_bonds", tuple(countries), tuple(tenors), strDate))
        tickers = [f"{c}{t}Y.GBOND" for c in countries for t in tenors]
        return pd.DataFrame({"Rates": [4.25] * len(tickers)}, index=tickers)


def _source():
    return AsyncEodSource(token="unit-test-key", client=FakeAsyncEodClient())


def test_declares_all_four_capabilities():
    assert _source().capabilities == frozenset(Capability)


def test_is_unavailable_without_a_token():
    source = AsyncEodSource(token="", client=FakeAsyncEodClient())
    assert not source.is_available()
    assert "EOD_API_KEY" in source.unavailable_reason()


def test_price_history_raises_when_token_missing():
    source = AsyncEodSource(token="", client=FakeAsyncEodClient())
    with pytest.raises(SourceUnavailable):
        source.price_history(["AAPL.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_price_history_concatenates_series_into_one_frame():
    out = _source().price_history(
        ["AAPL.US", "MSFT.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US", "MSFT.US"]
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_slices_client_side():
    out = _source().price_history(
        ["AAPL.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 3)
    )
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03")


def test_price_history_drops_tickers_with_no_history():
    out = _source().price_history(
        ["AAPL.US", "MISSING.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL.US"]


def test_price_history_raises_when_nothing_came_back():
    with pytest.raises(TickerNotFound):
        _source().price_history(
            ["MISSING.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
        )


def test_client_side_slice_note_is_exposed():
    assert "start" in CLIENT_SIDE_SLICE_NOTE and "async_eod" in CLIENT_SIDE_SLICE_NOTE


def test_close_at_flattens_code_close_dicts():
    out = _source().close_at(["AAPL.US", "MSFT.US"], dt.date(2024, 1, 3))
    assert out["AAPL.US"] == 123.75
    assert out.dtype == "float64"


def test_close_at_skips_none_results():
    out = _source().close_at(["AAPL.US", "MISSING.US"], dt.date(2024, 1, 3))
    assert list(out.index) == ["AAPL.US"]


def test_latest_flattens_code_close_dicts():
    out = _source().latest(["AAPL.US", "MSFT.US"])
    assert out["MSFT.US"] == 55.5


def test_sovereign_yields_returns_rates_frame():
    out = _source().sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert list(out.columns) == ["Rates"]
    assert out.loc["US10Y.GBOND", "Rates"] == 4.25
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_async_eod_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.sources.async_eod_source'`

- [ ] **Step 3: Write `portfolio_ui/sources/async_eod_source.py`**

```python
"""Adapter for the parallel portfolio_construction.async_eod client."""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    SourceUnavailable,
    TickerNotFound,
    UpstreamError,
    normalize_price_frame,
    normalize_price_series,
)

CLIENT_SIDE_SLICE_NOTE = (
    "async_eod always requests 1990-01-01 to today and ignores start/end; "
    "the range was applied client-side"
)


class AsyncEodSource(BaseSource):
    """The parallel client: much faster for many tickers, fewer endpoints."""

    name = "async_eod"
    capabilities = frozenset(Capability)

    def __init__(self, token: str | None = None, client=None):
        if client is None:
            from portfolio_construction import async_eod

            client = async_eod
        self._client = client
        self._token = token if token is not None else os.environ.get("EOD_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._token)

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "EOD_API_KEY is not set"

    def _require_token(self) -> str:
        if not self._token:
            raise SourceUnavailable("EOD_API_KEY is not set")
        return self._token

    @staticmethod
    def _flatten_code_close(entries) -> dict[str, float]:
        values: dict[str, float] = {}
        for entry in entries or []:
            if not entry:
                continue
            values[entry["code"]] = entry["close"]
        return values

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        self._require_token()
        tickers = list(tickers)
        try:
            series_list = self._client.get_full_history(tickers)
        except Exception as exc:
            raise UpstreamError(f"async_eod price history failed: {exc}") from exc

        usable = [s for s in series_list if s is not None and len(s)]
        if not usable:
            raise TickerNotFound(f"no price history for {tickers}")

        raw = pd.concat(usable, axis=1)
        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"no price history for {tickers} between {start} and {end}"
            )
        return frame

    def close_at(self, tickers, on: dt.date) -> pd.Series:
        self._require_token()
        tickers = list(tickers)
        try:
            entries = self._client.get_historical(tickers, pd.Timestamp(on).to_pydatetime())
        except Exception as exc:
            raise UpstreamError(f"async_eod close_at failed: {exc}") from exc

        values = self._flatten_code_close(entries)
        if not values:
            raise TickerNotFound(f"no close on {on} for {tickers}")
        return normalize_price_series(values, tickers)

    def latest(self, tickers) -> pd.Series:
        self._require_token()
        tickers = list(tickers)
        try:
            entries = self._client.get_realtime(tickers)
        except Exception as exc:
            raise UpstreamError(f"async_eod latest failed: {exc}") from exc

        values = self._flatten_code_close(entries)
        if not values:
            raise TickerNotFound(f"no latest price for {tickers}")
        return normalize_price_series(values, tickers)

    def sovereign_yields(self, countries, tenors, on: dt.date) -> pd.DataFrame:
        self._require_token()
        str_date = pd.Timestamp(on).strftime("%Y-%m-%d")
        try:
            frame = self._client.sovereign_bonds(list(countries), list(tenors), str_date)
        except Exception as exc:
            raise UpstreamError(f"async_eod sovereign yields failed: {exc}") from exc

        if frame is None or frame.empty:
            raise TickerNotFound(f"no sovereign yields on {str_date}")
        return frame.astype("float64")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_async_eod_source.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/sources/async_eod_source.py tests/ui/test_async_eod_source.py
git commit -m "feat: add AsyncEodSource adapter"
```

---

### Task 7: `MarketAccessSource` and `LocalSource`

**Files:**
- Create: `portfolio_ui/sources/market_access_source.py`
- Create: `portfolio_ui/sources/local_source.py`
- Test: `tests/ui/test_market_access_source.py`
- Test: `tests/ui/test_local_source.py`

**Interfaces:**
- Consumes: `portfolio_ui.sources.base` (Task 4).
- Produces:
  - `MarketAccessSource(client=None)`, `name = "market_access"`, `capabilities = frozenset({Capability.PRICE_HISTORY})`.
  - `LocalSource(frame: pd.DataFrame, label: str)`, `name = "local"`, `capabilities = frozenset({Capability.PRICE_HISTORY})`; plus `LocalSource.from_upload(file_like, filename) -> LocalSource` reading `.csv`, `.xlsx` and `.parquet`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_market_access_source.py`:

```python
"""MarketAccessSource exposes yfinance history under the shared contract."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, CapabilityNotSupported, TickerNotFound
from portfolio_ui.sources.market_access_source import MarketAccessSource


class FakeMarketAccessClient:
    def yahooFinance_historical_data(self, ticker):
        if ticker == "MISSING":
            return pd.DataFrame()
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        return pd.DataFrame(
            {
                "Open": [99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 102.0],
                "Adj Close": [100.0, 101.0, 102.0],
            },
            index=index,
        )


def _source():
    return MarketAccessSource(client=FakeMarketAccessClient())


def test_declares_only_price_history():
    assert _source().capabilities == frozenset({Capability.PRICE_HISTORY})


def test_close_at_is_not_supported():
    with pytest.raises(CapabilityNotSupported, match="market_access"):
        _source().close_at(["AAPL"], dt.date(2024, 1, 3))


def test_latest_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        _source().latest(["AAPL"])


def test_sovereign_yields_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        _source().sovereign_yields(["US"], [10], dt.date(2024, 1, 3))


def test_price_history_builds_one_column_per_ticker():
    out = _source().price_history(
        ["AAPL", "MSFT"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert out.index.name == "Date"
    assert all(d == "float64" for d in out.dtypes)


def test_price_history_drops_empty_tickers():
    out = _source().price_history(
        ["AAPL", "MISSING"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert list(out.columns) == ["AAPL"]


def test_price_history_raises_when_nothing_came_back():
    with pytest.raises(TickerNotFound):
        _source().price_history(["MISSING"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
```

Create `tests/ui/test_local_source.py`:

```python
"""LocalSource serves an already-materialized price frame."""

import datetime as dt
import io

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, CapabilityNotSupported
from portfolio_ui.sources.local_source import LocalSource


def _frame():
    return pd.DataFrame(
        {"AAPL.US": [100.0, 101.0, 102.0], "MSFT.US": [400.0, 401.0, 402.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_declares_only_price_history():
    source = LocalSource(_frame(), label="upload:prices.csv")
    assert source.capabilities == frozenset({Capability.PRICE_HISTORY})


def test_name_reports_the_label():
    source = LocalSource(_frame(), label="upload:prices.csv")
    assert source.name == "upload:prices.csv"


def test_latest_is_not_supported():
    with pytest.raises(CapabilityNotSupported):
        LocalSource(_frame(), label="local").latest(["AAPL.US"])


def test_price_history_slices_and_selects():
    source = LocalSource(_frame(), label="local")
    out = source.price_history(
        ["MSFT.US"], dt.date(2024, 1, 3), dt.date(2024, 1, 4)
    )
    assert list(out.columns) == ["MSFT.US"]
    assert len(out) == 2


def test_available_tickers_lists_frame_columns():
    source = LocalSource(_frame(), label="local")
    assert source.available_tickers() == ["AAPL.US", "MSFT.US"]


def test_from_upload_reads_csv():
    csv = b"Date,AAPL.US\n2024-01-02,100.0\n2024-01-03,101.0\n"
    source = LocalSource.from_upload(io.BytesIO(csv), "prices.csv")
    assert source.name == "upload:prices.csv"
    assert source.available_tickers() == ["AAPL.US"]


def test_from_upload_rejects_unknown_extension():
    with pytest.raises(ValueError, match="unsupported"):
        LocalSource.from_upload(io.BytesIO(b""), "prices.txt")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_market_access_source.py tests/ui/test_local_source.py -v`
Expected: FAIL — both modules do not exist.

- [ ] **Step 3: Write `portfolio_ui/sources/market_access_source.py`**

```python
"""Adapter for the yfinance-backed portfolio_construction.market_access module."""

from __future__ import annotations

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    TickerNotFound,
    UpstreamError,
    normalize_price_frame,
)


class MarketAccessSource(BaseSource):
    """Free fallback path. History only - no point-in-time, realtime or bonds."""

    name = "market_access"
    capabilities = frozenset({Capability.PRICE_HISTORY})

    def __init__(self, client=None):
        if client is None:
            from portfolio_construction import market_access

            client = market_access
        self._client = client

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        tickers = list(tickers)
        columns = {}
        for ticker in tickers:
            try:
                history = self._client.yahooFinance_historical_data(ticker)
            except Exception as exc:
                raise UpstreamError(
                    f"market_access price history failed for {ticker}: {exc}"
                ) from exc

            if history is None or history.empty:
                continue
            column = "Adj Close" if "Adj Close" in history.columns else "Close"
            columns[ticker] = history[column]

        if not columns:
            raise TickerNotFound(f"no price history for {tickers}")

        raw = pd.DataFrame(columns)
        frame = normalize_price_frame(raw, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"no price history for {tickers} between {start} and {end}"
            )
        return frame
```

- [ ] **Step 4: Write `portfolio_ui/sources/local_source.py`**

```python
"""A price frame the user supplied by upload or reloaded from disk."""

from __future__ import annotations

from pathlib import PurePath

import pandas as pd

from portfolio_ui.sources.base import (
    BaseSource,
    Capability,
    TickerNotFound,
    normalize_price_frame,
)


class LocalSource(BaseSource):
    """Wraps an in-memory frame so uploads satisfy the same contract.

    Not offered in the source selector (spec section 5.1) - it is reached
    through upload and saved-dataset loading - but it implements PriceSource so
    the cross-source contract test covers it.
    """

    capabilities = frozenset({Capability.PRICE_HISTORY})

    def __init__(self, frame: pd.DataFrame, label: str):
        self._frame = frame
        self.name = label

    @classmethod
    def from_upload(cls, file_like, filename: str) -> "LocalSource":
        suffix = PurePath(filename).suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(file_like, index_col=0, parse_dates=True)
        elif suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(file_like, index_col=0, parse_dates=True)
        elif suffix == ".parquet":
            frame = pd.read_parquet(file_like)
        else:
            raise ValueError(
                f"unsupported file type '{suffix}' - use .csv, .xlsx or .parquet"
            )
        return cls(frame, label=f"upload:{filename}")

    def available_tickers(self) -> list[str]:
        return list(self._frame.columns)

    def price_history(self, tickers, start, end) -> pd.DataFrame:
        frame = normalize_price_frame(self._frame, tickers, start=start, end=end)
        if frame.empty:
            raise TickerNotFound(
                f"the loaded file has no data for {list(tickers)} in that range"
            )
        return frame
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_market_access_source.py tests/ui/test_local_source.py -v`
Expected: PASS (14 tests)

- [ ] **Step 6: Commit**

```bash
git add portfolio_ui/sources/market_access_source.py portfolio_ui/sources/local_source.py tests/ui/test_market_access_source.py tests/ui/test_local_source.py
git commit -m "feat: add MarketAccessSource and LocalSource adapters"
```

---

### Task 8: Registry and the cross-source contract test

This is the test that makes "swap the source, everything else keeps working" verified rather than hoped (spec §8).

**Files:**
- Create: `portfolio_ui/sources/registry.py`
- Test: `tests/ui/test_registry.py`
- Test: `tests/ui/test_source_contract.py`

**Interfaces:**
- Consumes: all four source classes (Tasks 5-7).
- Produces:
  - `SELECTABLE_SOURCES: tuple[str, ...]` — `("async_eod", "eod_api", "market_access")`, the selector's options in display order. `local` is deliberately excluded (spec §5.1).
  - `build_source(name: str, token: str | None = None) -> PriceSource` — raises `KeyError` for unknown names.
  - `describe_sources(token: str | None = None) -> list[SourceInfo]` where `SourceInfo` is a dataclass with `name: str`, `capabilities: frozenset[Capability]`, `available: bool`, `reason: str | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_registry.py`:

```python
"""The registry maps selector names to source instances."""

import pytest

from portfolio_ui.sources.base import Capability
from portfolio_ui.sources.registry import (
    SELECTABLE_SOURCES,
    build_source,
    describe_sources,
)


def test_selectable_sources_excludes_local():
    assert "local" not in SELECTABLE_SOURCES


def test_selectable_sources_lists_the_three_clients():
    assert set(SELECTABLE_SOURCES) == {"async_eod", "eod_api", "market_access"}


def test_build_source_returns_named_instance():
    assert build_source("eod_api", token="k").name == "eod_api"
    assert build_source("async_eod", token="k").name == "async_eod"
    assert build_source("market_access").name == "market_access"


def test_build_source_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_source("nope")


def test_describe_sources_reports_availability_without_a_key():
    infos = {info.name: info for info in describe_sources(token="")}
    assert not infos["eod_api"].available
    assert "EOD_API_KEY" in infos["eod_api"].reason
    assert infos["market_access"].available


def test_describe_sources_reports_capabilities():
    infos = {info.name: info for info in describe_sources(token="k")}
    assert infos["async_eod"].capabilities == frozenset(Capability)
    assert infos["market_access"].capabilities == frozenset(
        {Capability.PRICE_HISTORY}
    )
```

Create `tests/ui/test_source_contract.py`:

```python
"""One contract every source must satisfy for each capability it declares.

This is what lets the rest of the app treat sources interchangeably.
"""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.sources.base import Capability, PriceSource
from portfolio_ui.sources.local_source import LocalSource
from tests.ui.test_async_eod_source import FakeAsyncEodClient
from tests.ui.test_eod_api_source import FakeEodApiClient
from tests.ui.test_market_access_source import FakeMarketAccessClient

from portfolio_ui.sources.async_eod_source import AsyncEodSource
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource

START = dt.date(2024, 1, 1)
END = dt.date(2024, 1, 31)
TICKERS = ["AAPL.US", "MSFT.US"]


def _local_frame():
    return pd.DataFrame(
        {t: [100.0, 101.0, 102.0] for t in TICKERS},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


ALL_SOURCES = [
    pytest.param(
        lambda: EodApiSource(token="k", client=FakeEodApiClient()), id="eod_api"
    ),
    pytest.param(
        lambda: AsyncEodSource(token="k", client=FakeAsyncEodClient()), id="async_eod"
    ),
    pytest.param(
        lambda: MarketAccessSource(client=FakeMarketAccessClient()), id="market_access"
    ),
    pytest.param(
        lambda: LocalSource(_local_frame(), label="local"), id="local"
    ),
]


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_source_satisfies_the_protocol(factory):
    source = factory()
    assert isinstance(source, PriceSource)
    assert isinstance(source.name, str) and source.name
    assert isinstance(source.capabilities, frozenset)


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_price_history_contract(factory):
    source = factory()
    if not source.supports(Capability.PRICE_HISTORY):
        pytest.skip(f"{source.name} does not support price history")

    tickers = ["AAPL", "MSFT"] if source.name == "market_access" else TICKERS
    out = source.price_history(tickers, START, END)

    assert isinstance(out, pd.DataFrame)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert out.index.is_monotonic_increasing
    assert list(out.columns) == tickers
    assert all(d == "float64" for d in out.dtypes)
    assert not out.index.has_duplicates


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_close_at_contract(factory):
    source = factory()
    if not source.supports(Capability.CLOSE_AT):
        pytest.skip(f"{source.name} does not support close_at")

    out = source.close_at(TICKERS, dt.date(2024, 1, 3))
    assert isinstance(out, pd.Series)
    assert out.dtype == "float64"
    assert set(out.index).issubset(set(TICKERS))


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_latest_contract(factory):
    source = factory()
    if not source.supports(Capability.LATEST):
        pytest.skip(f"{source.name} does not support latest")

    out = source.latest(TICKERS)
    assert isinstance(out, pd.Series)
    assert out.dtype == "float64"
    assert set(out.index).issubset(set(TICKERS))


@pytest.mark.parametrize("factory", ALL_SOURCES)
def test_sovereign_yields_contract(factory):
    source = factory()
    if not source.supports(Capability.SOVEREIGN):
        pytest.skip(f"{source.name} does not support sovereign yields")

    out = source.sovereign_yields(["US", "FR"], [5, 10], dt.date(2024, 6, 28))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["Rates"]
    assert all(d == "float64" for d in out.dtypes)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_registry.py tests/ui/test_source_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.sources.registry'`

- [ ] **Step 3: Write `portfolio_ui/sources/registry.py`**

```python
"""Selector names to source instances, plus availability reporting."""

from __future__ import annotations

import os
from dataclasses import dataclass

from portfolio_ui.sources.async_eod_source import AsyncEodSource
from portfolio_ui.sources.base import Capability, PriceSource
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource

# Display order in the sidebar. "local" is intentionally absent: uploads and
# saved datasets are reached from the Data page, not the source selector.
SELECTABLE_SOURCES: tuple[str, ...] = ("async_eod", "eod_api", "market_access")


@dataclass(frozen=True)
class SourceInfo:
    """What the sidebar needs to render one source option."""

    name: str
    capabilities: frozenset[Capability]
    available: bool
    reason: str | None


def build_source(name: str, token: str | None = None) -> PriceSource:
    """Instantiate a source by its selector name."""
    if token is None:
        token = os.environ.get("EOD_API_KEY", "")

    if name == "async_eod":
        return AsyncEodSource(token=token)
    if name == "eod_api":
        return EodApiSource(token=token)
    if name == "market_access":
        return MarketAccessSource()
    raise KeyError(f"unknown source '{name}'")


def describe_sources(token: str | None = None) -> list[SourceInfo]:
    """Describe every selectable source for the sidebar."""
    infos = []
    for name in SELECTABLE_SOURCES:
        source = build_source(name, token=token)
        infos.append(
            SourceInfo(
                name=name,
                capabilities=source.capabilities,
                available=source.is_available(),
                reason=source.unavailable_reason(),
            )
        )
    return infos
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_registry.py tests/ui/test_source_contract.py -v`
Expected: PASS — 6 registry tests, and 20 contract tests (5 tests × 4 sources, with skips where a capability is not declared).

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/sources/registry.py tests/ui/test_registry.py tests/ui/test_source_contract.py
git commit -m "feat: add source registry and cross-source contract test"
```

---

### Task 9: `ActiveDataset` — build, validate, save, load

**Files:**
- Create: `portfolio_ui/dataset.py`
- Test: `tests/ui/test_dataset.py`

**Interfaces:**
- Consumes: `portfolio_ui.sources.base` (Task 4), `portfolio_ui.sources.async_eod_source.CLIENT_SIDE_SLICE_NOTE` (Task 6), `portfolio_construction.stats.time_series_frequence_inference`.
- Produces:
  - `ActiveDataset` frozen dataclass with fields `name, prices, source, tickers, start, end, fetched_at, frequency, notes`, plus `summary() -> str`.
  - `build_dataset(source, name, tickers, start, end) -> ActiveDataset`
  - `dataset_from_frame(frame, name, source_label, notes=()) -> ActiveDataset`
  - `notes_for_fetch(source_name, requested, returned_columns) -> tuple[str, ...]`
  - `infer_frequency(index) -> str`
  - `validate_prices(frame) -> None` raising `ValueError`
  - `save_dataset(dataset, directory=None) -> Path`
  - `load_dataset(name, directory=None) -> ActiveDataset`
  - `list_saved(directory=None) -> list[str]`
  - `default_directory() -> Path` honouring `PORTFOLIO_UI_DATA_DIR`, defaulting to `data/datasets`

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_dataset.py`:

```python
"""The shared dataset: building, validation and round-tripping to disk."""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.dataset import (
    ActiveDataset,
    build_dataset,
    dataset_from_frame,
    default_directory,
    infer_frequency,
    list_saved,
    load_dataset,
    notes_for_fetch,
    save_dataset,
    validate_prices,
)
from portfolio_ui.sources.local_source import LocalSource

TICKERS = ["AAPL.US", "MSFT.US"]


def _frame():
    return pd.DataFrame(
        {t: [100.0, 101.0, 102.0] for t in TICKERS},
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="Date"
        ),
    )


def test_validate_rejects_non_datetime_index():
    frame = _frame().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_prices(frame)


def test_validate_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        validate_prices(pd.DataFrame())


def test_validate_rejects_non_numeric_columns():
    frame = _frame()
    frame["TEXT"] = "abc"
    with pytest.raises(ValueError, match="numeric"):
        validate_prices(frame)


def test_validate_accepts_a_good_frame():
    validate_prices(_frame())  # must not raise


def test_build_dataset_records_provenance_and_metadata():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(source, "sample", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(ds, ActiveDataset)
    assert ds.name == "sample"
    assert ds.source == "local"
    assert ds.tickers == tuple(TICKERS)
    assert ds.frequency  # inferred, non-empty
    assert isinstance(ds.fetched_at, dt.datetime)


def test_build_dataset_narrows_range_to_what_came_back():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(source, "sample", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert ds.start == dt.date(2024, 1, 2)
    assert ds.end == dt.date(2024, 1, 4)


def test_build_dataset_notes_client_side_slicing_for_async_eod():
    # LocalSource takes its name from the label, so this stands in for a fetch
    # that actually ran through async_eod.
    source = LocalSource(_frame(), label="async_eod")
    ds = build_dataset(source, "s", TICKERS, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert any("client-side" in note for note in ds.notes)


def test_infer_frequency_of_a_single_row_is_unknown():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]), name="Date"),
    )
    assert infer_frequency(frame.index) == "Unknown"


def test_infer_frequency_of_daily_data():
    assert infer_frequency(_frame().index) == "Daily"


def test_notes_for_fetch_is_empty_when_everything_came_back():
    assert notes_for_fetch("eod_api", TICKERS, TICKERS) == ()


def test_build_dataset_notes_dropped_tickers():
    source = LocalSource(_frame(), label="local")
    ds = build_dataset(
        source, "s", TICKERS + ["GONE.US"], dt.date(2024, 1, 1), dt.date(2024, 1, 31)
    )
    assert any("GONE.US" in note for note in ds.notes)
    assert ds.tickers == tuple(TICKERS)


def test_dataset_is_frozen():
    ds = dataset_from_frame(_frame(), "sample", "upload:x.csv")
    with pytest.raises(Exception):
        ds.name = "other"


def test_save_and_load_round_trip(tmp_path):
    ds = dataset_from_frame(_frame(), "sample", "upload:x.csv")
    path = save_dataset(ds, directory=tmp_path)
    assert path.exists()

    loaded = load_dataset("sample", directory=tmp_path)
    pd.testing.assert_frame_equal(loaded.prices, ds.prices)
    assert loaded.source == "upload:x.csv"
    assert loaded.tickers == ds.tickers
    assert loaded.frequency == ds.frequency
    assert loaded.notes == ds.notes


def test_list_saved_returns_names(tmp_path):
    save_dataset(dataset_from_frame(_frame(), "one", "local"), directory=tmp_path)
    save_dataset(dataset_from_frame(_frame(), "two", "local"), directory=tmp_path)
    assert sorted(list_saved(directory=tmp_path)) == ["one", "two"]


def test_load_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("nope", directory=tmp_path)


def test_default_directory_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_UI_DATA_DIR", str(tmp_path))
    assert default_directory() == tmp_path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.dataset'`

- [ ] **Step 3: Write `portfolio_ui/dataset.py`**

```python
"""The active dataset every downstream page consumes.

No streamlit import here - session wiring lives in state.py.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from portfolio_construction.stats import time_series_frequence_inference
from portfolio_ui.sources.async_eod_source import CLIENT_SIDE_SLICE_NOTE


@dataclass(frozen=True)
class ActiveDataset:
    """A price matrix plus the provenance needed to reproduce it."""

    name: str
    prices: pd.DataFrame
    source: str
    tickers: tuple[str, ...]
    start: dt.date
    end: dt.date
    fetched_at: dt.datetime
    frequency: str
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{len(self.tickers)} cols - {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} "
            f"- via {self.source}"
        )


def validate_prices(frame: pd.DataFrame) -> None:
    """Reject frames that would break downstream analysis."""
    if frame is None or frame.empty:
        raise ValueError("price frame is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("price frame must have a DatetimeIndex")
    non_numeric = [
        col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        raise ValueError(f"non-numeric columns are not allowed: {non_numeric}")


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Frequency of the index, or "Unknown" when there is too little to infer.

    time_series_frequence_inference needs at least two observations to take a
    diff; a single-day fetch would otherwise raise.
    """
    if len(index) < 2:
        return "Unknown"
    return time_series_frequence_inference(index)


def notes_for_fetch(source_name: str, requested, returned_columns) -> tuple[str, ...]:
    """Caveats worth surfacing about a fetch that just completed.

    Shared by build_dataset and the Data page, which caches the frame itself.
    """
    notes: list[str] = []
    if source_name == "async_eod":
        notes.append(CLIENT_SIDE_SLICE_NOTE)

    dropped = [t for t in requested if t not in returned_columns]
    if dropped:
        notes.append(f"no data returned for: {', '.join(dropped)}")
    return tuple(notes)


def dataset_from_frame(
    frame: pd.DataFrame,
    name: str,
    source_label: str,
    notes: tuple[str, ...] = (),
) -> ActiveDataset:
    """Wrap an already-normalized frame as a dataset."""
    validate_prices(frame)
    return ActiveDataset(
        name=name,
        prices=frame,
        source=source_label,
        tickers=tuple(frame.columns),
        start=frame.index.min().date(),
        end=frame.index.max().date(),
        fetched_at=dt.datetime.now(),
        frequency=infer_frequency(frame.index),
        notes=notes,
    )


def build_dataset(source, name: str, tickers, start: dt.date, end: dt.date) -> ActiveDataset:
    """Fetch from a source and record what actually came back."""
    requested = list(tickers)
    frame = source.price_history(requested, start, end)
    validate_prices(frame)
    notes = notes_for_fetch(source.name, requested, frame.columns)
    return dataset_from_frame(frame, name, source.name, notes=notes)


def default_directory() -> Path:
    """Where saved datasets live. Override with PORTFOLIO_UI_DATA_DIR."""
    override = os.environ.get("PORTFOLIO_UI_DATA_DIR")
    if override:
        return Path(override)
    return Path("data") / "datasets"


def _paths(name: str, directory: Path | None) -> tuple[Path, Path]:
    base = Path(directory) if directory is not None else default_directory()
    return base / f"{name}.parquet", base / f"{name}.json"


def save_dataset(dataset: ActiveDataset, directory: Path | None = None) -> Path:
    """Write the frame as parquet plus a JSON metadata sidecar."""
    parquet_path, meta_path = _paths(dataset.name, directory)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.prices.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "name": dataset.name,
                "source": dataset.source,
                "tickers": list(dataset.tickers),
                "start": dataset.start.isoformat(),
                "end": dataset.end.isoformat(),
                "fetched_at": dataset.fetched_at.isoformat(),
                "frequency": dataset.frequency,
                "notes": list(dataset.notes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parquet_path


def load_dataset(name: str, directory: Path | None = None) -> ActiveDataset:
    """Read back a saved dataset with its recorded provenance."""
    parquet_path, meta_path = _paths(name, directory)
    if not parquet_path.exists():
        raise FileNotFoundError(f"no saved dataset named '{name}' in {parquet_path.parent}")

    frame = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_prices(frame)

    return ActiveDataset(
        name=meta["name"],
        prices=frame,
        source=meta["source"],
        tickers=tuple(meta["tickers"]),
        start=dt.date.fromisoformat(meta["start"]),
        end=dt.date.fromisoformat(meta["end"]),
        fetched_at=dt.datetime.fromisoformat(meta["fetched_at"]),
        frequency=meta["frequency"],
        notes=tuple(meta["notes"]),
    )


def list_saved(directory: Path | None = None) -> list[str]:
    """Names of every dataset saved in the directory."""
    base = Path(directory) if directory is not None else default_directory()
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.parquet"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_dataset.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/dataset.py tests/ui/test_dataset.py
git commit -m "feat: add ActiveDataset with validation and disk persistence"
```

---

### Task 10: Session state accessors and the cache layer

**Files:**
- Create: `portfolio_ui/state.py`
- Create: `portfolio_ui/cache.py`
- Test: `tests/ui/test_state.py`

**Interfaces:**
- Consumes: `portfolio_ui.dataset.ActiveDataset` (Task 9), `portfolio_ui.sources.registry` (Task 8).
- Produces:
  - `state.py` functions all taking `store: MutableMapping` as their first argument (in the app this is `st.session_state`; in tests a plain dict): `init_state(store)`, `get_source_name(store)`, `set_source_name(store, name)`, `get_active_dataset(store)`, `set_active_dataset(store, dataset)`, `add_derived(store, dataset)`, `list_derived(store)`, `has_active_dataset(store)`.
  - `cache.py`: `cached_price_history(source_name, token, tickers, start, end) -> pd.DataFrame`, a `@st.cache_data` wrapper. Not unit-tested (thin wiring).

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_state.py`:

```python
"""Session state accessors, exercised with a plain dict.

state.py takes the store as an argument precisely so it can be tested without
a running Streamlit script.
"""

import datetime as dt

import pandas as pd
import pytest

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.state import (
    add_derived,
    get_active_dataset,
    get_source_name,
    has_active_dataset,
    init_state,
    list_derived,
    set_active_dataset,
    set_source_name,
)


def _dataset(name="sample"):
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 101.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, name, "local")


def test_init_state_starts_with_no_source_selected():
    store = {}
    init_state(store)
    assert get_source_name(store) is None


def test_init_state_starts_with_no_active_dataset():
    store = {}
    init_state(store)
    assert not has_active_dataset(store)
    assert get_active_dataset(store) is None


def test_init_state_is_idempotent():
    store = {}
    init_state(store)
    set_source_name(store, "eod_api")
    init_state(store)
    assert get_source_name(store) == "eod_api"


def test_set_and_get_source_name():
    store = {}
    init_state(store)
    set_source_name(store, "async_eod")
    assert get_source_name(store) == "async_eod"


def test_set_source_rejects_unknown_name():
    store = {}
    init_state(store)
    with pytest.raises(KeyError):
        set_source_name(store, "nope")


def test_active_dataset_round_trip():
    store = {}
    init_state(store)
    ds = _dataset()
    set_active_dataset(store, ds)
    assert has_active_dataset(store)
    assert get_active_dataset(store) is ds


def test_changing_source_keeps_the_active_dataset():
    """Spec section 5.1: switching source does not invalidate the dataset."""
    store = {}
    init_state(store)
    set_source_name(store, "eod_api")
    ds = _dataset()
    set_active_dataset(store, ds)

    set_source_name(store, "async_eod")

    assert get_active_dataset(store) is ds
    assert get_active_dataset(store).source == "local"


def test_derived_datasets_are_listed_in_insertion_order():
    store = {}
    init_state(store)
    add_derived(store, _dataset("first"))
    add_derived(store, _dataset("second"))
    assert [d.name for d in list_derived(store)] == ["first", "second"]


def test_adding_a_derived_dataset_with_an_existing_name_replaces_it():
    store = {}
    init_state(store)
    add_derived(store, _dataset("same"))
    add_derived(store, _dataset("same"))
    assert len(list_derived(store)) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.state'`

- [ ] **Step 3: Write `portfolio_ui/state.py`**

```python
"""Typed access to the session store.

Every function takes the store explicitly (st.session_state in the app, a dict
in tests) so this module never imports streamlit and stays unit-testable.
"""

from __future__ import annotations

from typing import MutableMapping

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.registry import SELECTABLE_SOURCES

SOURCE_KEY = "source_name"
ACTIVE_KEY = "active_dataset"
DERIVED_KEY = "derived_datasets"


def init_state(store: MutableMapping) -> None:
    """Seed missing keys without disturbing existing ones."""
    store.setdefault(SOURCE_KEY, None)
    store.setdefault(ACTIVE_KEY, None)
    store.setdefault(DERIVED_KEY, {})


def get_source_name(store: MutableMapping) -> str | None:
    return store.get(SOURCE_KEY)


def set_source_name(store: MutableMapping, name: str) -> None:
    """Select the source used for the *next* fetch.

    Deliberately does not touch the active dataset: spec section 5.1 keeps a
    fetched dataset valid, with its own provenance, after the selector moves.
    """
    if name not in SELECTABLE_SOURCES:
        raise KeyError(f"unknown source '{name}'")
    store[SOURCE_KEY] = name


def get_active_dataset(store: MutableMapping) -> ActiveDataset | None:
    return store.get(ACTIVE_KEY)


def set_active_dataset(store: MutableMapping, dataset: ActiveDataset) -> None:
    store[ACTIVE_KEY] = dataset


def has_active_dataset(store: MutableMapping) -> bool:
    return store.get(ACTIVE_KEY) is not None


def add_derived(store: MutableMapping, dataset: ActiveDataset) -> None:
    """Register a computed dataset (e.g. a backtest equity curve)."""
    store.setdefault(DERIVED_KEY, {})[dataset.name] = dataset


def list_derived(store: MutableMapping) -> list[ActiveDataset]:
    return list(store.get(DERIVED_KEY, {}).values())
```

- [ ] **Step 4: Write `portfolio_ui/cache.py`**

```python
"""Streamlit caching around the pure source calls.

Kept separate from sources/ so that package stays importable and testable
without streamlit. This module is thin wiring and is not unit-tested.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from portfolio_ui.sources.registry import build_source

CACHE_TTL_SECONDS = 3600


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_price_history(
    source_name: str,
    token: str,
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Fetch price history, memoized on every argument.

    tickers is a tuple because cache keys must be hashable.
    """
    source = build_source(source_name, token=token)
    return source.price_history(list(tickers), start, end)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_state.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add portfolio_ui/state.py portfolio_ui/cache.py tests/ui/test_state.py
git commit -m "feat: add session state accessors and cached fetch layer"
```

---

### Task 11: Plotly chart builders

**Files:**
- Create: `portfolio_ui/charts.py`
- Test: `tests/ui/test_charts.py`

**Interfaces:**
- Consumes: `portfolio_ui.dataset.ActiveDataset` (Task 9).
- Produces: `price_history_figure(dataset, rebased=False) -> plotly.graph_objects.Figure`, `latest_prices_figure(series, title) -> Figure`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_charts.py`:

```python
"""Chart builders are asserted on structure, never rendered."""

import pandas as pd

from portfolio_ui.charts import latest_prices_figure, price_history_figure
from portfolio_ui.dataset import dataset_from_frame


def _dataset():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 110.0], "MSFT.US": [400.0, 420.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, "sample", "local")


def test_price_history_figure_has_one_trace_per_ticker():
    fig = price_history_figure(_dataset())
    assert len(fig.data) == 2
    assert {trace.name for trace in fig.data} == {"AAPL.US", "MSFT.US"}


def test_price_history_figure_titles_the_axes():
    fig = price_history_figure(_dataset())
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Price"


def test_price_history_figure_rebases_to_100():
    fig = price_history_figure(_dataset(), rebased=True)
    for trace in fig.data:
        assert trace.y[0] == 100.0
    assert fig.layout.yaxis.title.text == "Rebased to 100"


def test_latest_prices_figure_is_a_bar_per_ticker():
    series = pd.Series({"AAPL.US": 190.0, "MSFT.US": 410.0}, dtype="float64")
    fig = latest_prices_figure(series, title="Latest")
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["AAPL.US", "MSFT.US"]
    assert fig.layout.title.text == "Latest"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_charts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.charts'`

- [ ] **Step 3: Write `portfolio_ui/charts.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_charts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/charts.py tests/ui/test_charts.py
git commit -m "feat: add plotly chart builders"
```

---

### Task 12: App shell — navigation, sidebar and the source gate

**Files:**
- Create: `portfolio_ui/app.py`
- Create: `portfolio_ui/sidebar.py`
- Create: `portfolio_ui/views/__init__.py`
- Create: `portfolio_ui/views/placeholders.py`
- Test: `tests/ui/test_sidebar.py`

**Interfaces:**
- Consumes: `portfolio_ui.state` (Task 10), `portfolio_ui.sources.registry` (Task 8), `portfolio_ui.dataset` (Task 9).
- Produces:
  - `sidebar.source_option_label(info) -> str` and `sidebar.dataset_summary_lines(dataset) -> list[str]` — pure formatting helpers, tested.
  - `sidebar.render(store) -> None` — Streamlit wiring, untested.
  - `app.py` — the `streamlit run` entry point.

The five feature pages beyond Data are stubs in this plan; Plans 2-4 replace them.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_sidebar.py`:

```python
"""Pure formatting helpers behind the sidebar."""

import pandas as pd

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.sidebar import dataset_summary_lines, source_option_label
from portfolio_ui.sources.registry import describe_sources


def _dataset():
    frame = pd.DataFrame(
        {"AAPL.US": [100.0, 101.0], "MSFT.US": [400.0, 401.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    return dataset_from_frame(frame, "sample", "eod_api")


def test_available_source_label_is_just_the_name():
    info = {i.name: i for i in describe_sources(token="k")}["eod_api"]
    assert source_option_label(info) == "eod_api"


def test_unavailable_source_label_states_the_reason():
    info = {i.name: i for i in describe_sources(token="")}["eod_api"]
    label = source_option_label(info)
    assert "eod_api" in label
    assert "EOD_API_KEY is not set" in label


def test_dataset_summary_lines_report_shape_and_provenance():
    lines = dataset_summary_lines(_dataset())
    assert any("sample" in line for line in lines)
    assert any("2 cols" in line for line in lines)
    assert any("eod_api" in line for line in lines)


def test_dataset_summary_lines_include_notes():
    frame = _dataset().prices
    ds = dataset_from_frame(frame, "sample", "async_eod", notes=("sliced client-side",))
    assert any("sliced client-side" in line for line in dataset_summary_lines(ds))


def test_dataset_summary_lines_handle_no_dataset():
    assert dataset_summary_lines(None) == ["No active dataset"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_sidebar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.sidebar'`

- [ ] **Step 3: Write `portfolio_ui/sidebar.py`**

```python
"""The persistent sidebar: source selection and active dataset summary."""

from __future__ import annotations

from typing import MutableMapping

import streamlit as st

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.registry import SourceInfo, describe_sources
from portfolio_ui.state import get_active_dataset, get_source_name, set_source_name


def source_option_label(info: SourceInfo) -> str:
    """Name the source, and say why it cannot be used when that applies."""
    if info.available:
        return info.name
    return f"{info.name} - unavailable: {info.reason}"


def dataset_summary_lines(dataset: ActiveDataset | None) -> list[str]:
    """The few lines describing what is currently loaded."""
    if dataset is None:
        return ["No active dataset"]

    lines = [
        f"**{dataset.name}**",
        f"{len(dataset.tickers)} cols - {dataset.start:%Y-%m-%d} to {dataset.end:%Y-%m-%d}",
        f"via {dataset.source} - {dataset.frequency}",
    ]
    lines.extend(f"note: {note}" for note in dataset.notes)
    return lines


def render(store: MutableMapping) -> None:
    """Draw the sidebar. Streamlit wiring only."""
    with st.sidebar:
        st.subheader("Data source")

        infos = describe_sources()
        by_label = {source_option_label(info): info for info in infos}
        current = get_source_name(store)
        current_label = next(
            (label for label, info in by_label.items() if info.name == current), None
        )
        options = list(by_label)

        chosen_label = st.selectbox(
            "Choose a client before fetching",
            options=options,
            index=options.index(current_label) if current_label else None,
            placeholder="Select a source...",
            key="source_selectbox",
        )

        if chosen_label:
            info = by_label[chosen_label]
            if info.available:
                set_source_name(store, info.name)
            else:
                st.warning(info.reason)

        st.divider()
        st.subheader("Active dataset")
        for line in dataset_summary_lines(get_active_dataset(store)):
            st.caption(line)
```

- [ ] **Step 4: Write the placeholder pages**

Create `portfolio_ui/views/__init__.py`:

```python
"""Streamlit page modules. Wiring only - logic lives one layer down."""
```

Create `portfolio_ui/views/placeholders.py`:

```python
"""Stubs for the pages built in later plans.

Each states what it will do so the navigation shell is complete and honest.
"""

import streamlit as st

_COMING = {
    "Analysis": "Performance statistics, drawdown episodes and calendar returns.",
    "Risk": "VaR, expected shortfall and Monte Carlo simulation.",
    "Optimization": "Eight optimization objectives, bounds and covariance estimators.",
    "Backtest": "Rebalanced backtests with momentum universe selection.",
    "Market Data": "Fundamentals, dividends, earnings and macro series (eod_api only).",
}


def _stub(title: str) -> None:
    st.title(title)
    st.info(f"Not built yet. Planned: {_COMING[title]}")


def analysis_page():
    _stub("Analysis")


def risk_page():
    _stub("Risk")


def optimization_page():
    _stub("Optimization")


def backtest_page():
    _stub("Backtest")


def market_data_page():
    _stub("Market Data")
```

- [ ] **Step 5: Write `portfolio_ui/app.py`**

```python
"""Entry point: streamlit run portfolio_ui/app.py"""

import streamlit as st

from portfolio_ui import sidebar
from portfolio_ui.views.data_page import data_page
from portfolio_ui.views.placeholders import (
    analysis_page,
    backtest_page,
    market_data_page,
    optimization_page,
    risk_page,
)
from portfolio_ui.state import init_state


def main() -> None:
    st.set_page_config(page_title="Portfolio Construction", layout="wide")
    init_state(st.session_state)
    sidebar.render(st.session_state)

    pages = [
        st.Page(data_page, title="Data", default=True),
        st.Page(market_data_page, title="Market Data"),
        st.Page(analysis_page, title="Analysis"),
        st.Page(risk_page, title="Risk"),
        st.Page(optimization_page, title="Optimization"),
        st.Page(backtest_page, title="Backtest"),
    ]
    st.navigation(pages).run()


main()
```

`data_page` does not exist yet — Task 13 creates it. The app will not start until then; that is expected.

- [ ] **Step 6: Run the sidebar tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_sidebar.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add portfolio_ui/app.py portfolio_ui/sidebar.py portfolio_ui/views/ tests/ui/test_sidebar.py
git commit -m "feat: add app shell, sidebar source gate and page stubs"
```

---

### Task 13: The Data page

**Files:**
- Create: `portfolio_ui/views/data_page.py`
- Create: `portfolio_ui/guards.py`
- Test: `tests/ui/test_guards.py`
- Modify: `USAGE.md`

**Interfaces:**
- Consumes: `portfolio_ui.state`, `portfolio_ui.dataset`, `portfolio_ui.cache`, `portfolio_ui.charts`, `portfolio_ui.sources.*`.
- Produces:
  - `guards.parse_tickers(raw: str) -> list[str]` — splits on commas/whitespace/newlines, strips, uppercases, de-duplicates preserving order.
  - `guards.capability_blocked_reason(source, capability) -> str | None` — the tooltip text for a disabled action, `None` when allowed.
  - `guards.require_active_dataset(store) -> ActiveDataset | None` — the shared "no dataset yet" guard used by every later page.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_guards.py`:

```python
"""Input parsing and the disabled-action reasons shown as tooltips."""

import pandas as pd

from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.guards import (
    capability_blocked_reason,
    parse_tickers,
    require_active_dataset,
)
from portfolio_ui.sources.base import Capability
from portfolio_ui.sources.eod_api_source import EodApiSource
from portfolio_ui.sources.market_access_source import MarketAccessSource
from portfolio_ui.state import init_state, set_active_dataset


def test_parse_tickers_splits_on_commas_and_whitespace():
    assert parse_tickers("AAPL.US, MSFT.US\nGOOG.US  TSLA.US") == [
        "AAPL.US",
        "MSFT.US",
        "GOOG.US",
        "TSLA.US",
    ]


def test_parse_tickers_uppercases_and_strips():
    assert parse_tickers("  aapl.us ") == ["AAPL.US"]


def test_parse_tickers_deduplicates_preserving_order():
    assert parse_tickers("AAPL.US, MSFT.US, AAPL.US") == ["AAPL.US", "MSFT.US"]


def test_parse_tickers_of_empty_input_is_empty():
    assert parse_tickers("   ") == []


def test_capability_blocked_reason_is_none_when_supported():
    source = EodApiSource(token="k")
    assert capability_blocked_reason(source, Capability.LATEST) is None


def test_capability_blocked_reason_names_source_and_capability():
    source = MarketAccessSource()
    reason = capability_blocked_reason(source, Capability.LATEST)
    assert "market_access" in reason
    assert "latest" in reason


def test_capability_blocked_reason_reports_missing_key_first():
    source = EodApiSource(token="")
    reason = capability_blocked_reason(source, Capability.PRICE_HISTORY)
    assert "EOD_API_KEY" in reason


def test_require_active_dataset_returns_none_when_absent():
    store = {}
    init_state(store)
    assert require_active_dataset(store) is None


def test_require_active_dataset_returns_the_dataset():
    store = {}
    init_state(store)
    frame = pd.DataFrame(
        {"AAPL.US": [1.0, 2.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
    )
    ds = dataset_from_frame(frame, "sample", "local")
    set_active_dataset(store, ds)
    assert require_active_dataset(store) is ds
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_guards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.guards'`

- [ ] **Step 3: Write `portfolio_ui/guards.py`**

```python
"""Shared input parsing and gating logic. No streamlit import."""

from __future__ import annotations

import re
from typing import MutableMapping

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.base import Capability
from portfolio_ui.state import get_active_dataset

_SEPARATORS = re.compile(r"[,\s]+")


def parse_tickers(raw: str) -> list[str]:
    """Turn free-typed ticker input into a clean, ordered, unique list."""
    parts = [p.strip().upper() for p in _SEPARATORS.split(raw or "") if p.strip()]
    seen: dict[str, None] = {}
    for part in parts:
        seen.setdefault(part, None)
    return list(seen)


def capability_blocked_reason(source, capability: Capability) -> str | None:
    """Why this action is disabled, or None if it is allowed.

    Actions are never hidden (spec section 4.3) - they are disabled with the
    reason shown, so a feature never looks nonexistent.
    """
    if not source.is_available():
        return source.unavailable_reason()
    if not source.supports(capability):
        return (
            f"{source.name} does not support {capability.value} - "
            f"switch to a source that does"
        )
    return None


def require_active_dataset(store: MutableMapping) -> ActiveDataset | None:
    """The active dataset, or None when the user has not built one yet."""
    return get_active_dataset(store)
```

- [ ] **Step 4: Write `portfolio_ui/views/data_page.py`**

```python
"""Build the active dataset: fetch from a source, upload a file, or reload."""

from __future__ import annotations

import datetime as dt
import os

import streamlit as st

from portfolio_ui.cache import cached_price_history
from portfolio_ui.charts import latest_prices_figure, price_history_figure
from portfolio_construction.stats import annualization_factor
from portfolio_ui.dataset import (
    dataset_from_frame,
    list_saved,
    load_dataset,
    notes_for_fetch,
    save_dataset,
)
from portfolio_ui.guards import capability_blocked_reason, parse_tickers
from portfolio_ui.sources.base import Capability, SourceError
from portfolio_ui.sources.local_source import LocalSource
from portfolio_ui.sources.registry import build_source
from portfolio_ui.state import get_active_dataset, get_source_name, set_active_dataset


def _fetch_tab(store, source):
    raw = st.text_area(
        "Tickers", value="AAPL.US, MSFT.US, GOOG.US", help="Comma or newline separated"
    )
    tickers = parse_tickers(raw)

    col_start, col_end = st.columns(2)
    start = col_start.date_input("From", value=dt.date(2015, 1, 1))
    end = col_end.date_input("To", value=dt.date.today())

    name = st.text_input("Dataset name", value="working-set")
    blocked = capability_blocked_reason(source, Capability.PRICE_HISTORY)

    if st.button("Fetch price history", disabled=bool(blocked) or not tickers, help=blocked):
        with st.spinner(f"Fetching {len(tickers)} tickers via {source.name}..."):
            try:
                frame = cached_price_history(
                    source.name,
                    os.environ.get("EOD_API_KEY", ""),
                    tuple(tickers),
                    start,
                    end,
                )
            except SourceError as exc:
                st.error(str(exc))
                return

        # The cache returns the frame, so the dataset is assembled here rather
        # than through build_dataset - same notes, no second fetch.
        dataset = dataset_from_frame(
            frame,
            name,
            source.name,
            notes=notes_for_fetch(source.name, tickers, frame.columns),
        )
        set_active_dataset(store, dataset)
        st.success(f"Loaded {dataset.summary()}")
        for note in dataset.notes:
            st.warning(note)


def _point_in_time_tab(source):
    raw = st.text_area("Tickers", value="AAPL.US, MSFT.US", key="pit_tickers")
    tickers = parse_tickers(raw)
    on = st.date_input("As of", value=dt.date.today() - dt.timedelta(days=1))

    close_blocked = capability_blocked_reason(source, Capability.CLOSE_AT)
    latest_blocked = capability_blocked_reason(source, Capability.LATEST)

    col_close, col_latest = st.columns(2)
    if col_close.button("Close on date", disabled=bool(close_blocked), help=close_blocked):
        try:
            series = source.close_at(tickers, on)
            st.plotly_chart(
                latest_prices_figure(series, f"Close on {on:%Y-%m-%d}"),
                use_container_width=True,
            )
            st.dataframe(series.rename("close"))
        except SourceError as exc:
            st.error(str(exc))

    if col_latest.button("Latest prices", disabled=bool(latest_blocked), help=latest_blocked):
        try:
            series = source.latest(tickers)
            st.plotly_chart(
                latest_prices_figure(series, "Latest prices"), use_container_width=True
            )
            st.dataframe(series.rename("close"))
        except SourceError as exc:
            st.error(str(exc))


def _sovereign_tab(source):
    countries = parse_tickers(st.text_input("Countries (ISO codes)", value="US, FR, DE"))
    tenors_raw = st.text_input("Tenors (years)", value="2, 5, 10")
    tenors = [int(t) for t in parse_tickers(tenors_raw) if t.isdigit()]
    on = st.date_input("As of", value=dt.date.today() - dt.timedelta(days=1), key="sov_date")

    blocked = capability_blocked_reason(source, Capability.SOVEREIGN)
    if st.button("Fetch yields", disabled=bool(blocked), help=blocked):
        try:
            st.dataframe(source.sovereign_yields(countries, tenors, on))
        except SourceError as exc:
            st.error(str(exc))


def _upload_tab(store):
    uploaded = st.file_uploader("Price file", type=["csv", "xlsx", "parquet"])
    name = st.text_input("Dataset name", value="uploaded", key="upload_name")
    if uploaded is not None and st.button("Load file"):
        try:
            local = LocalSource.from_upload(uploaded, uploaded.name)
            frame = local.price_history(
                local.available_tickers(), dt.date(1900, 1, 1), dt.date.today()
            )
            dataset = dataset_from_frame(frame, name, local.name)
        except (ValueError, SourceError) as exc:
            st.error(str(exc))
            return
        set_active_dataset(store, dataset)
        st.success(f"Loaded {dataset.summary()}")


def _saved_tab(store):
    active = get_active_dataset(store)
    if active is not None and st.button("Save active dataset"):
        path = save_dataset(active)
        st.success(f"Saved to {path}")

    names = list_saved()
    if not names:
        st.caption("No saved datasets yet.")
        return

    chosen = st.selectbox("Saved datasets", options=names)
    if st.button("Load saved dataset"):
        set_active_dataset(store, load_dataset(chosen))
        st.success(f"Loaded {chosen}")


def data_page() -> None:
    st.title("Data")
    store = st.session_state
    source_name = get_source_name(store)

    if source_name is None:
        st.info("Choose a data source in the sidebar to begin.")
        return

    source = build_source(source_name)
    st.caption(f"Selected source: **{source.name}**")

    fetch, point_in_time, sovereign, upload, saved = st.tabs(
        ["Price history", "Point in time", "Sovereign yields", "Upload", "Saved"]
    )
    with fetch:
        _fetch_tab(store, source)
    with point_in_time:
        _point_in_time_tab(source)
    with sovereign:
        _sovereign_tab(source)
    with upload:
        _upload_tab(store)
    with saved:
        _saved_tab(store)

    active = get_active_dataset(store)
    if active is not None:
        st.divider()
        st.subheader("Active dataset")

        # Inferred metadata (spec section 5): frequency drives the annualization
        # factor every downstream statistic uses.
        meta_cols = st.columns(4)
        meta_cols[0].metric("Tickers", len(active.tickers))
        meta_cols[1].metric("Observations", len(active.prices))
        meta_cols[2].metric("Frequency", active.frequency)
        meta_cols[3].metric("Annualization", annualization_factor(active.frequency))

        rebased = st.checkbox("Rebase to 100", value=False)
        st.plotly_chart(price_history_figure(active, rebased=rebased), use_container_width=True)
        st.dataframe(active.prices.tail(20))
```

- [ ] **Step 5: Run the guard tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ui/test_guards.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass — the pre-existing core suite plus the new UI suite.

- [ ] **Step 7: Launch the app and confirm it runs**

Run: `.venv/Scripts/python.exe -m streamlit run portfolio_ui/app.py`

Verify by hand:
1. The app starts and the sidebar shows a source selector with no source chosen.
2. The Data page says "Choose a data source in the sidebar to begin."
3. Selecting `market_access` enables the Price history fetch button, and leaves "Latest prices" disabled with a tooltip naming the reason.
4. With `EOD_API_KEY` unset, `eod_api` and `async_eod` appear in the list labelled unavailable and their actions stay disabled — the app does not crash.
5. Uploading a CSV of prices sets the active dataset and draws the chart.
6. Saving then reloading a dataset reproduces the same chart.

- [ ] **Step 8: Document the UI in USAGE.md**

Add a new section after the "Installation" section:

````markdown
## Streamlit UI

Install the UI extra and launch:

```powershell
pip install -e ".[ui]"
streamlit run portfolio_ui/app.py
```

Pick a data source in the sidebar first — `async_eod` (fast, parallel),
`eod_api` (slower, far more endpoints) or `market_access` (yfinance, no key
needed). The Data page builds an **active dataset** that every other page
reuses, so you fetch once and analyze many times. Uploads (`.csv`, `.xlsx`,
`.parquet`) and saved datasets work without any API key.

Switching source later does not discard the active dataset: it keeps the
provenance of whichever client fetched it, and only the next fetch uses the new
selection.

Saved datasets go to `data/datasets/`, overridable with `PORTFOLIO_UI_DATA_DIR`.
````

- [ ] **Step 9: Commit**

```bash
git add portfolio_ui/guards.py portfolio_ui/views/data_page.py tests/ui/test_guards.py USAGE.md
git commit -m "feat: add Data page with source-gated fetch, upload and persistence"
```

---

## Definition of done

- [ ] `.venv/Scripts/python.exe -m pytest -q` passes.
- [ ] `.venv/Scripts/python.exe -m streamlit run portfolio_ui/app.py` starts with no source selected and no API key set.
- [ ] Every step in Task 13 Step 7's manual checklist verified.
- [ ] No `streamlit` or `plotly` import exists anywhere under `portfolio_construction/`.
- [ ] `KO6RI8AXUX0HDGC2` appears nowhere in the working tree.

## What this plan does not cover

Spec slices 4-9 — the Analysis, Risk, Optimization, Backtest and Market Data
pages, and the §7.2 backtest wiring — are Plans 2 to 4. The placeholder pages
from Task 12 name what replaces them.
