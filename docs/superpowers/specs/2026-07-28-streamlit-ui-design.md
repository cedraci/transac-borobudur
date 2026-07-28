# Streamlit UI for the Portfolio Construction Toolkit — Design

Date: 2026-07-28
Status: approved, ready for implementation planning

## 1. Purpose

Build a Streamlit application that makes the whole `portfolio_construction`
package usable from a browser as a **working research tool**: pull market data,
optimize, backtest, and analyze portfolios end to end, keeping results between
steps.

The user selects a market-data source before doing anything else. Both EOD
clients — `async_eod.py` and `eod_api.py` — are first-class choices.

### Success criteria

1. Every public function in the package is reachable from the UI, or is
   explicitly marked unavailable with a stated reason.
2. A dataset fetched once is reused by the optimization, backtest, analysis and
   risk pages without re-fetching.
3. Switching the source changes which client runs, and the active source is
   visible at all times.
4. The logic layer is covered by tests that run without network access.

## 2. Constraints

- Core `portfolio_construction` package must remain installable and usable
  without Streamlit or Plotly.
- No authentication, no multi-user deployment, no live streaming.
- API calls cost money and quota: caching and offline work are requirements,
  not nice-to-haves.

## 3. The source asymmetry

The two EOD clients are **not** interchangeable. They overlap on four
capabilities only:

| Capability | `async_eod` | `eod_api` |
|---|---|---|
| Price history | `get_full_history` | `adjusted_prices` / `download_universe` |
| Close on a date | `get_historical` | `adjClose_atDate` / `recursive_adjClose_atDate` |
| Realtime / last price | `get_realtime` | `last_prices_universe` / `last_prices` |
| Sovereign bond yields | `sovereign_bonds` | `get_SovBond` |

Everything else — `fundamentals`, `index_constituents`, `search_query`, `ohlcv`,
`dividends`, `splits`, `earnings_calendar`, `macro_events`, `macro_indicators`,
`fixed_income_etf`, `stock_historical_dividend_yield` — exists **only** in
`eod_api.py`. `async_eod.py` has no unique features; it is purely the fast
parallel path.

**Decision.** A thin adapter exposes the four shared capabilities behind one
interface; the source selector routes those calls. The `eod_api`-only features
live in their own page that stays available regardless of the selected source.
This gives full feature coverage while being honest about the asymmetry.

Rejected alternatives:

- *Strict gate* (hide everything the chosen source cannot do): selecting
  `async_eod` would shrink the app to four features and make most of the
  toolkit unreachable.
- *Preference with auto-fallback*: loses visibility into which client actually
  ran. Unacceptable for a research tool where provenance matters.

## 4. Architecture

### 4.1 Layout

```
portfolio_construction/       # core library — unchanged except the fixes in §7
portfolio_ui/
  app.py                      # entry point: st.navigation, sidebar, session bootstrap
  sources/
    base.py                   # PriceSource protocol, Capability enum, exceptions
    async_eod_source.py       # wraps portfolio_construction.async_eod
    eod_api_source.py         # wraps portfolio_construction.eod_api
    market_access_source.py   # wraps portfolio_construction.market_access
    local_source.py           # file upload + saved-dataset store
    registry.py               # name -> source instance, capability lookup
  dataset.py                  # ActiveDataset dataclass, build/validate/save/load
  state.py                    # typed session-state accessors
  cache.py                    # @st.cache_data wrappers around pure source calls
  charts.py                   # plotly figure builders
  formatting.py               # results -> display-ready frames
  pages/                      # thin wiring only, no logic
tests/ui/
  fixtures/                   # recorded JSON API payloads
```

Dependency direction is one-way: `portfolio_ui` imports `portfolio_construction`,
never the reverse.

### 4.2 The adapter

```python
class Capability(StrEnum):
    PRICE_HISTORY = "price_history"
    CLOSE_AT      = "close_at"
    LATEST        = "latest"
    SOVEREIGN     = "sovereign_yields"

class PriceSource(Protocol):
    name: str
    capabilities: frozenset[Capability]

    def price_history(self, tickers: list[str], start: date, end: date) -> pd.DataFrame: ...
    def close_at(self, tickers: list[str], on: date) -> pd.Series: ...
    def latest(self, tickers: list[str]) -> pd.Series: ...
    def sovereign_yields(self, countries: list[str], tenors: list[int], on: date) -> pd.DataFrame: ...
```

**Normalization is the adapter's core job.** The raw clients disagree on return
shape: `async_eod.get_full_history` returns a *list of Series*,
`eod_api.adjusted_prices` returns a `merge_asof`-joined frame, and
`market_access.yahooFinance_historical_data` returns OHLCV for a single ticker.
Each wrapper normalizes to a single contract:

- `price_history` → wide `DataFrame`, ascending `DatetimeIndex`, `float64`,
  one column per ticker named exactly as requested.
- `close_at`, `latest` → `Series` indexed by ticker, `float64`.
- `sovereign_yields` → `DataFrame` indexed by bond ticker with a `Rates` column.

Source wrappers contain **no Streamlit import**, which is what makes them
testable.

### 4.3 Capability matrix

| Source | price_history | close_at | latest | sovereign |
|---|:--:|:--:|:--:|:--:|
| `async_eod` | yes | yes | yes | yes |
| `eod_api` | yes | yes | yes | yes |
| `market_access` | yes | no | no | no |
| local / upload | yes | no | no | no |

Unsupported actions render **disabled with a tooltip naming the reason**
(e.g. "async_eod has no fundamentals endpoint — switch to eod_api"). They are
never hidden: hiding makes a feature look nonexistent.

`local` implements the same `PriceSource` protocol so it passes the contract
test of §8, but it is not offered in the source selector — see §5.1.

### 4.4 Asymmetries the adapter documents rather than hides

- `async_eod.get_full_history` ignores start/end and always requests
  1990→today. The wrapper fetches full history, slices client-side, and records
  a note in the dataset metadata.
- `eod_api` takes the token as an explicit argument; `async_eod` reads it at
  import time. Both wrappers read `EOD_API_KEY` themselves and pass it down, so
  pages never handle tokens.

## 5. Pages

Navigation via `st.navigation`, with a persistent sidebar showing the selected
source, its key status, and the active dataset summary.

```
┌─ sidebar ────────────┐┌─ main ──────────────────────────┐
│ Source: [async_eod ▾]││ ⚠ No active dataset — build one │
│ ● key detected       ││    on the Data page             │
│ ─────────────────    ││                                 │
│ Dataset: MSCI-8      ││                                 │
│ 8 cols · 2015→2026   ││                                 │
│ via eod_api · cached ││                                 │
│ ─────────────────    ││                                 │
│ Data / Market Data   ││                                 │
│ Analysis / Risk      ││                                 │
│ Optimization         ││                                 │
│ Backtest             ││                                 │
└──────────────────────┘└─────────────────────────────────┘
```

Pages downstream of Data refuse to render controls when no dataset is active;
they show the warning above with a jump link instead of failing on a `None`.

### 5.1 Selecting the source

The source selector starts **empty**. Until a source is chosen, the Data page's
fetch controls are disabled and the sidebar prompts for a selection — this is
the "choose a client before running anything" gate.

Changing the source mid-session does **not** invalidate the active dataset. The
dataset records the source that actually produced it in `ActiveDataset.source`,
and the sidebar keeps showing that provenance ("via eod_api") even after the
selector moves to another source. Only the *next* fetch uses the new selection.
This keeps a long-running analysis reproducible while letting the user reach a
capability the current source lacks.

Local/uploaded datasets are not a selectable "source" in the same sense: upload
and saved-dataset loading are always available on the Data page regardless of
the selected source, and produce datasets whose `source` is recorded as
`upload:<filename>` or `saved:<name>`.

| Page | Functions surfaced |
|---|---|
| **Data** | the four adapter capabilities; file upload; saved-dataset store; `time_series_frequence_inference` and `annualization_factor` displayed as inferred metadata |
| **Market Data** | `fundamentals`, `index_constituents`, `search_query`, `ohlcv`, `dividends`, `splits`, `earnings_calendar`, `macro_events`, `macro_indicators`, `fixed_income_etf`, `stock_historical_dividend_yield`; `yahooFinance_company_snapshot`, `alphaVantage_company_snapshot`, `historical_fx`; `adjust_for_corporates_actions`, `clean_data` |
| **Analysis** | `stats_report`, `annualized_return`, `annualized_volatility`, `annualized_sharpe_ratio`, `historical_drawdown`, `maximum_drawdown`, `monthly_returns`, `calendar_performances`, `compute_drawdown_periods`, `drawdowns_table`, `rebased_from_date`, `cagr_rolled` |
| **Risk** | `historical_var`, `parametric_var`, `monteCarlo_var`, `historical_es`, `parametric_es`, `monteCarlo_es`, `gbm_multiple_path`, `portfolio_path_cholesky`, `covariance_parametric_var`, `get_stressed_covariance` |
| **Optimization** | `portfolio_optimization` (all eight objectives, bounds editor, `cov_mat` ∈ sample/shrunked/gerber, target return); `risk_contribution`; `InverseVol`; `estimate_bayes_stein`; `gerber_statistic`, `gerber_correlation_matrix`; `optimize_capital_protection`; `max_momentum_optimization`, `longshort_momentum_optimization` |
| **Backtest** | `Backtest`, `RebalancingCalendar`, `universe_selection`, `universe_selection_v2`, `momentum`, `naive_momentum` → equity curve, weights-over-time area chart, hand-off of the strategy series to Analysis/Risk as a derived dataset |

This covers every public function in the package. `utils.Portfolio` is internal
to the backtest loop and is not surfaced directly.

## 6. State, caching, persistence, errors

### 6.1 Active dataset

```python
@dataclass(frozen=True)
class ActiveDataset:
    name: str
    prices: pd.DataFrame          # normalized per §4.2
    source: str                   # provenance: which client actually ran
    tickers: tuple[str, ...]
    start: date
    end: date
    fetched_at: datetime
    frequency: str                # via time_series_frequence_inference
    notes: tuple[str, ...]        # e.g. "async_eod ignored start/end; sliced client-side"
```

Frozen, so it is replaced wholesale rather than mutated. `state.py` exposes
typed accessors (`get_active_dataset`, `set_active_dataset`, `list_derived`);
pages never touch raw `st.session_state` keys.

Backtest results register as **derived datasets**, making "backtest a strategy →
analyze its equity curve" a two-click flow.

### 6.2 Caching

`cache.py` holds `@st.cache_data` wrappers keyed on
`(source_name, tickers, start, end)` with a 1-hour TTL. Every expensive
operation — fetch, backtest, Monte Carlo — sits behind an explicit **Run**
button with a spinner, so Streamlit's rerun-on-widget-change does not
re-trigger a 5000-path simulation when a slider moves.

### 6.3 Persistence

Saved datasets are written to `data/datasets/<name>.parquet` with a `.json`
metadata sidecar. The directory is gitignored and overridable via
`PORTFOLIO_UI_DATA_DIR`.

### 6.4 Errors

Sources raise typed exceptions defined in `sources/base.py`:

- `SourceUnavailable` — required API key missing.
- `TickerNotFound` — upstream returned no data for a ticker.
- `UpstreamError` — HTTP or parse failure.

Pages catch these and render `st.error` naming the failing tickers; tracebacks
never reach the browser. Partial failures build the dataset from surviving
tickers with a visible warning listing what was dropped. This matters because
`eod_api.adjusted_prices` currently swallows failures in a bare `except:` with a
`print` that would be invisible in a browser.

### 6.5 Secrets

- `EOD_API_KEY` — both EOD sources.
- `ALPHAVANTAGE_API_KEY` — `market_access`, replacing the hardcoded key.

The sidebar shows detected/missing per source. A missing key leaves the source
selectable but disables its actions with the reason stated.

## 7. Changes to the core package

These are targeted fixes to code paths the UI depends on. Each gets a
regression test.

### 7.1 Bugs

| Location | Problem | Fix |
|---|---|---|
| `async_eod.py:163` | `full_history_url` formats the end date `%y-%m-%d`, sending `to=26-07-28` | use `%Y-%m-%d` |
| `eod_api.py:188-189` | `last_prices_universe` calls `.values` then indexes `df['close']` on a numpy array → `TypeError` on every call | drop the `.values` call and return the `close` column of the DataFrame |
| `eod_api.py:169` | `last_prices` passes the module global `tok` instead of its `token` parameter | pass `token` |
| `async_eod.py:240-241` | `sovereign_bonds` does `tmp.update(elt)` although `async_sovereign_bond` returns `None` on no data | skip `None` results |

### 7.2 Unreachable features to wire in

1. **`RebalancingCalendar` is never used by `Backtest`.**
   `initialize_parameters` hardcodes end-of-month rebalancing. Thread the
   calendar through so a frequency selector (`eos`/`eoq`/`eom`/`eow`/`bim`)
   drives the backtest.
2. **Momentum universe selection is unreachable.** `target_weights` accepts
   `stock_picking`, but `simulations` never passes it. Thread it through so the
   toggle works.

### 7.3 Not wired in

The `bayes` flag on `Backtest.simulations` is accepted and ignored — Bayes-Stein
shrinkage is implemented as `estimate_bayes_stein` but not plugged into
`portfolio_optimization`. Implementing it is real quantitative work, not
wiring, and is **out of scope**. The flag is deliberately omitted from the UI
rather than shown as a control that silently does nothing.

### 7.4 Hardcoded secret

`market_access.py` contains an AlphaVantage API key in source at lines 7, 13 and
40, committed to git. Replace with `ALPHAVANTAGE_API_KEY` read from the
environment. Note that the key remains in git history; rotating it at the
provider is recommended and is the user's call.

### 7.5 Documentation

The `portfolio_backtest` module docstring lists `som` and `dow` as rebalancing
codes, but `define_calendar` implements `eos`, `eoq`, `eom`, `eow` and `bim`.
Correct the docstring.

## 8. Testing

Tests live in `tests/ui/` and run without network access.

- **Fixtures.** JSON payloads recorded once from real API responses, stored in
  `tests/ui/fixtures/`.
- **Contract test.** One test parameterized across all four sources: for each
  declared capability, assert the normalized shape from §4.2. This is what makes
  "swap the source, everything else keeps working" verified rather than hoped.
- **Source wrappers.** Per-source tests against fixtures, including partial
  failure and empty-response handling.
- **`dataset.py`.** Save/load round-trip, metadata correctness, validation
  rejects malformed frames.
- **`charts.py`.** Builders return figures with the expected traces; no
  rendering.
- **Regressions.** One test per fix in §7.1 and per wiring in §7.2.
- **Pages.** Not tested. They stay logic-free by design; any logic worth testing
  belongs in the layer below.

## 9. Packaging

```toml
[project.optional-dependencies]
ui = ["streamlit", "plotly", "pyarrow"]
test = ["pytest"]

[tool.setuptools.packages.find]
include = ["portfolio_construction*", "portfolio_ui*"]
```

Launch: `streamlit run portfolio_ui/app.py`.

USAGE.md gains a section documenting installation of the `ui` extra, the
required environment variables, and the page walkthrough.

## 10. Suggested build order

The scope is broad enough that it should be built in vertical slices, each one
leaving the app runnable:

1. **Core fixes** (§7.1, §7.4, §7.5) with regression tests. No UI yet.
2. **Adapter + contract test** (§4.2, §4.3, §8) — `sources/`, all four
   implementations, fixtures. Still no UI.
3. **Shell + Data page** — `app.py`, `state.py`, `dataset.py`, `cache.py`,
   source selector, fetch, upload, save/load. First runnable app.
4. **Analysis page** — the cheapest consumer of the active dataset, proves the
   shared-dataset flow works.
5. **Optimization page**.
6. **Backtest page** — includes the wiring of §7.2.
7. **Risk page**.
8. **Market Data page** — the `eod_api`-only surface; largest number of
   distinct calls, least coupled to everything else.
9. **USAGE.md documentation** (§9).

## 11. Out of scope

- Authentication and multi-user deployment.
- Live or streaming price updates.
- Writing results back to any external system.
- Implementing Bayes-Stein in the optimizer (§7.3).
- Refactoring core modules beyond the fixes in §7.
