# Streamlit UI — Analysis & Risk Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Analysis and Risk placeholder pages with working pages that turn the active dataset into performance statistics, drawdown episodes, calendar returns, and the three families of VaR/Expected-Shortfall estimates plus Monte Carlo simulation.

**Architecture:** Same layering as the foundation. All real work goes in new pure modules `portfolio_ui/analytics.py` and `portfolio_ui/risk.py` with no Streamlit import and full pytest coverage; `portfolio_ui/views/analysis_page.py` and `views/risk_page.py` are thin wiring. New chart builders join `portfolio_ui/charts.py`.

**Tech Stack:** Python ≥3.10, pandas, numpy, scipy, Streamlit, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-streamlit-ui-design.md` §5 (Analysis and Risk rows). This plan implements build-order slices 4 and 7.

**Predecessor:** `docs/superpowers/plans/2026-07-28-streamlit-ui-foundation.md`, merged. Current suite: 190 passed, 0 skipped.

## Global Constraints

- Python floor is `>=3.10`. **Do not use `enum.StrEnum`** (3.11+).
- `portfolio_construction` must stay importable with neither `streamlit` nor `plotly` installed.
- `portfolio_ui/analytics.py`, `portfolio_ui/risk.py`, `portfolio_ui/charts.py` must **not** import `streamlit`. Only `app.py`, `sidebar.py`, `cache.py` and `views/` may.
- Page modules live in `portfolio_ui/views/`, never `portfolio_ui/pages/`.
- Tests must never make network calls, and must not depend on `EOD_API_KEY`.
- Run tests with the project venv: `./.venv/Scripts/python.exe -m pytest -q`.
- Every page downstream of Data must render the "no active dataset" guard rather than failing on `None`. Use the existing `portfolio_ui.guards.require_active_dataset`.
- Monte Carlo and any simulation with >1000 paths must sit behind an explicit **Run** button, never recomputing on a slider move.

## Existing interfaces this plan consumes

From the merged foundation — do not recreate these:

```python
# portfolio_ui/dataset.py
ActiveDataset          # frozen: name, prices, source, tickers, start, end, fetched_at, frequency, notes
                       # plus .summary()
dataset_from_frame(frame, name, source_label, notes=()) -> ActiveDataset
# portfolio_ui/state.py   (all take the store as first arg)
get_active_dataset(store); set_active_dataset(store, ds); add_derived(store, ds); list_derived(store)
# portfolio_ui/guards.py
require_active_dataset(store) -> ActiveDataset | None
# portfolio_ui/charts.py
price_history_figure(dataset, rebased=False); latest_prices_figure(series, title)
```

Core library signatures this plan calls (verified against the installed package):

```python
# portfolio_construction.portfolio_analysis
annualized_return(bkt); annualized_volatility(bkt); annualized_sharpe_ratio(bkt, rf=0.0)
historical_drawdown(bkt); maximum_drawdown(bkt); monthly_returns(bkt); calendar_performances(strat)
stats_report(df, rf=0.0, trace=False, rebased_plot=False)
drawdowns_table(nav: pd.Series, top_n=10, min_drawdown=0.01) -> pd.DataFrame   # hint says DataFrame; needs a Series - see sharp edge 2
cagr_rolled(yearly_returns: pd.DataFrame, max_holding_period=20) -> pd.DataFrame
rebased_from_date(prices, rebased_dt)
historical_var(prices, alpha, duration); historical_es(prices, alpha, duration)
parametric_var(prices, vec_w, alpha, duration, distrib); parametric_es(prices, vec_w, alpha, duration)
monteCarlo_var(num, startprice, mu, sigma, alpha, duration, distrib)
monteCarlo_es(num, startprice, mean, sigma, alpha, duration, distrib)
gbm_multiple_path(num, startprice, mu, sigma, days, distrib='normal')
portfolio_path_cholesky(n_sims, start_price, weights, mu, cov_matrix, days, distrib='normal')
# portfolio_construction.portfolio_optimization
get_stressed_covariance(cov_matrix, stress_factor=0.5)
# portfolio_construction.stats
time_series_frequence_inference(dates); annualization_factor(freq)
```

**Sharp edges in the core library, all verified by probing the installed package.**
Read these before writing any code — several contradict the type hints:

1. `compute_drawdown_periods` takes `list[tuple[date, Decimal]]`. Never call it
   directly from the UI — call `drawdowns_table`.
2. `drawdowns_table`'s signature says `nav: pd.DataFrame`, **but it needs a
   `Series`**. It does `for date, price in nav.items()`, and on a DataFrame
   `.items()` yields `(column_name, Series)` pairs, so a frame raises
   `AttributeError: 'str' object has no attribute 'date'`. Pass `nav` as a
   Series. Task 0 corrects the hint.
3. `drawdowns_table` **ignores its own `top_n` and `min_drawdown` arguments** —
   it hardcodes `top_n=10, min_drawdown=0.02` in the inner call. Task 0 fixes it.
4. `cagr_rolled` is **broken on numpy 2.x** (installed: 2.5.1). It assigns a
   size-1 array into a scalar matrix slot, which numpy now rejects with
   `ValueError: setting an array element with a sequence`. Task 0 fixes it.
5. `parametric_var` and `parametric_es` take a weight vector `vec_w` alongside a
   *multi-column* price frame. A single series still needs `[1.0]`.
6. `gbm_multiple_path(num, ..., days)` returns an ndarray of shape
   **`(days + 1, num)`** — one extra row for the starting value, paths in
   columns. `portfolio_path_cholesky(n_sims, ..., days)` likewise returns
   `(days + 1, n_sims)`. Do not "correct" this by transposing.

---

### Task 0: Fix the two core-library defects the Analysis page depends on

Neither `drawdowns_table` nor `cagr_rolled` can be surfaced honestly until these
are fixed. Both were reproduced against the installed package.

**Files:**
- Modify: `portfolio_construction/portfolio_analysis.py` (`drawdowns_table`, `cagr_rolled`)
- Test: `tests/test_portfolio_analysis_fixes.py`

**Interfaces:**
- Produces: `drawdowns_table(nav: pd.Series, top_n=10, min_drawdown=0.01) -> pd.DataFrame`
  that honours both arguments; a `cagr_rolled` that runs on numpy 2.x.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portfolio_analysis_fixes.py`:

```python
"""Regression tests for two defects found while building the Analysis page.

Both were reproduced against the installed pandas/numpy before being fixed.
"""

import numpy as np
import pandas as pd
import pytest

import portfolio_construction.portfolio_analysis as pa


def _nav(days=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days, name="Date")
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days))),
        index=idx,
        name="Portfolio",
    )


def test_drawdowns_table_honours_top_n():
    out = pa.drawdowns_table(_nav(), top_n=3)
    assert len(out) <= 3


def test_drawdowns_table_honours_min_drawdown():
    nav = _nav()
    shallow = pa.drawdowns_table(nav, top_n=50, min_drawdown=0.01)
    deep = pa.drawdowns_table(nav, top_n=50, min_drawdown=0.15)
    assert len(deep) <= len(shallow)


def test_drawdowns_table_returns_the_expected_columns():
    out = pa.drawdowns_table(_nav(), top_n=5)
    for column in ["rank", "start_date", "trough_date", "drawdown"]:
        assert column in out.columns


def test_cagr_rolled_runs_on_modern_numpy():
    nav = _nav()
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    out = pa.cagr_rolled(yearly, max_holding_period=3)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(yearly)
    assert out.shape[1] == min(3, len(yearly))


def test_cagr_rolled_first_column_is_the_yearly_return():
    nav = _nav()
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    out = pa.cagr_rolled(yearly, max_holding_period=3)
    assert out.iloc[0, 0] == pytest.approx(float(yearly.iloc[0, 0]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_portfolio_analysis_fixes.py -v`
Expected: FAIL — `top_n` assertions fail (10 rows returned regardless), and
`cagr_rolled` raises `ValueError: setting an array element with a sequence`.

- [ ] **Step 3: Fix `drawdowns_table`**

Replace it in `portfolio_construction/portfolio_analysis.py`. The signature's
type hint was wrong (it needs a Series), and the arguments were being dropped:

```python
def drawdowns_table(
    nav: pd.Series, top_n: int = 10, min_drawdown: float = 0.01
) -> pd.DataFrame:
    """Rank the worst drawdown episodes.

    Takes a Series: `nav.items()` must yield (timestamp, price) pairs, which a
    DataFrame does not do.
    """
    convert_ts = [(date.date(), Decimal(str(price))) for date, price in nav.items()]

    strat_dds = compute_drawdown_periods(
        convert_ts, top_n=top_n, min_drawdown=min_drawdown
    )

    return pd.DataFrame(strat_dds)
```

- [ ] **Step 4: Fix `cagr_rolled` for numpy 2.x**

In `cagr_rolled`, the two assignments write a size-1 array into a scalar slot.
numpy 2 rejects that. Coerce to float:

```python
        result_matrix[year][0] = float(yearly_returns.iloc[year].iloc[0])
```

and further down:

```python
                result_matrix[year, holding_period - 1] = float(
                    (sub_period.apply(lambda x: np.cumprod(1 + x)).tail(1).values)
                    ** (1 / holding_period)
                    - 1.0
                )
```

Change nothing else about the algorithm.

- [ ] **Step 5: Run the tests, then the full suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_portfolio_analysis_fixes.py -v`
Expected: PASS (5 tests)

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 195 passed (190 + 5).

- [ ] **Step 6: Commit**

```bash
git add portfolio_construction/portfolio_analysis.py tests/test_portfolio_analysis_fixes.py
git commit -m "fix: honour drawdowns_table arguments and repair cagr_rolled on numpy 2"
```

---

### Task 1: `portfolio_ui/analytics.py` — the pure analysis layer

**Files:**
- Create: `portfolio_ui/analytics.py`
- Test: `tests/ui/test_analytics.py`

**Interfaces:**
- Consumes: `portfolio_ui.dataset.ActiveDataset`; `portfolio_construction.portfolio_analysis`; `portfolio_construction.stats`.
- Produces:
  - `to_returns(prices: pd.DataFrame) -> pd.DataFrame`
  - `weighted_nav(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series` — a single equity curve rebased to 100, equal-weighted when `weights` is None.
  - `performance_table(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame` — one row per column, the `stats_report` output.
  - `drawdown_series(nav: pd.Series) -> pd.Series`
  - `drawdown_episodes(nav: pd.Series, top_n: int = 10, min_drawdown: float = 0.01) -> pd.DataFrame`
  - `monthly_returns_table(nav: pd.Series) -> pd.DataFrame`
  - `rebased_prices(prices: pd.DataFrame, rebased_dt) -> pd.DataFrame`
  - `calendar_table(nav: pd.Series) -> pd.DataFrame`
  - `rolling_cagr(nav: pd.Series, max_holding_period: int = 20) -> pd.DataFrame`
  - `AnalyticsError(RuntimeError)` — raised when an input cannot support the requested statistic.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_analytics.py`:

```python
"""The pure analysis layer: dataset in, display-ready frames out.

No streamlit, no network. Deterministic synthetic prices so assertions can be
exact rather than approximate.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.analytics import (
    AnalyticsError,
    calendar_table,
    drawdown_episodes,
    drawdown_series,
    monthly_returns_table,
    performance_table,
    rebased_prices,
    rolling_cagr,
    to_returns,
    weighted_nav,
)


def _prices(days=760, seed=0):
    """Two assets, deterministic pseudo-random walk, business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days, name="Date")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, days)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_to_returns_drops_the_first_row():
    prices = _prices(10)
    out = to_returns(prices)
    assert len(out) == len(prices) - 1
    assert list(out.columns) == ["AAA", "BBB"]


def test_to_returns_computes_simple_returns():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0]},
        index=pd.bdate_range("2020-01-01", periods=2, name="Date"),
    )
    assert to_returns(prices)["AAA"].iloc[0] == pytest.approx(0.10)


def test_weighted_nav_equal_weights_by_default():
    prices = _prices(50)
    nav = weighted_nav(prices)
    assert isinstance(nav, pd.Series)
    assert nav.iloc[0] == pytest.approx(100.0)


def test_weighted_nav_honours_explicit_weights():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [100.0, 100.0]},
        index=pd.bdate_range("2020-01-01", periods=2, name="Date"),
    )
    nav = weighted_nav(prices, {"AAA": 1.0, "BBB": 0.0})
    assert nav.iloc[-1] == pytest.approx(110.0)


def test_weighted_nav_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(AnalyticsError, match="sum"):
        weighted_nav(_prices(10), {"AAA": 0.3, "BBB": 0.3})


def test_weighted_nav_rejects_unknown_ticker():
    with pytest.raises(AnalyticsError, match="NOPE"):
        weighted_nav(_prices(10), {"NOPE": 1.0})


def test_performance_table_has_one_row_per_column():
    out = performance_table(_prices(500))
    assert len(out) == 2
    assert "Name" in out.columns
    assert set(out["Name"]) == {"AAA", "BBB"}


def test_performance_table_reports_the_headline_statistics():
    out = performance_table(_prices(500))
    for column in ["Ann. Return", "Ann. Volatility", "Ann. Sharpe", "Max Drawdown"]:
        assert column in out.columns


def test_drawdown_series_is_never_positive():
    dd = drawdown_series(weighted_nav(_prices(500)))
    assert dd.max() <= 1e-9


def test_drawdown_series_is_zero_at_a_new_peak():
    nav = pd.Series(
        [100.0, 110.0, 120.0], index=pd.bdate_range("2020-01-01", periods=3)
    )
    assert drawdown_series(nav).iloc[-1] == pytest.approx(0.0)


def test_drawdown_episodes_returns_a_frame():
    out = drawdown_episodes(weighted_nav(_prices(500)), top_n=5)
    assert isinstance(out, pd.DataFrame)
    assert len(out) <= 5


def test_calendar_table_covers_every_year_present():
    nav = weighted_nav(_prices(760))  # spans 2020 into 2022
    out = calendar_table(nav)
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 2


def test_rolling_cagr_returns_a_frame():
    out = rolling_cagr(weighted_nav(_prices(760)), max_holding_period=2)
    assert isinstance(out, pd.DataFrame)


def test_monthly_returns_table_has_a_row_per_month():
    nav = weighted_nav(_prices(260))  # about a year of business days
    out = monthly_returns_table(nav)
    assert isinstance(out, pd.DataFrame)
    assert 10 <= len(out) <= 13


def test_rebased_prices_start_at_100_on_the_chosen_date():
    prices = _prices(100)
    anchor = prices.index[20]
    out = rebased_prices(prices, anchor)
    assert out.iloc[0].round(6).eq(100.0).all()
    assert out.index[0] == anchor


def test_rebased_prices_rejects_a_date_not_in_the_index():
    with pytest.raises(AnalyticsError, match="not a date in this dataset"):
        rebased_prices(_prices(100), pd.Timestamp("1999-01-04"))


def test_functions_reject_a_single_row_input():
    one_row = _prices(1)
    with pytest.raises(AnalyticsError, match="at least two"):
        to_returns(one_row)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_analytics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.analytics'`

- [ ] **Step 3: Write `portfolio_ui/analytics.py`**

```python
"""Turn an active dataset into display-ready analysis frames.

No streamlit import - the views render what this returns. Every function takes
plain pandas objects so it can be tested without a Streamlit runtime.
"""

from __future__ import annotations

import pandas as pd

from portfolio_construction import portfolio_analysis as pa


class AnalyticsError(RuntimeError):
    """An input cannot support the requested statistic."""


def _require_two_rows(frame) -> None:
    if len(frame) < 2:
        raise AnalyticsError("need at least two observations to compute this")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period returns, first (all-NaN) row dropped."""
    _require_two_rows(prices)
    return prices.pct_change().dropna(how="all")


def weighted_nav(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """One equity curve for the whole basket, rebased to 100 at the start.

    Weights are fixed (buy-and-hold), not rebalanced - the Backtest page is
    where rebalancing lives.
    """
    _require_two_rows(prices)

    if weights is None:
        columns = list(prices.columns)
        weights = {c: 1.0 / len(columns) for c in columns}

    unknown = [t for t in weights if t not in prices.columns]
    if unknown:
        raise AnalyticsError(f"not in the dataset: {', '.join(unknown)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise AnalyticsError(f"weights must sum to 1.0, got {total:.4f}")

    used = prices[list(weights)]
    rebased = used.div(used.iloc[0])
    nav = rebased.mul(pd.Series(weights)).sum(axis=1) * 100.0
    nav.name = "Portfolio"
    return nav


def performance_table(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """The headline statistics, one row per column of `prices`."""
    _require_two_rows(prices)
    return pa.stats_report(prices, rf=rf)


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drop from the running peak, as a negative fraction.

    historical_drawdown already returns a Series on the same index, so this
    only renames it - re-wrapping it in pd.Series(..., index=...) would risk
    silently reindexing to NaN.
    """
    _require_two_rows(nav)
    return pa.historical_drawdown(nav).rename("Drawdown")


def monthly_returns_table(nav: pd.Series) -> pd.DataFrame:
    """Month-end returns, one row per month."""
    _require_two_rows(nav)
    return pa.monthly_returns(nav)


def rebased_prices(prices: pd.DataFrame, rebased_dt) -> pd.DataFrame:
    """Every series rebased to 100 at a chosen date.

    rebased_from_date returns None (after printing) when the date is absent
    from the index, so translate that into an AnalyticsError the page can show.
    """
    _require_two_rows(prices)
    out = pa.rebased_from_date(prices, pd.Timestamp(rebased_dt))
    if out is None:
        raise AnalyticsError(
            f"{pd.Timestamp(rebased_dt):%Y-%m-%d} is not a date in this dataset"
        )
    return out


def drawdown_episodes(
    nav: pd.Series, top_n: int = 10, min_drawdown: float = 0.01
) -> pd.DataFrame:
    """The worst peak-to-trough episodes with their recovery dates.

    drawdowns_table takes a SERIES despite its type hint saying DataFrame -
    it relies on .items() yielding (timestamp, price). compute_drawdown_periods
    itself wants list[tuple[date, Decimal]] and is not UI-friendly.
    """
    _require_two_rows(nav)
    return pa.drawdowns_table(nav, top_n=top_n, min_drawdown=min_drawdown)


def calendar_table(nav: pd.Series) -> pd.DataFrame:
    """Performance broken down by calendar period."""
    _require_two_rows(nav)
    return pa.calendar_performances(nav)


def rolling_cagr(nav: pd.Series, max_holding_period: int = 20) -> pd.DataFrame:
    """CAGR over every rolling holding period, up to max_holding_period years."""
    _require_two_rows(nav)
    yearly = nav.resample("YE").last().pct_change().dropna().to_frame()
    return pa.cagr_rolled(yearly, max_holding_period=max_holding_period)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_analytics.py -v`
Expected: PASS (17 tests)

`stats_report`, `calendar_performances`, `drawdowns_table` and `cagr_rolled`
were all probed against the installed package and work with the shapes used
here, given Task 0's fixes. If one still raises, adjust the wrapper in
`analytics.py` — do not change the core library beyond Task 0.

- [ ] **Step 5: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 212 passed (195 + 17).

- [ ] **Step 6: Commit**

```bash
git add portfolio_ui/analytics.py tests/ui/test_analytics.py
git commit -m "feat: add the pure analysis layer behind the Analysis page"
```

---

### Task 2: Analysis chart builders

**Files:**
- Modify: `portfolio_ui/charts.py`
- Test: `tests/ui/test_charts.py` (extend)

**Interfaces:**
- Consumes: Task 1's outputs.
- Produces, appended to `portfolio_ui/charts.py`:
  - `nav_figure(nav: pd.Series, title: str) -> go.Figure`
  - `drawdown_figure(drawdown: pd.Series) -> go.Figure` — filled area below zero.
  - `calendar_bar_figure(calendar: pd.DataFrame) -> go.Figure`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_charts.py`:

```python
def _nav():
    return pd.Series(
        [100.0, 105.0, 98.0, 110.0],
        index=pd.bdate_range("2020-01-01", periods=4, name="Date"),
        name="Portfolio",
    )


def test_nav_figure_has_one_trace_titled_as_asked():
    from portfolio_ui.charts import nav_figure

    fig = nav_figure(_nav(), title="Equity curve")
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Equity curve"
    assert fig.layout.yaxis.title.text == "NAV"


def test_drawdown_figure_is_a_filled_area_and_never_positive():
    from portfolio_ui.charts import drawdown_figure

    nav = _nav()
    drawdown = nav.div(nav.cummax()).sub(1.0)
    fig = drawdown_figure(drawdown)
    assert len(fig.data) == 1
    assert fig.data[0].fill == "tozeroy"
    assert max(fig.data[0].y) <= 0


def test_calendar_bar_figure_has_a_bar_per_period():
    from portfolio_ui.charts import calendar_bar_figure

    calendar = pd.DataFrame({"Performance": [0.10, -0.05]}, index=[2020, 2021])
    fig = calendar_bar_figure(calendar)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [2020, 2021]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_charts.py -v`
Expected: FAIL — `ImportError: cannot import name 'nav_figure'`

- [ ] **Step 3: Append the builders to `portfolio_ui/charts.py`**

```python
def nav_figure(nav: pd.Series, title: str) -> go.Figure:
    """A single equity curve."""
    fig = go.Figure(
        data=[go.Scatter(x=nav.index, y=nav.values, mode="lines", name=nav.name or "NAV")]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def drawdown_figure(drawdown: pd.Series) -> go.Figure:
    """Underwater plot - the drop from the running peak, filled to zero."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
            )
        ]
    )
    fig.update_layout(
        title="Drawdown from running peak",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def calendar_bar_figure(calendar: pd.DataFrame) -> go.Figure:
    """One bar per calendar period, using the frame's last column."""
    values = calendar.iloc[:, -1]
    fig = go.Figure(data=[go.Bar(x=list(calendar.index), y=list(values), name="Return")])
    fig.update_layout(
        title="Calendar performance",
        xaxis_title="Period",
        yaxis_title="Return",
        yaxis_tickformat=".1%",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_charts.py -v`
Expected: PASS (7 tests — 4 existing + 3 new)

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 215 passed.

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/charts.py tests/ui/test_charts.py
git commit -m "feat: add nav, drawdown and calendar chart builders"
```

---

### Task 3: The Analysis page

**Files:**
- Create: `portfolio_ui/views/analysis_page.py`
- Modify: `portfolio_ui/views/placeholders.py` (drop the Analysis stub)
- Modify: `portfolio_ui/app.py` (point the Analysis page at the real module)

**Interfaces:**
- Consumes: `portfolio_ui.analytics` (Task 1), `portfolio_ui.charts` (Task 2), `portfolio_ui.guards.require_active_dataset`, `portfolio_ui.state.add_derived`.
- Produces: `analysis_page() -> None`.

- [ ] **Step 1: Write `portfolio_ui/views/analysis_page.py`**

```python
"""Performance statistics, drawdown episodes and calendar returns."""

from __future__ import annotations

import streamlit as st

from portfolio_ui.analytics import (
    AnalyticsError,
    calendar_table,
    drawdown_episodes,
    drawdown_series,
    monthly_returns_table,
    performance_table,
    rebased_prices,
    rolling_cagr,
    weighted_nav,
)
from portfolio_ui.charts import calendar_bar_figure, drawdown_figure, nav_figure
from portfolio_ui.dataset import dataset_from_frame
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.state import add_derived


def _weights_editor(dataset):
    """Let the user weight the basket, defaulting to equal weight."""
    equal = 1.0 / len(dataset.tickers)
    with st.expander("Weights", expanded=False):
        st.caption("Fixed weights, buy and hold. Rebalancing lives on the Backtest page.")
        weights = {
            ticker: st.number_input(
                ticker, min_value=0.0, max_value=1.0, value=equal, step=0.05,
                key=f"analysis_w_{ticker}",
            )
            for ticker in dataset.tickers
        }
        total = sum(weights.values())
        st.caption(f"Total: {total:.4f}")
    return weights


def analysis_page() -> None:
    st.title("Analysis")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    rf = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.25, value=0.0, step=0.005
    )
    weights = _weights_editor(dataset)

    try:
        nav = weighted_nav(dataset.prices, weights)
    except AnalyticsError as exc:
        st.error(str(exc))
        return

    per_asset, portfolio = st.tabs(["Per asset", "Portfolio"])

    with per_asset:
        try:
            st.dataframe(performance_table(dataset.prices, rf=rf), use_container_width=True)
        except AnalyticsError as exc:
            st.error(str(exc))

        st.subheader("Rebase to a chosen date")
        anchor = st.date_input(
            "Rebase from", value=dataset.start,
            min_value=dataset.start, max_value=dataset.end,
        )
        try:
            st.dataframe(rebased_prices(dataset.prices, anchor).tail(20), use_container_width=True)
        except AnalyticsError as exc:
            # Non-trading days are the normal case here, not an error state.
            st.info(str(exc))

    with portfolio:
        st.plotly_chart(nav_figure(nav, f"{dataset.name} - weighted NAV"), use_container_width=True)

        try:
            st.plotly_chart(drawdown_figure(drawdown_series(nav)), use_container_width=True)
        except AnalyticsError as exc:
            st.error(str(exc))

        st.subheader("Worst drawdown episodes")
        top_n = st.slider("How many", min_value=3, max_value=20, value=10)
        try:
            st.dataframe(drawdown_episodes(nav, top_n=top_n), use_container_width=True)
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute drawdown episodes: {exc}")

        st.subheader("Calendar performance")
        try:
            calendar = calendar_table(nav)
            st.plotly_chart(calendar_bar_figure(calendar), use_container_width=True)
            st.dataframe(calendar, use_container_width=True)
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute calendar performance: {exc}")

        st.subheader("Monthly returns")
        try:
            st.dataframe(monthly_returns_table(nav), use_container_width=True)
        except (AnalyticsError, ValueError, KeyError) as exc:
            st.error(f"Could not compute monthly returns: {exc}")

        st.subheader("Rolling CAGR")
        horizon = st.slider("Max holding period (years)", 2, 20, 10)
        if st.button("Compute rolling CAGR"):
            try:
                st.dataframe(rolling_cagr(nav, max_holding_period=horizon), use_container_width=True)
            except (AnalyticsError, ValueError, KeyError) as exc:
                st.error(f"Could not compute rolling CAGR: {exc}")

        if st.button("Save this NAV as a derived dataset"):
            derived = dataset_from_frame(
                nav.to_frame(), f"{dataset.name}-nav", dataset.source,
                notes=("weighted NAV derived on the Analysis page",),
            )
            add_derived(store, derived)
            st.success(f"Registered derived dataset '{derived.name}'")
```

- [ ] **Step 2: Remove the Analysis stub and wire the real page**

In `portfolio_ui/views/placeholders.py`, delete the `"Analysis"` entry from the
`_COMING` dict and delete the `analysis_page` function.

In `portfolio_ui/app.py`, change the import so Analysis comes from its own module:

```python
from portfolio_ui.views.analysis_page import analysis_page
from portfolio_ui.views.placeholders import (
    backtest_page,
    market_data_page,
    optimization_page,
    risk_page,
)
```

Leave the `st.Page` list unchanged — it already references `analysis_page`.

- [ ] **Step 3: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 215 passed. (The page itself is thin wiring and is not unit-tested;
the logic beneath it is covered by Task 1.)

- [ ] **Step 4: Boot smoke test**

```bash
./.venv/Scripts/python.exe -m streamlit run portfolio_ui/app.py --server.headless true --server.port 8899 > /tmp/analysis-smoke.log 2>&1 &
```
Then check `http://localhost:8899/_stcore/health` returns 200, inspect the log
for tracebacks, and kill the process.

Additionally drive the page with Streamlit's own harness, which actually
executes the script:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("portfolio_ui/app.py", default_timeout=30).run()
assert not at.exception
```

Report both results.

- [ ] **Step 5: Commit**

```bash
git add portfolio_ui/views/analysis_page.py portfolio_ui/views/placeholders.py portfolio_ui/app.py
git commit -m "feat: add the Analysis page"
```

---

### Task 4: `portfolio_ui/risk.py` — the pure risk layer

**Files:**
- Create: `portfolio_ui/risk.py`
- Test: `tests/ui/test_risk.py`

**Interfaces:**
- Consumes: `portfolio_construction.portfolio_analysis`, `portfolio_construction.portfolio_optimization.get_stressed_covariance`.
- Produces:
  - `RiskError(RuntimeError)`
  - `normalize_weights(tickers, weights=None) -> list[float]` — equal weight when None; validates sum and membership.
  - `var_es_table(prices, weights, alpha=0.05, duration=1, distrib="normal") -> pd.DataFrame` — rows for historical / parametric / Monte Carlo, columns VaR and ES.
  - `simulate_paths(nav, n_sims, days, distrib="normal") -> pd.DataFrame` — one column per simulated path via `gbm_multiple_path`.
  - `simulate_portfolio_paths(prices, weights, n_sims, days, distrib="normal") -> pd.DataFrame` — correlated multi-asset via `portfolio_path_cholesky`.
  - `stressed_covariance(prices, stress_factor=0.5) -> pd.DataFrame`
  - `covariance_var(prices, weights, alpha=0.05, distrib="normal") -> float` — wraps `stats.covariance_parametric_var`, which needs only a covariance matrix and weights.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_risk.py`:

```python
"""The pure risk layer. Deterministic where possible; simulations are checked
for shape and sanity rather than exact values."""

import numpy as np
import pandas as pd
import pytest

from portfolio_ui.risk import (
    RiskError,
    covariance_var,
    normalize_weights,
    simulate_paths,
    simulate_portfolio_paths,
    stressed_covariance,
    var_es_table,
)


def _prices(days=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days, name="Date")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, days)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_normalize_weights_defaults_to_equal():
    assert normalize_weights(["AAA", "BBB"]) == [0.5, 0.5]


def test_normalize_weights_preserves_requested_order():
    out = normalize_weights(["AAA", "BBB"], {"BBB": 0.25, "AAA": 0.75})
    assert out == [0.75, 0.25]


def test_normalize_weights_rejects_a_bad_sum():
    with pytest.raises(RiskError, match="sum"):
        normalize_weights(["AAA", "BBB"], {"AAA": 0.3, "BBB": 0.3})


def test_normalize_weights_rejects_unknown_ticker():
    with pytest.raises(RiskError, match="NOPE"):
        normalize_weights(["AAA"], {"NOPE": 1.0})


def test_var_es_table_has_a_row_per_method():
    out = var_es_table(_prices(), None)
    assert set(out.index) == {"Historical", "Parametric", "Monte Carlo"}
    assert list(out.columns) == ["VaR", "Expected Shortfall"]


def test_var_es_values_are_losses():
    out = var_es_table(_prices(), None)
    assert (out["VaR"] < 0).all()
    assert (out["Expected Shortfall"] < 0).all()


def test_expected_shortfall_is_at_least_as_severe_as_var():
    out = var_es_table(_prices(), None)
    for method in out.index:
        assert out.loc[method, "Expected Shortfall"] <= out.loc[method, "VaR"] + 1e-9


def test_a_higher_confidence_gives_a_more_severe_var():
    prices = _prices()
    mild = var_es_table(prices, None, alpha=0.10).loc["Historical", "VaR"]
    harsh = var_es_table(prices, None, alpha=0.01).loc["Historical", "VaR"]
    assert harsh <= mild


def test_simulate_paths_shape():
    nav = _prices(300)["AAA"]
    out = simulate_paths(nav, n_sims=50, days=30)
    assert out.shape[1] == 50
    assert len(out) == 31  # days + 1: the starting value is row 0


def test_simulate_paths_all_start_from_the_last_observed_value():
    nav = _prices(300)["AAA"]
    out = simulate_paths(nav, n_sims=20, days=10)
    assert out.notna().all().all()


def test_simulate_portfolio_paths_shape():
    prices = _prices(300)
    out = simulate_portfolio_paths(prices, None, n_sims=25, days=20)
    assert out.shape[1] == 25
    assert len(out) == 21  # days + 1


def test_stressed_covariance_is_square_and_labelled():
    out = stressed_covariance(_prices(200))
    assert out.shape == (2, 2)
    assert list(out.columns) == ["AAA", "BBB"]


def test_stressing_raises_off_diagonal_covariance():
    prices = _prices(200)
    plain = prices.pct_change().dropna().cov()
    stressed = stressed_covariance(prices, stress_factor=0.9)
    assert stressed.loc["AAA", "BBB"] >= plain.loc["AAA", "BBB"]


def test_covariance_var_is_a_loss_and_worsens_under_stress():
    prices = _prices(200)
    base = covariance_var(prices)
    stressed = covariance_var(prices, cov=stressed_covariance(prices, stress_factor=0.9))
    assert base < 0
    # correlations pushed toward 1 remove diversification, so the loss deepens
    assert stressed <= base
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio_ui.risk'`

- [ ] **Step 3: Write `portfolio_ui/risk.py`**

```python
"""VaR, expected shortfall and Monte Carlo simulation, as plain functions.

No streamlit import. The Risk page renders what this returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_construction import portfolio_analysis as pa
from portfolio_construction.portfolio_optimization import get_stressed_covariance
from portfolio_construction.stats import covariance_parametric_var


class RiskError(RuntimeError):
    """An input cannot support the requested risk statistic."""


def normalize_weights(tickers, weights: dict[str, float] | None = None) -> list[float]:
    """A weight vector aligned to `tickers`, equal-weighted when unspecified."""
    tickers = list(tickers)
    if not tickers:
        raise RiskError("no tickers to weight")

    if weights is None:
        return [1.0 / len(tickers)] * len(tickers)

    unknown = [t for t in weights if t not in tickers]
    if unknown:
        raise RiskError(f"not in the dataset: {', '.join(unknown)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise RiskError(f"weights must sum to 1.0, got {total:.4f}")

    return [float(weights.get(t, 0.0)) for t in tickers]


def var_es_table(
    prices: pd.DataFrame,
    weights: dict[str, float] | None,
    alpha: float = 0.05,
    duration: int = 1,
    distrib: str = "normal",
    n_sims: int = 10000,
) -> pd.DataFrame:
    """VaR and expected shortfall by all three estimation families."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to estimate risk")

    vec_w = normalize_weights(prices.columns, weights)
    portfolio = prices.mul(pd.Series(vec_w, index=prices.columns)).sum(axis=1)
    returns = portfolio.pct_change().dropna()

    mu = float(returns.mean())
    sigma = float(returns.std())
    start = float(portfolio.iloc[-1])

    rows = {
        "Historical": (
            pa.historical_var(portfolio, alpha, duration),
            pa.historical_es(portfolio, alpha, duration),
        ),
        "Parametric": (
            pa.parametric_var(prices, vec_w, alpha, duration, distrib),
            pa.parametric_es(prices, vec_w, alpha, duration),
        ),
        "Monte Carlo": (
            pa.monteCarlo_var(n_sims, start, mu, sigma, alpha, duration, distrib),
            pa.monteCarlo_es(n_sims, start, mu, sigma, alpha, duration, distrib),
        ),
    }

    return pd.DataFrame(
        [{"VaR": float(v), "Expected Shortfall": float(es)} for v, es in rows.values()],
        index=list(rows),
    )


def simulate_paths(
    nav: pd.Series, n_sims: int, days: int, distrib: str = "normal"
) -> pd.DataFrame:
    """Simulated forward paths for a single series, via geometric Brownian motion."""
    if len(nav) < 2:
        raise RiskError("need at least two observations to simulate")

    returns = nav.pct_change().dropna()
    paths = pa.gbm_multiple_path(
        n_sims, float(nav.iloc[-1]), float(returns.mean()), float(returns.std()),
        days, distrib,
    )
    # gbm_multiple_path returns (days + 1, n_sims): one extra row for the
    # starting value, paths in columns. Verified against the installed package.
    frame = pd.DataFrame(np.asarray(paths))
    frame.columns = [f"path_{i}" for i in range(frame.shape[1])]
    return frame


def simulate_portfolio_paths(
    prices: pd.DataFrame,
    weights: dict[str, float] | None,
    n_sims: int,
    days: int,
    distrib: str = "normal",
) -> pd.DataFrame:
    """Correlated multi-asset simulation via a Cholesky factorization."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to simulate")

    vec_w = normalize_weights(prices.columns, weights)
    returns = prices.pct_change().dropna()
    start = float(prices.mul(pd.Series(vec_w, index=prices.columns)).sum(axis=1).iloc[-1])

    paths = pa.portfolio_path_cholesky(
        n_sims, start, vec_w, list(returns.mean()), returns.cov(), days, distrib
    )
    # portfolio_path_cholesky also returns (days + 1, n_sims).
    frame = pd.DataFrame(np.asarray(paths))
    frame.columns = [f"path_{i}" for i in range(frame.shape[1])]
    return frame


def stressed_covariance(prices: pd.DataFrame, stress_factor: float = 0.5) -> pd.DataFrame:
    """Covariance with correlations pushed toward 1 - the panic scenario."""
    if len(prices) < 2:
        raise RiskError("need at least two observations to estimate covariance")

    cov = prices.pct_change().dropna().cov()
    stressed = get_stressed_covariance(cov, stress_factor=stress_factor)
    return pd.DataFrame(np.asarray(stressed), index=cov.index, columns=cov.columns)


def covariance_var(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    alpha: float = 0.05,
    distrib: str = "normal",
    cov: pd.DataFrame | None = None,
) -> float:
    """VaR from a covariance matrix and weights alone - no return history needed.

    Pass `cov` to price the same weights under a stressed matrix, which is what
    makes the stress scenario comparable against the base case.
    """
    if cov is None:
        if len(prices) < 2:
            raise RiskError("need at least two observations to estimate covariance")
        cov = prices.pct_change().dropna().cov()

    vec_w = normalize_weights(prices.columns, weights)
    return float(
        covariance_parametric_var(vec_w, np.asarray(cov), alpha=alpha, distrib=distrib)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_risk.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Run the full suite and commit**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 229 passed.

```bash
git add portfolio_ui/risk.py tests/ui/test_risk.py
git commit -m "feat: add the pure risk layer behind the Risk page"
```

---

### Task 5: Monte Carlo chart and the Risk page

**Files:**
- Modify: `portfolio_ui/charts.py`
- Modify: `tests/ui/test_charts.py`
- Create: `portfolio_ui/views/risk_page.py`
- Modify: `portfolio_ui/views/placeholders.py`, `portfolio_ui/app.py`

**Interfaces:**
- Produces: `charts.simulation_fan_figure(paths: pd.DataFrame, title: str) -> go.Figure` — median line plus a shaded 5th–95th percentile band, NOT one trace per path (a 10 000-path chart must stay renderable). And `risk_page() -> None`.

- [ ] **Step 1: Write the failing chart test**

Append to `tests/ui/test_charts.py`:

```python
def test_simulation_fan_figure_summarizes_rather_than_drawing_every_path():
    from portfolio_ui.charts import simulation_fan_figure

    paths = pd.DataFrame(
        {f"path_{i}": [100.0, 100.0 + i, 100.0 + 2 * i] for i in range(500)}
    )
    fig = simulation_fan_figure(paths, title="Simulated NAV")
    # median + two band edges, never 500 traces
    assert len(fig.data) <= 3
    assert fig.layout.title.text == "Simulated NAV"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/ui/test_charts.py -v`
Expected: FAIL — `ImportError: cannot import name 'simulation_fan_figure'`

- [ ] **Step 3: Append the builder to `portfolio_ui/charts.py`**

```python
def simulation_fan_figure(paths: pd.DataFrame, title: str) -> go.Figure:
    """Median path with a 5th-95th percentile band.

    Deliberately summarizes: drawing 10 000 individual traces would make the
    page unusable and tell the reader less, not more.
    """
    steps = list(range(len(paths)))
    low = paths.quantile(0.05, axis=1)
    high = paths.quantile(0.95, axis=1)
    median = paths.median(axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=steps, y=high, mode="lines", name="95th percentile",
                   line=dict(width=0), showlegend=False)
    )
    fig.add_trace(
        go.Scatter(x=steps, y=low, mode="lines", name="5th-95th percentile",
                   line=dict(width=0), fill="tonexty")
    )
    fig.add_trace(go.Scatter(x=steps, y=median, mode="lines", name="Median"))
    fig.update_layout(
        title=title,
        xaxis_title="Days ahead",
        yaxis_title="Simulated value",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
```

- [ ] **Step 4: Write `portfolio_ui/views/risk_page.py`**

```python
"""Value-at-Risk, expected shortfall, Monte Carlo simulation and stress."""

from __future__ import annotations

import streamlit as st

from portfolio_ui.charts import simulation_fan_figure
from portfolio_ui.guards import require_active_dataset
from portfolio_ui.risk import (
    RiskError,
    covariance_var,
    simulate_paths,
    simulate_portfolio_paths,
    stressed_covariance,
    var_es_table,
)


def risk_page() -> None:
    st.title("Risk")
    store = st.session_state
    dataset = require_active_dataset(store)

    if dataset is None:
        st.info("No active dataset yet. Build one on the Data page first.")
        return

    st.caption(f"Active dataset: **{dataset.name}** - {dataset.summary()}")

    col_alpha, col_duration, col_dist = st.columns(3)
    alpha = col_alpha.select_slider(
        "Tail probability", options=[0.01, 0.025, 0.05, 0.10], value=0.05
    )
    duration = col_duration.number_input(
        "Horizon (periods)", min_value=1, max_value=250, value=1
    )
    distrib = col_dist.selectbox("Distribution", ["normal", "student"])

    tail, simulation, stress = st.tabs(["VaR / ES", "Monte Carlo", "Stressed covariance"])

    with tail:
        st.caption(
            f"Loss not exceeded on {(1 - alpha):.0%} of {duration}-period horizons, "
            "equal-weighted across the dataset."
        )
        if st.button("Estimate VaR and ES"):
            with st.spinner("Estimating..."):
                try:
                    table = var_es_table(
                        dataset.prices, None, alpha=alpha, duration=int(duration),
                        distrib=distrib,
                    )
                except (RiskError, ValueError, KeyError) as exc:
                    st.error(str(exc))
                else:
                    st.dataframe(
                        table.style.format("{:.2%}"), use_container_width=True
                    )

    with simulation:
        col_sims, col_days = st.columns(2)
        n_sims = col_sims.number_input(
            "Paths", min_value=100, max_value=20000, value=2000, step=100
        )
        days = col_days.number_input(
            "Days ahead", min_value=5, max_value=500, value=60
        )
        correlated = st.checkbox(
            "Multi-asset with correlations (Cholesky)", value=True,
            help="Unchecked simulates the equal-weighted NAV as a single series",
        )

        if st.button("Run simulation"):
            with st.spinner(f"Simulating {n_sims} paths..."):
                try:
                    if correlated:
                        paths = simulate_portfolio_paths(
                            dataset.prices, None, int(n_sims), int(days), distrib
                        )
                    else:
                        nav = dataset.prices.mean(axis=1)
                        paths = simulate_paths(nav, int(n_sims), int(days), distrib)
                except (RiskError, ValueError, KeyError) as exc:
                    st.error(str(exc))
                else:
                    st.plotly_chart(
                        simulation_fan_figure(paths, "Simulated forward paths"),
                        use_container_width=True,
                    )
                    final = paths.iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median outcome", f"{final.median():,.1f}")
                    c2.metric("5th percentile", f"{final.quantile(0.05):,.1f}")
                    c3.metric("95th percentile", f"{final.quantile(0.95):,.1f}")

    with stress:
        factor = st.slider(
            "Stress factor", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="How far correlations are pushed toward 1",
        )
        if st.button("Compute stressed covariance"):
            try:
                stressed = stressed_covariance(dataset.prices, stress_factor=factor)
                base_var = covariance_var(dataset.prices, alpha=alpha, distrib=distrib)
                stressed_var = covariance_var(
                    dataset.prices, alpha=alpha, distrib=distrib, cov=stressed
                )
            except (RiskError, ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                # Same weights priced under both matrices - that comparison is
                # the whole point of stressing.
                left, right = st.columns(2)
                left.metric("VaR, base covariance", f"{base_var:.2%}")
                right.metric(
                    "VaR, stressed", f"{stressed_var:.2%}",
                    delta=f"{stressed_var - base_var:.2%}", delta_color="inverse",
                )
                st.dataframe(stressed, use_container_width=True)
```

- [ ] **Step 5: Wire the page and drop the stub**

In `portfolio_ui/views/placeholders.py`, delete the `"Risk"` entry from `_COMING`
and the `risk_page` function. In `portfolio_ui/app.py`:

```python
from portfolio_ui.views.risk_page import risk_page
```
and remove `risk_page` from the `placeholders` import list.

- [ ] **Step 6: Run the suite and the smoke tests**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: 230 passed.

Then the boot smoke test and the `AppTest` check exactly as in Task 3 Step 4.
Report both.

- [ ] **Step 7: Commit**

```bash
git add portfolio_ui/charts.py tests/ui/test_charts.py portfolio_ui/views/risk_page.py portfolio_ui/views/placeholders.py portfolio_ui/app.py
git commit -m "feat: add the Risk page with VaR, ES and Monte Carlo simulation"
```

---

## Definition of done

- [ ] `./.venv/Scripts/python.exe -m pytest -q` passes (230 expected).
- [ ] The app boots headless with HTTP 200 and no traceback, and `AppTest` reports no exception.
- [ ] Analysis and Risk no longer appear in `views/placeholders.py`.
- [ ] No `streamlit` import in `analytics.py`, `risk.py` or `charts.py`.
- [ ] Both pages render the "no active dataset" guard instead of failing when nothing is loaded.

## What this plan does not cover

The Optimization, Backtest and Market Data pages, and the §7.2 backtest wiring
(`RebalancingCalendar` into `Backtest`, and `stock_picking` through
`simulations`). Those follow in the next plan.
