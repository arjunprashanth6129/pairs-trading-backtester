# Pairs-trading backtester

A walk-forward evaluation of cointegration-based pairs trading on eleven NSE bank
stocks. Pairs are selected by the Engle-Granger test on a 12-month formation
period and traded over the following 6 months on a fixed z-score rule, with
costs charged on both legs at entry and exit.

**Live report:** https://arjun-quant-projects.vercel.app/pairs-trading
&nbsp;·&nbsp; **Full write-up:** [the report](docs/REPORT.md)

---

## Contents

- [Summary of results](#summary-of-results)
- [Quick start](#quick-start)
- [What the pipeline does](#what-the-pipeline-does)
- [Architecture](#architecture)
- [How lookahead is prevented](#how-lookahead-is-prevented)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Tests](#tests)
- [Extending it](#extending-it)

## Summary of results

19 rolling windows, 2,347 out-of-sample trading days (2017-01-02 to 2026-06-30),
242 round trips at 20 bps per leg.

| | Before costs | After costs | Hold the 11 banks |
|---|---:|---:|---:|
| Annualised return | +1.27% | -1.29% | +13.81% |
| Growth of 1 | 1.12x | 0.89x | 3.34x |
| Max drawdown | -13.59% | -20.17% | -57.95% |
| Beta to the Nifty | -0.026 | -0.026 | |

| How the trade ended | Trades | Gross P&L |
|---|---:|---:|
| Reverted | 71 | +₹3,20,605 |
| Stopped out | 130 | -₹3,05,070 |
| Timed out or window ended | 41 | +₹30,833 |
| **All trades** | **242** | **+₹46,368**, against ₹97,486 of costs |

Trades that converged were profitable, but stop-outs outnumbered them by 1.8 to
1 and offset almost all of their gains. The mean daily return before costs is not
significantly different from zero (Newey-West t = 0.67), and the formation
p-value is not associated with the stop-out rate (χ²(2) = 3.40, p = 0.18). The
break-even cost is 9.8 bps per leg. The strategy is market-neutral, with an R² of
0.003 against the Nifty 50.

![Gross P&L by exit reason, then costs](https://arjun-quant-projects.vercel.app/figures/pairs/pnl-by-exit.png)

The [report](docs/REPORT.md) sets out the method, the full results, the
correction of an entry-rule error that had overstated the after-cost loss by 8.5
percentage points a year, and the limitations.

## Quick start

```bash
make setup                 # venv + pinned dependencies
make run                   # data -> selection -> backtest -> figures -> dashboard
make test                  # 31 tests
```

Or without make:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pairs_backtester.run
PYTHONPATH=src .venv/bin/python -m pytest -q
```

One run regenerates everything in `results/` and `docs/figures/`, plus a
manifest recording the config, git hash and interpreter that produced them.
Prices are cached to `data/cache/` on the first run, so later runs are offline
and reproduce the same numbers.

## What the pipeline does

```
config/config.yaml
        |
        v
  [1] data.fetch_prices        yfinance, auto_adjust=True, cached to CSV
        |                      14 NSE bank tickers + ^NSEI as a benchmark
        v
  [2] data.align               drop series under 95% coverage, then keep only
        |                      dates where every remaining ticker traded
        v
  [3] data.make_windows        12m formation -> 6m trading, stepping 6m
        |                      19 non-overlapping trading windows
        v
    for each window:
        |
        +-- [4] pairs.select_pairs        FORMATION ONLY
        |         Engle-Granger on all 55 candidate pairs
        |         keep <= 5 with p <= 0.05, lowest first
        |         fit hedge ratio, spread mean, spread std, half-life
        |
        +-- [5] backtest.run_pair         TRADING ONLY
        |         z = (spread - formation_mean) / formation_std
        |         signals.generate -> entries, exits, stops, timeouts
        |         P&L with costs on both legs, entry and exit
        |
        +-- [6] backtest.combine          equal-weight the window's pairs
        |
        v
  [7] metrics, figures, manifest, dashboard.json
```

Selection and estimation use only the formation period, and the backtest uses
only the trading period. The two periods do not overlap.

## Architecture

```
src/pairs_backtester/
  config.py       loads config/config.yaml, resolves output paths
  data.py         fetch, cache, align, and cut the timeline into windows
  pairs.py        hedge ratio, spread, half-life, z-score, cointegration,
                  selection. The only module that estimates parameters
  signals.py      z-score path -> list of trades. No prices, no money
  backtest.py     P&L accounting, costs, position sizing, combining pairs
  metrics.py      performance, trade statistics, market neutrality, cost sweep
  plots.py        the five report figures
  manifest.py     config + git hash + timestamp for every run
  export_site.py  compact JSON for the web page
  run.py          the entrypoint that wires the above together
tests/            31 tests: hedge ratio, cointegration, signals, lookahead
config/           every setting
data/cache/       price cache, gitignored and regenerable
results/          committed artifacts from the last run
docs/REPORT.md    the full write-up
docs/figures/     committed figures from the last run
```

`signals.py` maps a z-score series to trades, so the entry and exit rules can be
tested on hand-written paths without prices, costs or capital. `pairs.py` is the
only module that estimates parameters, so the absence of look-ahead can be
verified in one place.

### Position sizing and costs

Each open position has a fixed gross notional, `capital_per_pair`: one unit of
the dependent leg plus `|beta|` units of the other, scaled to that total. Both
legs must be financed and both incur costs, so positions are sized on gross
rather than net notional.

Costs are `bps_per_leg` on traded notional, charged on both legs at entry and
again at exit. The backtest also records the cost line at 1 bp, and the cost
sweep scales it, so the sweep at the configured rate matches the backtest
exactly.

## How lookahead is prevented

Look-ahead does not cause a backtest to fail; it inflates its results. Three
mechanisms prevent it, and each has a test that fails if the mechanism is
removed.

**Parameters are estimated in one function.** `pairs.fit_pair` only receives a
formation slice. The hedge ratio, spread mean and spread standard deviation are
fixed before the trading window starts.
`test_fitted_parameters_are_blind_to_the_trading_window` adds noise to every
trading-window price and asserts the fitted parameters do not change at all.

**The z-score cannot re-centre itself.** `pairs.zscore(sp, mean, std)` takes the
standardisation constants as arguments instead of computing them from its input.
The test checks for the signature of a self-standardised series: one computed
against its own window has mean zero by construction, and one computed against
fixed formation parameters does not.

**Execution lags the signal by one bar.** The decision uses the close of bar `i`
and the fill is booked at the close of bar `i+1`. Same-bar fills would bias this
strategy in particular, because entries occur on large price moves.

Each guard was verified by reintroducing the error and running the suite:

| Deliberate bug | Tests that fail |
|---|---|
| `zscore` recomputes its own mean and standard deviation | 3 |
| execution lag removed, same-bar fill | 1 |
| flat book allowed to enter while \|z\| is past the stop | 1 |

## Configuration

Everything lives in `config/config.yaml`. Nothing is hardcoded in the modules.

| Section | Key settings |
|---|---|
| `universe` | tickers, benchmark, date range |
| `windows` | `formation_months`, `trading_months`, `step_months` |
| `selection` | `pvalue_threshold`, `max_pairs`, `min_observations`, `min_spread_std` |
| `signals` | `entry_z`, `exit_z`, `stop_z`, `max_holding_days` |
| `costs` | `bps_per_leg` |
| `backtest` | `capital_per_pair`, `trading_days_per_year`, `risk_free_annual` |
| `run` | `seed`, output directories |

The signal thresholds are conventional values chosen in advance. Selecting them
by performance on this data would introduce look-ahead at the parameter level;
doing so properly would require a further holdout period.

## Outputs

Written to `results/` on every run:

| File | Contents |
|---|---|
| `selected_pairs.csv` | every pair chosen, with its window, p-value, hedge ratio, spread mean and standard deviation, and half-life |
| `trades.csv` | every round trip: entry and exit dates, side, entry and exit z, holding days, gross P&L, cost, net P&L, and exit reason |
| `daily_returns.csv` | daily portfolio return before and after costs, and the cost line at 1 bp |
| `cost_sensitivity.csv` | the same trades re-priced across a grid of cost levels |
| `summary.json` | performance, trade statistics and market-neutrality figures |
| `manifest.json` | config, git hash, interpreter and platform for the run |
| `dashboard.json` | compact payload for the web page |

And to `docs/figures/`:

| Figure | Shows |
|---|---|
| `pair-example.png` | one pair's z-score through one trading window, with every trade shaded |
| `pnl-by-exit.png` | gross P&L by exit reason, then costs, as a waterfall |
| `equity.png` | growth of 1 before and after costs, and the drawdown |
| `annual.png` | calendar-year return before and after costs |
| `cost-sensitivity.png` | annualised return across cost levels, with the break-even |

## Tests

```bash
make test
```

31 tests in four files:

- **`test_hedge_ratio.py`** recovers a known beta and alpha, checks that the
  spread of a perfect relationship is flat, confirms OLS is not symmetric (which
  is why pairs are oriented by a fixed rule), and checks the half-life estimator
  against simulated Ornstein-Uhlenbeck processes with known reversion speed.
- **`test_cointegration.py`** checks the selection test. The main case builds
  two series from identical shocks plus a small additional drift: their daily
  changes have a correlation above 0.99, yet Engle-Granger rejects cointegration
  because the price levels diverge without bound.
- **`test_signals.py`** covers entry, exit, stop, timeout, forced close at the
  window end, one position at a time, no entry while the spread is past the stop,
  NaN handling, and the one-bar execution lag.
- **`test_lookahead.py`** covers the lookahead guards above.

## Extending it

In order of priority:

- **Limit how many selected pairs share one bank.** The worst window lost
  ₹43,737 before costs, mostly on three positions that were each short Kotak
  Mahindra Bank.
- **Re-test cointegration during the trading window**, using only past data, and
  stop trading a pair once it fails.
- **A Kalman-filtered hedge ratio** that adapts during the trading window, with
  its own look-ahead test, since an adaptively estimated parameter is the most
  likely route for future data to enter.
- **Johansen alongside Engle-Granger**, which handles more than two series and
  does not depend on which leg is regressed on which.
- **Model the short leg through single-stock futures**, including basis and roll
  costs.

This is a student project and not investment advice.
