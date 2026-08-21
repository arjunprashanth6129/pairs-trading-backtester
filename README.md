# Pairs-trading backtester

A market-neutral statistical arbitrage system on Indian bank stocks.
Cointegration selects the pairs, a formation window fixes every parameter, and a
walk-forward backtest charges all four legs of every round trip.

**Live dashboard:** https://arjun-quant-projects.vercel.app/pairs-trading

This is the applied project in my quant portfolio, alongside two research studies
[on momentum and market efficiency](https://github.com/arjunprashanth6129/price-prediction-linear-regression)
and [on CVaR portfolio construction](https://github.com/arjunprashanth6129/cvar-portfolio-optimizer).
Those ask whether an effect exists. This one asks whether a working system built
on a real effect survives the cost of running it.

---

## Contents

- [Quick start](#quick-start)
- [What the pipeline does](#what-the-pipeline-does)
- [Architecture](#architecture)
- [How lookahead is prevented](#how-lookahead-is-prevented)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Tests](#tests)
- [Extending it](#extending-it)

## Quick start

```bash
make setup                 # venv + pinned dependencies
make run                   # data -> selection -> backtest -> report -> artifacts
make test                  # 29 tests
```

Or without make:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pairs_backtester.run
PYTHONPATH=src .venv/bin/pytest -q
```

One run regenerates everything in `results/` and `docs/figures/`, plus a
manifest recording the config, git hash and interpreter that produced them.
Prices are cached to `data/cache/` on first run, so subsequent runs are offline
and reproduce byte-identical numbers.

## What the pipeline does

```
config/config.yaml
        |
        v
  [1] data.fetch_prices        yfinance, auto_adjust=True, cached to CSV
        |                      14 NSE bank tickers + ^NSEI as a benchmark
        v
  [2] data.align               drop series under 95% coverage, then keep only
        |                      dates where every survivor traded
        v
  [3] data.make_windows        12m formation -> 6m trading, stepping 6m
        |                      19 non-overlapping trading windows
        v
    for each window:
        |
        +-- [4] pairs.select_pairs        FORMATION ONLY
        |         Engle-Granger on all 55 candidate pairs
        |         keep <= 5 with p <= 0.05, best first
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
  [7] metrics + plots + manifest
```

The important structural point is the `for each window` block. Selection and
fitting see only the formation slice; the backtest sees only the trading slice;
the two never overlap by a single bar.

## Architecture

```
src/pairs_backtester/
  config.py       loads config/config.yaml, resolves output paths
  data.py         fetch, cache, align, and cut the timeline into windows
  pairs.py        hedge ratio, spread, half-life, z-score, cointegration,
                  selection. THE ONLY MODULE THAT ESTIMATES PARAMETERS
  signals.py      z-score path -> list of trades. No prices, no money
  backtest.py     P&L accounting, costs, position sizing, combining pairs
  metrics.py      performance, trade statistics, market neutrality, cost sweep
  plots.py        figures, on the same palette as the dashboard
  manifest.py     config + git hash + timestamp for every run
  export_site.py  compact JSON for the dashboard
  run.py          the entrypoint that wires the above together
tests/            29 tests: hedge ratio, cointegration, signals, lookahead
config/           every knob
data/cache/       price cache, gitignored and regenerable
results/          committed artifacts from the last run
docs/figures/     committed figures from the last run
```

The split is deliberate rather than decorative. `signals.py` takes a z-score
series and returns trades, so entry and exit rules can be tested against a
hand-written path with no prices, costs or capital involved. `pairs.py` is the
only module that estimates anything, which is what makes the no-lookahead
property checkable in one place instead of everywhere.

### Position sizing and costs

Gross notional is fixed at `capital_per_pair`: one unit of the dependent leg
plus `|beta|` units of the other. Sizing on gross rather than net is the honest
choice for a long-short book, because both legs are real positions that must be
financed and both pay to trade.

Costs are `bps_per_leg` on the traded notional, charged on both legs at entry and
again at exit. That is four chargeable legs per round trip. Pairs trading turns
over often, so this is the line item most likely to decide whether the strategy
works, and it is the reason results are reported before and after costs
throughout rather than net only.

## How lookahead is prevented

A backtest that sees the future fails silently: it produces better numbers, not
an error. Three mechanisms keep it out, and each has a test that fails if the
mechanism is removed.

**Parameters are estimated in one function.** `pairs.fit_pair` only ever receives
a formation slice. The hedge ratio, spread mean and spread standard deviation are
fixed before the trading window starts. `test_fitted_parameters_are_blind_to_the_trading_window`
corrupts every trading-window price with noise and asserts the fitted parameters
do not move by so much as a rounding error.

**The z-score cannot re-centre itself.** `pairs.zscore(sp, mean, std)` takes the
standardisation constants as arguments rather than computing them from its input,
so standardising a trading spread against its own statistics is not something a
caller can do by accident. The test works off a signature: a z-score computed on
its own window is centred at zero by construction, and one computed against fixed
formation parameters is not.

**Execution lags the signal by one bar.** The decision uses the close of bar `i`
and the fill books at the close of bar `i+1`, because acting on bar `i`'s close
requires having seen it. Same-bar fills are a small but real lookahead and they
flatter this strategy specifically, since entries cluster on large moves.

These were verified by mutation rather than by reading them:

| Deliberate bug | Tests that fail |
|---|---|
| `zscore` recomputes its own mean and standard deviation | 3 |
| execution lag removed, same-bar fill | 1 |

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

The signal parameters are conventional values chosen in advance rather than
searched. Searching them for the best result would reintroduce lookahead one
level up, and the honest fix is a third holdout period this project does not
have.

## Outputs

Written to `results/` on every run:

| File | Contents |
|---|---|
| `selected_pairs.csv` | every pair chosen, with the window, p-value, hedge ratio, spread mean and standard deviation, and half-life behind the choice |
| `trades.csv` | every round trip: entry and exit dates, side, entry and exit z, holding days, gross P&L, cost, net P&L, and why it closed |
| `daily_returns.csv` | daily portfolio return, gross and net |
| `cost_sensitivity.csv` | the same trades re-priced across a grid of cost assumptions |
| `summary.json` | performance, trade statistics and market-neutrality figures |
| `manifest.json` | config, git hash, interpreter and platform for the run |
| `dashboard.json` | compact payload for the web dashboard |

And to `docs/figures/`: the spread and z-score of a sample pair with every entry
and exit marked, the equity curve with drawdown, and the cost sensitivity curve.

## Tests

```bash
make test
```

29 tests in four areas:

- **`test_hedge_ratio.py`** recovers a known beta and alpha, checks the spread of
  a perfect relationship is flat, confirms OLS is not symmetric (which is why
  pairs are oriented by a fixed rule), and validates the half-life estimator
  against simulated Ornstein-Uhlenbeck processes with known reversion speed.
- **`test_cointegration.py`** asserts the gate opens and shuts correctly. The
  case that matters builds two series from identical shocks plus a small extra
  drift: their daily changes correlate above 0.99 and Engle-Granger still refuses
  them, because the levels separate without bound.
- **`test_signals.py`** covers entry, exit, stop, timeout, forced close at window
  end, one-position-at-a-time, NaN handling, and the one-bar execution lag.
- **`test_lookahead.py`** is the guard described above, from four angles.

## Extending it

The obvious next steps, in the order I would do them:

- **Johansen** alongside Engle-Granger, which handles more than two series and
  does not depend on which leg is regressed on which.
- **A Kalman-filtered hedge ratio** instead of a fixed OLS beta, so the hedge
  adapts within the trading window. This has to be done carefully: an adaptive
  parameter is exactly where lookahead creeps back in.
- **Intraday or VWAP execution** rather than trading at a close nobody gets
  filled at.
- **A slower variant.** The break-even cost result points at holding period as
  the binding constraint, so a version entering at a wider band and holding
  longer is the change most likely to matter.

## A note on scope

This is a student project. It is not investment advice, and the results should
be read as a study of whether a mechanism survives its own costs rather than as
a strategy to run.
