# Report: does a cointegration pairs strategy on Indian banks survive its costs?

A walk-forward pairs-trading backtest on eleven NSE bank stocks, from pair
selection through to net P&L. The code, tests and instructions for running it
are in the [repository README](../README.md), and there is an
[interactive version](https://arjun-quant-projects.vercel.app/pairs-trading)
of the same results on the site.

---

## Abstract

Every six months from January 2017 to June 2026, all 55 pairs of eleven NSE
banks were tested for cointegration on the previous twelve months of prices. Up
to five pairs that passed were then traded for the next six months on a z-score
rule: enter at 2 standard deviations, exit at 0.5, stop out at 4, give up after
60 days. Every parameter was fixed before the trading period began, and costs
were charged at 20 bps on each leg at entry and again at exit.

Over 2,347 out-of-sample trading days and 242 round trips, the strategy returned
**+1.27% a year before costs** and **-1.29% a year after them**, ending at 1.12x
and 0.89x of its starting capital. Buying and holding the same eleven banks
returned 13.81% a year over the same days.

The gross figure is the sum of two large numbers that almost cancel. The 71
trades that reverted earned **+₹3.21 lakh**; the 130 trades that were stopped out
lost **-₹3.05 lakh**. Stops outnumbered reversions by almost two to one, so a
pair that passed the cointegration test on one year more often than not stopped
behaving that way in the next six months. The ₹46,000 left over was less than
half of the ₹97,000 paid in costs. Break-even is **9.8 bps per leg**, against the
20 bps assumed.

The strategy was market-neutral as designed: its beta to the Nifty was **-0.026**
and the market explained 0.3% of its daily variance.

An earlier version of this backtest reported a far worse result, -9.82% a year
from 1,008 trades. Most of those trades came from a bug in the entry rule, which
is described in section 5.

---

## Table of contents

- [Abstract](#abstract)
- [1. Research question](#1-research-question)
- [2. Data](#2-data)
- [3. Method](#3-method)
- [4. Results](#4-results)
  - [4.1 Headline numbers](#41-headline-numbers)
  - [4.2 One pair, one window](#42-one-pair-one-window)
  - [4.3 Stops cancelled the reversions](#43-stops-cancelled-the-reversions)
  - [4.4 Reversions ran at the speed the formation window predicted](#44-reversions-ran-at-the-speed-the-formation-window-predicted)
  - [4.5 A lower p-value did not pick better pairs](#45-a-lower-p-value-did-not-pick-better-pairs)
  - [4.6 Two bad windows](#46-two-bad-windows)
  - [4.7 Costs](#47-costs)
  - [4.8 Market neutrality](#48-market-neutrality)
- [5. The re-entry bug](#5-the-re-entry-bug)
- [6. Interpretation](#6-interpretation)
- [7. Limitations](#7-limitations)
- [8. What I would do next](#8-what-i-would-do-next)
- [Notes on the figures](#notes-on-the-figures)

---

## 1. Research question

Pairs trading rests on one claim: if two stocks are tied together by a long-run
relationship, then when the gap between them widens it will tend to close, and a
position long the cheap one and short the expensive one will profit when it does.
The question here is whether that claim survives being run forward on data the
model has not seen, at a realistic cost.

> If pairs of Indian bank stocks are chosen by a cointegration test on one year
> of prices and traded on a fixed z-score rule over the next six months, does the
> strategy make money after transaction costs, and is it actually independent of
> the market?

Banks were chosen because they are the sector where a stable relationship between
two prices is most plausible. They share a regulator, a rate cycle and a credit
cycle, so a large part of what moves one moves the others.

## 2. Data

| | |
|---|---|
| **Universe** | 14 NSE banks configured, 11 traded |
| **Prices** | Daily closes from Yahoo Finance, adjusted for splits and dividends |
| **Full window** | 2016-01-01 to 2026-07-31, 2,616 trading days |
| **Out-of-sample period** | 2017-01-02 to 2026-06-30, 2,347 trading days |
| **Benchmark** | Nifty 50 (`^NSEI`), used only to measure beta |
| **Risk-free rate** | 6% a year |

The eleven traded banks are HDFC Bank, ICICI Bank, Axis Bank, Kotak Mahindra
Bank, SBI, IndusInd Bank, Federal Bank, Bank of Baroda, PNB, Canara Bank and IDFC
First Bank. AU Small Finance Bank, Bandhan Bank and RBL Bank were configured but
dropped because they cover less than 95% of the window: RBL listed in August
2016, AU in July 2017 and Bandhan in March 2018. Keeping them would have forced
the whole sample to start later.
After that filter, only dates on which all eleven traded are kept.

## 3. Method

### Pair selection

The timeline is cut into 19 windows. Each has a 12-month **formation** period
followed by a 6-month **trading** period, and the windows step forward six months
at a time, so the trading periods tile 2017 to mid-2026 without overlapping.

On each formation period, all 55 pairs are tested with the Engle-Granger
procedure. One price is regressed on the other by OLS, which gives the hedge
ratio β, and the residual is tested for a unit root. A pair is kept if the
p-value is at most 0.05, and at most five pairs are kept per window, lowest
p-value first. 77 pair-windows were selected in total, covering 45 distinct
pairs, with a median p-value of 0.015.

Cointegration rather than correlation is the right gate. Two stocks can have
daily returns that are 0.99 correlated and still drift apart without limit if one
grows slightly faster. Their spread then has no mean to come back to. The test
suite builds exactly that case and checks that Engle-Granger rejects it.

For each selected pair the formation period also fixes the spread's mean and
standard deviation, and its half-life: how many days a deviation takes to halve,
from an AR(1) fit on the spread. The median half-life across all selections was
6.8 days.

### Trading rules

In the trading period, the spread is `y - βx` and the z-score is its distance
from the formation mean in formation standard deviations. The rules are:

| Condition | Action |
|---|---|
| z above +2 | short the spread: sell y, buy β units of x |
| z below -2 | long the spread: buy y, sell β units of x |
| \|z\| back inside 0.5 | close, reverted |
| \|z\| beyond 4 | close, stopped out |
| 60 trading days open | close, timed out |
| trading period ends | close at the last bar |
| \|z\| already beyond 4 while flat | do nothing |

The last rule is the fix described in section 5. One position per pair is open
at a time. The decision uses a day's close, and the fill is booked at the next
day's close.

These thresholds are conventional values chosen before the backtest was run. I
did not search over them, because choosing them on the same data they are scored
on would bring lookahead back in at the level of the parameters.

### Sizing and costs

Each open position has ₹1 lakh of gross notional: one unit of y plus |β| units of
x, scaled so the two legs sum to ₹1 lakh. Capital is split equally across the
pairs selected for a window, and a pair with no open position earns zero that
day rather than being left out of the average.

Costs are 20 bps per leg on traded notional: about 5 bps of brokerage and taxes,
10 bps of half-spread and 5 bps of slippage. Both legs pay at entry and again at
exit, which comes to 40 bps of gross notional per round trip, or about ₹400.

### Keeping the future out

Three mechanisms keep trading-period information out of the decisions, and each
has a test that fails when the mechanism is removed.

- **Parameters come from one function that only sees the formation period.** The
  test adds noise to every trading-period price and checks that the hedge ratio,
  mean and standard deviation do not change at all.
- **The z-score function takes the mean and standard deviation as arguments.** It
  cannot standardise a spread against its own statistics. The test relies on the
  fact that a series standardised against itself has mean zero by construction,
  and one standardised against fixed formation values does not.
- **Fills lag the signal by one day.** Filling at the same close that triggered
  the signal assumes that close was known before it happened. This matters more
  for pairs trading than for most strategies, because entries happen on large
  moves.

Making the z-score recompute its own statistics fails three tests, and removing
the one-day lag fails one.

## 4. Results

### 4.1 Headline numbers

| | Before costs | After costs | Buy and hold the 11 banks | Nifty 50 |
|---|---:|---:|---:|---:|
| Growth of 1 | 1.12x | 0.89x | 3.34x | 2.92x |
| Annualised return | +1.27% | -1.29% | +13.81% | +12.18% |
| Volatility | 7.27% | 7.30% | 25.31% | 16.35% |
| Sharpe (6% risk-free) | -0.65 | -1.00 | 0.31 | 0.38 |
| Max drawdown | -13.59% | -20.17% | -57.95% | -38.44% |

| Trades | |
|---|---:|
| Round trips | 242 |
| Average holding period | 15.9 days (median 11) |
| Profitable before costs | 52.1% |
| Profitable after costs | 46.7% |
| Average gross P&L per trade | ₹192, or 19 bps of notional |
| Average cost per trade | ₹403, or 40 bps of notional |

The Sharpe ratio is negative even before costs because 1.27% a year is well below
the 6% risk-free rate. A market-neutral book would normally earn interest on the
cash it posts as collateral, which this backtest does not credit; section 7
returns to this.

![Out-of-sample growth of 1 before and after costs, with the after-cost drawdown](https://arjun-quant-projects.vercel.app/figures/pairs/equity.png)

The before-cost line peaked at about 1.24x in late 2023. Its two largest falls
came in 2020 and in March 2025, and section 4.6 traces both. The after-cost line
has the same shape and finishes 23 points lower, which is the cumulative cost of
242 round trips.

### 4.2 One pair, one window

![HDFC Bank against Kotak Mahindra Bank, January to June 2017, with every trade shaded](https://arjun-quant-projects.vercel.app/figures/pairs/pair-example.png)

HDFC Bank against Kotak Mahindra Bank in the first trading window shows both
outcomes the rules produce. The pair passed the test with p = 0.017 and a
half-life of 6.4 days. It was selected in four of the 19 windows, which ties it
for the most often selected pair.

- **January to February:** the spread opened above +2 and drifted further up
  instead of reverting. It crossed +4 in late February and the trade was stopped
  out for -₹3,396.
- **March to May:** three short trades were signalled at z = 3.29, 2.47 and 2.58.
  Each closed when z came back to within 0.2 of zero, for +₹3,468, +₹1,048 and
  +₹3,346.
- **June:** the spread climbed from +2 to above +4 again, and the last trade was
  stopped out for -₹2,005.

Markers sit at the fill, one day after the signal, which is why some entries
land inside the entry band and some exits overshoot the exit band.

Three of five trades reverted in this window. Across the whole backtest the
figure was 71 of 242.

### 4.3 Stops cancelled the reversions

![Waterfall of gross P&L by exit reason, then costs, to net P&L](https://arjun-quant-projects.vercel.app/figures/pairs/pnl-by-exit.png)

| How the trade ended | Trades | Share | Gross P&L | Costs | Net P&L | Median days held |
|---|---:|---:|---:|---:|---:|---:|
| Reverted inside 0.5 | 71 | 29.3% | +₹3,20,605 | ₹28,477 | +₹2,92,128 | 14 |
| Stopped out beyond 4 | 130 | 53.7% | -₹3,05,070 | ₹52,401 | -₹3,57,471 | 3.5 |
| Timed out at 60 days | 8 | 3.3% | +₹10,173 | ₹3,243 | +₹6,930 | 60 |
| Trading period ended | 33 | 13.6% | +₹20,660 | ₹13,365 | +₹7,295 | 25 |
| **Total** | **242** | | **+₹46,368** | **₹97,486** | **-₹51,118** | 11 |

This table is the main result. A reverted trade earned ₹4,516 on average, and a
stopped trade lost ₹2,347, about half as much. With those payoffs the strategy
needs reversions to be at least about a third of the reversion-plus-stop total
to break even before costs. They were 35%, which is why gross P&L is barely
positive.

A stop at |z| = 4 means the spread moved two more formation standard deviations
against the position. 130 of 242 trades did that, half of them within four days
of entry. These pairs had passed a cointegration test on the previous twelve months,
and in more than half of the trades the relationship did not hold over the next
few weeks.

### 4.4 Reversions ran at the speed the formation window predicted

The formation half-life is a forecast of how fast a deviation should decay. For
the 71 trades that reverted, the median holding period was 14 days against a
median formation half-life of 6.4 days for those pairs, a ratio of 1.98.

That ratio is close to what the half-life implies. The median reverted trade
entered at |z| = 2.28 and exits at 0.5, so the deviation has to shrink by a
factor of 4.6, which takes log2(4.6) = 2.2 half-lives. When a pair did revert, it
reverted at about the predicted speed. The estimate of speed held up. What failed
was the assumption that the pair would keep reverting at all.

### 4.5 A lower p-value did not pick better pairs

If the cointegration test measured how reliable a pair would be, pairs with
smaller p-values should have stopped out less often. They did not.

| Formation p-value | Trades | Stopped out | Reverted | Average gross P&L |
|---|---:|---:|---:|---:|
| 0 to 0.01 | 93 | 59.1% | 26.9% | ₹85 |
| 0.01 to 0.03 | 89 | 46.1% | 33.7% | ₹302 |
| 0.03 to 0.05 | 60 | 56.7% | 26.7% | ₹192 |

The pairs with the strongest test results had the highest stop rate and the
lowest average P&L. With 60 to 93 trades per row, the differences between rows
are within noise, so the fair reading is that the p-value carried no information
about the next six months. Ranking pairs by p-value, as the selection does, is
therefore not doing useful work beyond the 0.05 cut-off.

### 4.6 Two bad windows

![Calendar-year return before and after costs](https://arjun-quant-projects.vercel.app/figures/pairs/annual.png)

Before costs the strategy made money in 6 of 9 full calendar years, and in 12 of
19 trading windows. Most of the loss sits in a few episodes.

**2020, -9.86% before costs.** The COVID crash moved bank stocks by very
different amounts. Axis Bank against Federal Bank was stopped out six times that
year for -₹20,810, and IndusInd Bank against IDFC First Bank lost -₹20,500 across
seven trades.

**January to June 2025, -₹43,737 in one window.** The formation year (2024)
selected five pairs, and four of them had Kotak Mahindra Bank as one leg. On
2025-01-21 three of those four entered on the same day, each short Kotak against
a public-sector bank (Bank of Baroda, Canara Bank and PNB). All three were
stopped out on 2025-03-24 for about -₹12,000 each. This was a single bet against
Kotak placed three times. That one window lost almost exactly the strategy's
entire gross profit over nine and a half years.

The selection rule allows this. Nothing stops one bank appearing in several of
the five pairs, and in 7 of the 19 windows a single bank was a leg of three or
more selected pairs.

### 4.7 Costs

![Annualised return against the per-leg cost assumption, with the break-even marked](https://arjun-quant-projects.vercel.app/figures/pairs/cost-sensitivity.png)

The signal never uses costs, so the same 242 trades can be re-priced at any cost
level. Cost is linear in the rate, so each point on the curve is exact rather
than an approximation, and the point at 20 bps reproduces the backtest's own net
figure.

The break-even follows from two numbers in section 4.1. The average trade earned
19 bps of notional before costs, and a round trip costs 2c of notional at c bps
per leg. Setting 2c = 19 gives c = 9.6 bps; the compounded return series puts it
at 9.8 bps. Either way, the strategy needs roughly half the cost assumed here.

That is within reach of institutional execution on large-cap Indian banks, but
not of retail execution. It would also not be enough: at zero cost the strategy
returns 1.27% a year with 7.3% volatility, which does not beat a bank deposit.

### 4.8 Market neutrality

| | Value |
|---|---:|
| Beta to the Nifty | -0.026 |
| Correlation with the Nifty | -0.057 |
| R² | 0.003 |
| Nifty annualised return, same days | 12.18% |

A beta of -0.026 means a 1% move in the Nifty moved the strategy by about 0.03%
on average. The long-short structure removed market exposure as designed. The
P&L came entirely from relative moves between banks, and those relative moves did
not reliably reverse.

## 5. The re-entry bug

The first complete run of this backtest reported 1,008 trades, -9.82% a year
after costs, a 62% drawdown and a break-even cost of 1.9 bps per leg. While
writing this report I broke the trades down by exit reason and found 892 stops
with a median holding period of one day.

The cause was the entry rule. When the book was flat, it entered whenever
|z| ≥ 2, without checking whether |z| was also ≥ 4. Once a spread broke past the
stop, the strategy entered, was stopped out the next day, entered again the day
after, and repeated this for as long as the spread stayed wide. Each loop paid
four legs of cost.

| | First run | After the fix |
|---|---:|---:|
| Trades | 1,008 | 242 |
| Entered with \|z\| already ≥ 4 | 816 | 0 |
| Stopped out | 892 | 130 |
| Median holding period | 1 day | 11 days |
| Annualised return before costs | +1.04% | +1.27% |
| Annualised return after costs | -9.82% | -1.29% |
| Max drawdown after costs | -62.16% | -20.17% |
| Break-even cost per leg | 1.9 bps | 9.8 bps |

The fix is one condition: a flat book does not enter while |z| is at or beyond
the stop. It can re-enter once the spread comes back inside the stop band. Two
tests now cover both halves of that rule.

A second, smaller error was in the cost sweep. It spread each trade's cost across
the average number of pairs per window instead of the actual number in that
window, so at the assumed 20 bps it reported -9.20% a year while the backtest
reported -9.82%. The sweep now scales the backtest's own daily cost series, so
the two agree exactly.

The bug moved the net result by 8.5 percentage points a year and the gross
result by only 0.2. Because the damage showed up almost entirely as cost, the
first run read as a strategy with a real edge that costs destroyed. That reading
was wrong, and the exit-reason breakdown is what exposed it.

## 6. Interpretation

**Cointegration over a year does not mean cointegration over the next six
months.** The test found pairs whose spread reverted at a measurable speed in the
formation period, and when those pairs did revert in the trading period they did
so at that speed. But 54% of trades were stopped out, and the p-value gave no
warning of which ones. The relationship the strategy depends on held less often
than it broke.

**The edge was already too small before costs.** Gross of costs the
strategy returned 1.27% a year. Half the assumed cost would have brought it to
zero, and zero cost would still have left it below the risk-free rate. Cheaper
execution alone cannot fix this strategy.

**Concentration caused the worst losses.** The largest single loss came from
three positions that were effectively one view on Kotak Mahindra Bank. A cap on
how many selected pairs can share a bank would have prevented it.

**The structure did what it was built to do.** Beta was -0.026 and the market
explained 0.3% of daily variance. Market neutrality is the part of the design
that worked.

## 7. Limitations

- **One sector, one market.** All eleven stocks are Indian banks. The result says
  nothing about pairs in other sectors or markets.
- **No interest on collateral.** A real long-short book posts margin that earns
  interest. Crediting it would lift both the gross and net returns by roughly the
  short-term rate on the margin posted, which would make the Sharpe ratio less
  negative but would not change the trade-level result in section 4.3.
- **Short selling is assumed to be free and always available.** In India,
  shorting a stock for more than a day needs securities lending or single-stock
  futures, and both have costs this backtest ignores. Futures exist for all
  eleven banks, so the position is feasible, but the futures basis and roll costs
  are not modelled.
- **Flat costs.** 20 bps per leg does not vary with volatility, trade size or
  liquidity, and it is likely too low on days when spreads blow out, which are
  the days the stops fire.
- **Daily closes.** Every trade fills at a closing price, which nobody can
  transact at in size.
- **Thresholds chosen, not validated.** Entry at 2, exit at 0.5, stop at 4, a
  60-day timeout and a 12-month formation window are conventional values. A
  proper test of alternatives would need a third holdout period this sample does
  not have.
- **Survivorship.** The universe was picked in 2026 from banks that are still
  listed. Banks merged away during the sample, such as Syndicate Bank, Oriental
  Bank of Commerce and Allahabad Bank in the 2020 public-sector mergers, and
  Lakshmi Vilas Bank, merged into DBS India the same year, were never
  candidates. Those are the pairs most likely to have broken down.
- **Small samples per bucket.** The p-value and exit-reason tables rest on 60 to
  130 trades per row. The differences in section 4.5 are not statistically
  significant, which is the reason for reading them as "no relationship" rather
  than as a reversed one.

## 8. What I would do next

In order of expected impact:

1. **Cap how many selected pairs can share one bank**, for example at two. This
   addresses the largest single loss directly.
2. **Re-test cointegration during the trading period** on a rolling window that
   only uses past data, and close or skip a pair once it fails. A stop at |z| = 4
   detects a broken pair late and at the worst price.
3. **A Kalman-filtered hedge ratio** that updates through the trading period, so
   that a slow drift in the relationship is tracked rather than treated as a
   signal. The update has to use only past data, which needs its own lookahead
   test.
4. **Johansen instead of Engle-Granger**, which does not depend on which stock is
   regressed on which, and extends to baskets of three or more banks.
5. **Model short-side costs through single-stock futures**, including the basis
   and the monthly roll.

## Notes on the figures

All five figures are drawn by
[`src/pairs_backtester/plots.py`](../src/pairs_backtester/plots.py) on every run
of the pipeline, from the same data that produces the tables, so a figure cannot
disagree with the numbers next to it. The files are in
[`docs/figures/`](figures/); the copies linked above are served from the site.

Blue and amber carry the two sides of every comparison: before and after costs,
profit and loss. Green and red are the usual choice, but they are hard to tell
apart for readers with red-green colour blindness, and blue and amber are not.
Every other element is grey, and labels are in black or grey text rather than in
the colour of the line they describe.
