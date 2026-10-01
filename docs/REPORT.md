# Cointegration-based pairs trading in Indian bank stocks: a walk-forward evaluation

This report documents the study in full: question, data, method, results and
limitations. Code, tests and instructions for reproducing every number are in the
[repository README](../README.md). An
[interactive version](https://arjun-quant-projects.vercel.app/pairs-trading) of
the same results is on the project site.

---

## Abstract

This study evaluates a pairs-trading strategy on eleven NSE-listed banks. Every
six months from January 2017 to June 2026, all 55 pairs are tested for
cointegration with the Engle-Granger procedure on the preceding twelve months,
and up to five pairs with p ≤ 0.05 are traded over the following six months. A
position is opened when the standardised spread moves two standard deviations
from its formation-period mean, closed when it returns within 0.5, and stopped
out at 4. All parameters are fixed before each trading period, and transaction
costs of 20 bps per leg are charged at entry and exit.

Over 2,347 out-of-sample trading days and 242 round trips, the strategy returns
+1.27% a year before costs and -1.29% after costs, with an annualised volatility
of 7.3%. The mean daily return before costs is not significantly different from
zero (Newey-West t = 0.67). The 71 trades that converged earned ₹3.21 lakh before
costs and the 130 trades that were stopped out lost ₹3.05 lakh, so stopped trades
outnumbered converged trades by 1.8 to 1. The formation-period p-value is not
associated with the stop-out rate (χ²(2) = 3.40, p = 0.18). The strategy's beta to
the Nifty 50 is -0.026, and the break-even transaction cost is 9.8 bps per leg.

An earlier version of the backtest contained an entry-rule error that generated
816 spurious trades and overstated the after-cost loss by 8.5 percentage points
a year. Section 5 describes the error and its correction.

---

## Contents

- [1. Introduction](#1-introduction)
- [2. Data](#2-data)
- [3. Methodology](#3-methodology)
- [4. Results](#4-results)
  - [4.1 Performance](#41-performance)
  - [4.2 An example pair](#42-an-example-pair)
  - [4.3 Decomposition by exit type](#43-decomposition-by-exit-type)
  - [4.4 Speed of convergence](#44-speed-of-convergence)
  - [4.5 Formation p-value and trade outcome](#45-formation-p-value-and-trade-outcome)
  - [4.6 Performance by period](#46-performance-by-period)
  - [4.7 Transaction costs](#47-transaction-costs)
  - [4.8 Market exposure](#48-market-exposure)
- [5. Correction of an entry-rule error](#5-correction-of-an-entry-rule-error)
- [6. Discussion](#6-discussion)
- [7. Limitations](#7-limitations)
- [8. Further work](#8-further-work)
- [References](#references)
- [Appendix: figures](#appendix-figures)

---

## 1. Introduction

Pairs trading takes offsetting positions in two securities whose prices have
historically moved together, buying the relatively cheap one and selling the
relatively expensive one when their price relationship diverges, and closing the
position when it reverts. Gatev, Goetzmann and Rouwenhorst (2006) report average
annualised excess returns of up to 11% for a distance-based version of the rule
on US equities from 1962 to 2002. Do and Faff (2010) find that the profitability
of the same rule declined substantially in later years. Krauss (2017) reviews the
main variants, among which the cointegration approach selects pairs whose price
spread is stationary.

Cointegration (Engle and Granger, 1987) is a stricter condition than correlation.
Two price series can have highly correlated daily returns and still diverge
without bound if one has a slightly higher drift. If instead a linear combination
$y_t - \beta x_t$ is stationary, deviations from its mean are expected to decay,
which is the property on which a convergence trade depends (Vidyamurthy, 2004).

This study tests that property out of sample:

> If pairs of Indian bank stocks are selected by a cointegration test on twelve
> months of prices and traded on a fixed z-score rule over the following six
> months, does the strategy earn a positive return after transaction costs, and
> is its return independent of the market?

Bank stocks were chosen because they share a regulator, an interest-rate cycle
and a credit cycle, which makes a stable long-run relationship between their
prices economically plausible.

## 2. Data

| | |
|---|---|
| **Universe** | 14 NSE banks configured, 11 traded |
| **Prices** | Daily closes from Yahoo Finance, adjusted for splits and dividends |
| **Full sample** | 2016-01-01 to 2026-07-31, 2,616 trading days |
| **Out-of-sample period** | 2017-01-02 to 2026-06-30, 2,347 trading days |
| **Benchmark** | Nifty 50 (`^NSEI`), used only to estimate market exposure |
| **Risk-free rate** | 6% per year |

The eleven traded banks are HDFC Bank, ICICI Bank, Axis Bank, Kotak Mahindra
Bank, State Bank of India, IndusInd Bank, Federal Bank, Bank of Baroda, Punjab
National Bank, Canara Bank and IDFC First Bank. RBL Bank (listed August 2016), AU
Small Finance Bank (July 2017) and Bandhan Bank (March 2018) were excluded because
their histories cover less than 95% of the sample. After that filter, only dates
on which all eleven banks traded are retained.

## 3. Methodology

### 3.1 Formation and trading periods

The sample is divided into 19 windows, each consisting of a 12-month formation
period followed by a 6-month trading period. Windows advance by six months, so the
trading periods are contiguous and do not overlap, covering January 2017 to June
2026.

### 3.2 Pair selection

In each formation period, each of the 55 pairs is tested with the two-step
Engle-Granger procedure. The first step estimates the hedge ratio by ordinary
least squares,

$$
y_t = \alpha + \beta x_t + e_t ,
$$

and the second tests the residual $e_t$ for a unit root with an augmented
Dickey-Fuller test using MacKinnon critical values (`statsmodels.tsa.stattools.coint`).
The null hypothesis is no cointegration. The dependent leg
is the stock with the higher average price, so that the orientation of each pair
is fixed by rule. Pairs with p ≤ 0.05 are retained, at most five per window,
lowest p-value first.

For each retained pair the formation period also fixes the spread mean $\mu_s$,
the spread standard deviation $\sigma_s$ and the half-life of mean reversion,
$h = -\ln 2 / \kappa$, where $\kappa$ is the slope from regressing the daily change
in the formation spread on its lagged level,
$\Delta s_t = a + \kappa s_{t-1} + u_t$ (a discrete Ornstein-Uhlenbeck
approximation). Over the sample, 77 pair-windows were selected from 45
distinct pairs, with a median p-value of 0.015 and a median half-life of 6.8
trading days.

### 3.3 Trading rule

During the trading period the spread is $s_t = y_t - \beta x_t$ and the z-score is
$z_t = (s_t - \mu_s) / \sigma_s$, using the formation-period values throughout.

| Condition | Action |
|---|---|
| $z_t > +2$ | open short the spread: sell $y$, buy $\beta$ units of $x$ |
| $z_t < -2$ | open long the spread: buy $y$, sell $\beta$ units of $x$ |
| $\lvert z_t \rvert \le 0.5$ | close (converged) |
| $\lvert z_t \rvert \ge 4$ | close (stopped out) |
| position open 60 trading days | close (timed out) |
| end of trading period | close at the last bar |
| flat and $\lvert z_t \rvert \ge 4$ | no entry |

At most one position per pair is open at a time. Signals are computed from the
close of day $t$ and executed at the close of day $t+1$. The thresholds are
conventional values fixed before the backtest was run; they were not selected by
performance, which would introduce look-ahead at the level of the parameters.

### 3.4 Position sizing and costs

Each position has a gross notional of ₹1 lakh, the sum of the absolute values of
the two legs. Capital is divided equally among the pairs selected for a window,
and a pair without an open position contributes a zero return for the day.

Costs are 20 bps per leg on traded notional, comprising about 5 bps of brokerage
and taxes, 10 bps of half bid-ask spread and 5 bps of slippage. Both legs are
charged at entry and at exit, so a round trip costs 40 bps of gross notional,
about ₹400 per trade.

### 3.5 Controls against look-ahead

Three mechanisms prevent trading-period data from entering any decision, and
each is covered by a test that fails when the mechanism is removed.

1. **Estimation is confined to the formation period.** The hedge ratio, spread
   mean and spread standard deviation are computed by a single function that
   receives only formation data. The corresponding test perturbs every
   trading-period price and asserts that the estimated parameters are unchanged.
2. **The z-score uses fixed parameters.** The standardisation function takes the
   mean and standard deviation as arguments rather than computing them from its
   input, so a trading-period spread cannot be standardised against its own
   statistics.
3. **Execution is lagged by one day.** A signal computed from the close of day $t$
   is filled at the close of day $t+1$. Filling at the same close would assume the
   signal was known before the price that produced it.

Reintroducing the second error causes three tests to fail, and removing the
execution lag causes one.

## 4. Results

### 4.1 Performance

| | Before costs | After costs | Equal-weight buy and hold, 11 banks | Nifty 50 |
|---|---:|---:|---:|---:|
| Growth of 1 | 1.12 | 0.89 | 3.34 | 2.92 |
| Annualised return | +1.27% | -1.29% | +13.81% | +12.18% |
| Annualised volatility | 7.27% | 7.30% | 25.31% | 16.35% |
| Sharpe ratio ($r_f$ = 6%) | -0.65 | -1.00 | 0.31 | 0.38 |
| Maximum drawdown | -13.59% | -20.17% | -57.95% | -38.44% |

| Trade statistics | |
|---|---:|
| Round trips | 242 |
| Mean holding period | 15.9 trading days (median 11) |
| Profitable before costs | 52.1% |
| Profitable after costs | 46.7% |
| Mean gross P&L per trade | ₹192 (19 bps of notional) |
| Mean cost per trade | ₹403 (40 bps of notional) |

The mean daily return before costs is 0.61 bps. With Newey-West standard errors
(10 lags), its t-statistic is 0.67 (p = 0.50); after costs it is -0.46 (p = 0.65).
At the trade level, the mean gross P&L of ₹192 has a t-statistic of 0.70. The
gross return is therefore not distinguishable from zero.

The Sharpe ratio is negative before costs because the annual return of 1.27% is
below the 6% risk-free rate. A market-neutral portfolio would in practice earn
interest on posted collateral, which this backtest does not credit (section 7).

![Out-of-sample growth of 1 before and after costs, with the after-cost drawdown](https://arjun-quant-projects.vercel.app/figures/pairs/equity.png)

The before-cost series peaks at about 1.24 in late 2023. Its two largest declines
occur in 2020 and in March 2025 (section 4.6). The after-cost series has the same
shape and ends 23 points lower, which is the cumulative cost of 242 round trips.

### 4.2 An example pair

![HDFC Bank against Kotak Mahindra Bank, January to June 2017, with each trade shaded](https://arjun-quant-projects.vercel.app/figures/pairs/pair-example.png)

HDFC Bank against Kotak Mahindra Bank in the first trading window illustrates
each outcome of the rule. The pair passed the formation test with p = 0.017 and a
half-life of 6.4 days, and it is one of the two pairs selected most often (four
of 19 windows).

- **January to February.** The spread opened above +2 and continued to widen. It
  crossed +4 in late February and the position was stopped out for -₹3,396.
- **March to May.** Three short positions were signalled at z = 3.29, 2.47 and
  2.58. Each closed when z returned to within 0.2 of zero, for +₹3,468, +₹1,048
  and +₹3,346.
- **June.** The spread rose from +2 to above +4 again and the final position was
  stopped out for -₹2,005.

Markers are plotted at execution, one day after the signal, so some entries fall
inside the entry band and some exits beyond the exit band. Three of the five
trades in this window converged, against 71 of 242 across the whole backtest.

### 4.3 Decomposition by exit type

![Waterfall of gross P&L by exit type, then costs, to net P&L](https://arjun-quant-projects.vercel.app/figures/pairs/pnl-by-exit.png)

| Exit | Trades | Share | Gross P&L | Costs | Net P&L | Median holding (days) |
|---|---:|---:|---:|---:|---:|---:|
| Converged within 0.5 | 71 | 29.3% | +₹3,20,605 | ₹28,477 | +₹2,92,128 | 14 |
| Stopped out at 4 | 130 | 53.7% | -₹3,05,070 | ₹52,401 | -₹3,57,471 | 3.5 |
| Timed out at 60 days | 8 | 3.3% | +₹10,173 | ₹3,243 | +₹6,930 | 60 |
| Trading period ended | 33 | 13.6% | +₹20,660 | ₹13,365 | +₹7,295 | 25 |
| **Total** | **242** | | **+₹46,368** | **₹97,486** | **-₹51,118** | 11 |

A converged trade earned ₹4,516 on average and a stopped trade lost ₹2,347. At
these average payoffs, converged trades must make up at least 34.2% of converged
and stopped trades combined for the strategy to break even before costs. The
realised share is 35.3% (71 of 201), which is not significantly different from
that threshold (binomial p = 0.77). This is why the gross P&L is close to zero.

A stop at $\lvert z \rvert = 4$ means the spread moved a further two formation
standard deviations against the position. 130 of 242 trades reached that level,
half of them within four trading days of entry. For more than half of the trades,
then, a relationship that passed a cointegration test over the previous twelve
months did not hold over the following weeks.

### 4.4 Speed of convergence

The formation half-life predicts how quickly a deviation should decay. For the 71
converged trades, the median holding period is 14 trading days and the median
formation half-life of the corresponding pairs is 6.4 days, a ratio of 1.98.

That ratio is close to the theoretical value. The median converged trade entered
at $\lvert z \rvert = 2.28$ and exits at 0.5, so the deviation must decline by a
factor of 4.6, which takes $\log_2 4.6 = 2.2$ half-lives under an AR(1) process.
Conditional on convergence, the spread reverted at approximately the rate
estimated in the formation period. The estimate of speed held out of sample; the
assumption that the spread would continue to revert did not hold for most trades.

### 4.5 Formation p-value and trade outcome

If the formation p-value measured how reliable a pair's relationship would be, a
smaller p-value should be associated with fewer stop-outs.

| Formation p-value | Trades | Stopped out | Converged | Mean gross P&L |
|---|---:|---:|---:|---:|
| 0 to 0.01 | 93 | 59.1% | 26.9% | ₹85 |
| 0.01 to 0.03 | 89 | 46.1% | 33.7% | ₹302 |
| 0.03 to 0.05 | 60 | 56.7% | 26.7% | ₹192 |

The stop-out rate does not decline with the p-value; the lowest-p-value group has
the highest rate. A χ² test of independence between p-value group and stop-out
gives χ²(2) = 3.40, p = 0.18. Within the pairs that pass the 0.05 threshold, the
p-value carries no detectable information about outcomes in the following six
months, so ranking pairs by it contributes nothing beyond the threshold itself.

### 4.6 Performance by period

![Calendar-year return before and after costs](https://arjun-quant-projects.vercel.app/figures/pairs/annual.png)

Before costs the strategy was profitable in 6 of 9 full calendar years and in 12
of 19 trading windows. The losses are concentrated in a small number of episodes.

**2020: -9.86% before costs.** During the COVID-19 sell-off, bank stocks fell by
very different amounts. Axis Bank against Federal Bank was stopped out six times
during the year for -₹20,810, and IndusInd Bank against IDFC First Bank lost
-₹20,500 over seven trades.

**January to June 2025: -₹43,737 in one window.** Four of the five pairs selected
on the 2024 formation period had Kotak Mahindra Bank as a leg. On 2025-01-21,
three of them entered on the same day, each short Kotak against a public-sector
bank (Bank of Baroda, Canara Bank and Punjab National Bank). All three were
stopped out on 2025-03-24 for about -₹12,000 each. The three positions amounted to
a single exposure to Kotak relative to public-sector banks. The window's loss
before costs approximately equals the strategy's entire gross profit over the
nine and a half years.

The selection rule does not limit how many selected pairs share a stock. In 7 of
the 19 windows, one bank was a leg of three or more selected pairs.

### 4.7 Transaction costs

![Annualised return against the per-leg cost assumption, with the break-even marked](https://arjun-quant-projects.vercel.app/figures/pairs/cost-sensitivity.png)

The signal does not depend on costs, so the same 242 trades can be repriced at any
cost level. Costs are linear in the rate, so each point on the curve is exact,
and the value at 20 bps equals the backtest's net result.

The break-even follows from the trade statistics in section 4.1. The mean trade
earned 19 bps of notional before costs, and a round trip costs $2c$ bps at $c$ bps
per leg, giving $c \approx 9.6$ bps; the compounded daily return series gives 9.8
bps. The strategy would break even at roughly half the assumed cost, which is
within the range of institutional execution in large-capitalisation Indian banks
but below typical retail costs. At zero cost it would return 1.27% a year with
7.3% volatility, which is below the risk-free rate.

### 4.8 Market exposure

| | Value |
|---|---:|
| Beta to the Nifty 50 | -0.026 |
| Correlation with the Nifty 50 | -0.057 |
| $R^2$ | 0.003 |
| Nifty 50 annualised return, same days | 12.18% |

A beta of -0.026 implies that a 1% move in the Nifty 50 is associated with a 0.03%
move in the strategy. The long-short construction removed market exposure as
intended, and the strategy's return derives from relative price movements between
banks.

## 5. Correction of an entry-rule error

The first complete run of the backtest reported 1,008 trades, an after-cost
return of -9.82% a year, a maximum drawdown of 62% and a break-even cost of 1.9
bps per leg. A breakdown by exit type showed 892 stop-outs with a median holding
period of one day.

The entry rule opened a position whenever $\lvert z_t \rvert \ge 2$ without
checking whether $\lvert z_t \rvert$ also exceeded the stop level of 4. When a
spread moved beyond the stop, the strategy entered, was stopped out the next day,
and re-entered, for as long as the spread remained beyond the stop, paying four
legs of cost on each cycle.

| | First run | Corrected |
|---|---:|---:|
| Trades | 1,008 | 242 |
| Entered with \|z\| already ≥ 4 | 816 | 0 |
| Stopped out | 892 | 130 |
| Median holding period | 1 day | 11 days |
| Annualised return before costs | +1.04% | +1.27% |
| Annualised return after costs | -9.82% | -1.29% |
| Maximum drawdown after costs | -62.16% | -20.17% |
| Break-even cost per leg | 1.9 bps | 9.8 bps |

The correction adds one condition: no entry while $\lvert z_t \rvert \ge 4$. A
position may be opened once the spread returns inside the stop level. Two tests
cover the rule.

A second error affected the cost-sensitivity calculation, which allocated each
trade's cost using the average number of pairs per window rather than the actual
number, giving -9.20% a year at 20 bps against the backtest's -9.82%. The
calculation now scales the backtest's own daily cost series, and the two agree
exactly.

The first error changed the after-cost return by 8.5 percentage points a year and
the before-cost return by 0.2. Because its effect appeared almost entirely as
cost, the first run was consistent with a strategy that has a gross edge but
cannot cover its costs. The decomposition by exit type showed that this reading
was incorrect.

## 6. Discussion

**Cointegration in one year does not imply cointegration in the next six
months.** The formation test selected pairs whose spreads reverted at a
measurable rate, and when those spreads converged in the trading period they did
so at approximately that rate. However, 54% of trades were stopped out, and the
formation p-value did not predict which. The relationship on which the strategy
depends broke down more often than it held.

**The gross return is indistinguishable from zero.** Before costs the strategy
returned 1.27% a year, with a Newey-West t-statistic of 0.67. Lower transaction
costs would raise the net return towards that figure but would not produce a
return above the risk-free rate. This is consistent with the decline in
pairs-trading profitability documented by Do and Faff (2010).

**Concentration produced the largest loss.** The worst window's loss came from
three positions that were in effect a single exposure to one stock. A limit on how
many selected pairs may share a stock would have prevented it.

**The market-neutral construction worked as intended.** Beta to the Nifty 50 was
-0.026 and the market explained 0.3% of daily return variance.

## 7. Limitations

- **One sector, one market.** All eleven stocks are Indian banks, and the results
  do not extend to other sectors or markets.
- **No interest on collateral.** A long-short portfolio posts margin that earns
  interest. Crediting it would raise both the before-cost and after-cost returns
  and make the Sharpe ratio less negative, but would not change the trade-level
  results in section 4.3.
- **Short positions assumed costless.** Holding a short equity position in India
  for more than a day requires securities lending or single-stock futures. Futures
  are available on all eleven banks, but their basis and roll costs are not
  modelled.
- **Constant costs.** The 20 bps per leg does not vary with volatility, trade size
  or liquidity, and is likely too low on days when spreads widen sharply, which are
  the days on which stops are triggered.
- **Daily closing prices.** All trades are executed at closing prices, at which
  large orders could not be filled.
- **Thresholds not validated.** The entry, exit and stop levels, the 60-day limit
  and the 12-month formation period are conventional values. Testing alternatives
  properly would require a further holdout period.
- **Survivorship.** The universe consists of banks listed in 2026. Banks merged
  during the sample, including Syndicate Bank, Oriental Bank of Commerce and
  Allahabad Bank in the 2020 public-sector mergers and Lakshmi Vilas Bank, merged
  into DBS Bank India in 2020, were never candidates.
- **Sample sizes.** The tables in sections 4.3 and 4.5 rest on 60 to 130 trades per
  row, and differences between rows should be read with that in mind.

## 8. Further work

1. **Limit stock concentration**, for example to two selected pairs per bank.
2. **Test cointegration during the trading period** on a rolling window of past
   data, and stop trading a pair once the test fails. A stop at $\lvert z \rvert = 4$
   detects a broken relationship late and at an adverse price.
3. **Estimate the hedge ratio with a Kalman filter** so that it adapts during the
   trading period, with a dedicated look-ahead test for the filtered estimate.
4. **Use the Johansen procedure**, which does not depend on the choice of dependent
   variable and extends to baskets of three or more stocks.
5. **Model the short leg through single-stock futures**, including basis and roll
   costs.

## References

Do, B., & Faff, R. (2010). Does simple pairs trading still work? *Financial
Analysts Journal*, 66(4), 83-95.

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction:
Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading:
Performance of a relative-value arbitrage rule. *Review of Financial Studies*,
19(3), 797-827.

Krauss, C. (2017). Statistical arbitrage pairs trading strategies: Review and
outlook. *Journal of Economic Surveys*, 31(2), 513-545.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
*Econometrica*, 55(3), 703-708.

Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*.
Hoboken, NJ: Wiley.

## Appendix: figures

All five figures are generated by
[`src/pairs_backtester/plots.py`](../src/pairs_backtester/plots.py) on every run
of the pipeline, from the same data that produces the tables. The files are in
[`docs/figures/`](figures/); the copies linked above are served from the project
site.

Blue and amber encode the two sides of every comparison (before and after costs,
profit and loss). They remain distinguishable under red-green colour vision
deficiency, which the conventional green and red do not. All other elements are
grey, and labels are set in text colours rather than series colours.
