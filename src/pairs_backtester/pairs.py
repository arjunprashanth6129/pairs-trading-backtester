"""Hedge ratio, spread construction, and the parameters a pair carries forward.

This module is the only place parameters are estimated. Everything downstream
consumes a frozen `PairParams` and re-estimates nothing, which is what makes the
no-lookahead guarantee checkable in one place instead of everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class PairParams:
    """Everything the trading window is allowed to know about a pair.

    All five numbers are computed on the formation window and then held fixed.
    `mean` and `std` in particular are the z-score's reference point: recomputing
    them on trading data would be the classic lookahead bug, because the strategy
    would be judging today's spread against a mean that includes today.
    """

    y: str          # the dependent leg
    x: str          # the independent leg
    beta: float     # hedge ratio, units of x per unit of y
    alpha: float    # OLS intercept
    mean: float     # formation spread mean
    std: float      # formation spread standard deviation
    pvalue: float   # Engle-Granger p-value on the formation window
    half_life: float  # mean-reversion half-life in days, diagnostic only

    def key(self) -> str:
        return f"{self.y}~{self.x}"

    def as_dict(self) -> dict:
        return asdict(self)


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS of y on x. Returns (beta, alpha).

    beta is how many units of x it takes to offset one unit of y. Trading y
    against beta units of x is what removes the shared market direction: both
    legs rise and fall with the sector, and the difference is what is left.
    """
    xc = sm.add_constant(x.to_numpy(), has_constant="add")
    fit = sm.OLS(y.to_numpy(), xc).fit()
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    return beta, alpha


def spread(y: pd.Series, x: pd.Series, beta: float, alpha: float) -> pd.Series:
    """The residual leg: y - beta*x - alpha."""
    return y - beta * x - alpha


def half_life(sp: pd.Series) -> float:
    """Days for a deviation to decay halfway back, from an AR(1) fit.

    Regress the daily change in the spread on its own lagged level. A negative
    coefficient means the spread is pulled back toward its mean, and the size of
    that coefficient sets how fast. Reported as a diagnostic: a pair whose
    half-life is longer than the trading window cannot revert inside it.
    """
    s = sp.dropna()
    if len(s) < 10:
        return float("nan")
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag, delta = lag.align(delta, join="inner")
    xc = sm.add_constant(lag.to_numpy(), has_constant="add")
    k = float(sm.OLS(delta.to_numpy(), xc).fit().params[1])
    if k >= 0:
        return float("inf")  # diverging, not reverting
    return float(-np.log(2) / k)


def zscore(sp: pd.Series, mean: float, std: float) -> pd.Series:
    """Standardise a spread against FIXED formation parameters.

    Deliberately takes mean and std as arguments rather than computing them from
    `sp`. A rolling or full-sample z-score computed on the trading window would
    let the strategy see the future, and this signature makes that impossible to
    do by accident.
    """
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=sp.index)
    return (sp - mean) / std


def fit_pair(
    formation: pd.DataFrame,
    y: str,
    x: str,
    pvalue: float = float("nan"),
    min_spread_std: float = 1e-6,
) -> PairParams | None:
    """Estimate every parameter a pair carries into its trading window.

    Only ever called with a formation slice. Returns None when the spread is too
    flat to standardise, which would otherwise produce z-scores in the hundreds
    from rounding noise.
    """
    ys, xs = formation[y], formation[x]
    beta, alpha = hedge_ratio(ys, xs)
    sp = spread(ys, xs, beta, alpha)
    sd = float(sp.std(ddof=1))
    if not np.isfinite(sd) or sd < min_spread_std:
        return None
    return PairParams(
        y=y, x=x, beta=beta, alpha=alpha,
        mean=float(sp.mean()), std=sd,
        pvalue=float(pvalue), half_life=half_life(sp),
    )


# ----------------------------------------------------------------------
# Cointegration: the gate
# ----------------------------------------------------------------------
#
# Correlation and cointegration answer different questions, and using the first
# where the second belongs is the most common way this strategy fails.
#
# Correlation asks whether two series move together day to day. Two stocks can
# be highly correlated and still drift apart forever: if both rise, but one rises
# faster, every daily move agrees in direction while the gap between them grows
# without limit. A spread built on that pair has no mean to revert to, so a
# z-score computed against a historical mean is measuring a distance from a
# number that is no longer meaningful.
#
# Cointegration asks something stricter: is there a fixed combination of the two
# whose value is stationary - wandering around a constant mean with finite
# variance rather than drifting? If y - beta*x is stationary, then when it is far
# from its mean there is a statistical reason to expect it to come back, which is
# precisely the bet the strategy places. Correlation says the two walk in step;
# cointegration says they are tied together by a rope of fixed length.
#
# Engle-Granger tests this in two steps: regress y on x, then test the residual
# for a unit root with an augmented Dickey-Fuller test. The null hypothesis is
# NO cointegration, so a small p-value is evidence the residual is stationary.

def engle_granger(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger cointegration p-value. Small means cointegrated."""
    from statsmodels.tsa.stattools import coint

    try:
        _, pvalue, _ = coint(y.to_numpy(float), x.to_numpy(float),
                             trend="c", autolag="aic")
        return float(pvalue)
    except Exception:
        return float("nan")


def select_pairs(
    formation: pd.DataFrame,
    pvalue_threshold: float,
    max_pairs: int,
    min_observations: int,
    min_spread_std: float,
) -> list[PairParams]:
    """Test every candidate pair on the formation window and keep the best.

    Ordering matters here. The pair is oriented so that the leg with the larger
    average price is the dependent one, which keeps the hedge ratio near unity
    and avoids the numerically awkward case of regressing a 50-rupee stock on a
    1,900-rupee one. Engle-Granger is not symmetric - testing y on x and x on y
    gives different p-values - so a fixed rule is better than an arbitrary one.

    Both the test and the fit see only `formation`.
    """
    cols = list(formation.columns)
    if len(formation) < min_observations:
        return []

    means = formation.mean()
    out: list[PairParams] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            y, x = (a, b) if means[a] >= means[b] else (b, a)
            p = engle_granger(formation[y], formation[x])
            if not np.isfinite(p) or p > pvalue_threshold:
                continue
            params = fit_pair(formation, y, x, pvalue=p,
                              min_spread_std=min_spread_std)
            if params is not None:
                out.append(params)

    # Strongest evidence of cointegration first.
    out.sort(key=lambda pp: pp.pvalue)
    return out[:max_pairs]
