"""Cointegration is the gate, so the gate has to open and shut correctly.

The distinction these tests encode: correlated-but-drifting must be REJECTED,
cointegrated must be ACCEPTED. Getting this backwards is the single most
expensive mistake available in this strategy, because it admits pairs whose
spread has no mean to revert to.
"""

import numpy as np
import pandas as pd

from pairs_backtester.pairs import engle_granger, select_pairs


def _two_random_walks(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    b = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    return pd.Series(a), pd.Series(b)


def test_independent_random_walks_are_not_cointegrated():
    a, b = _two_random_walks(seed=3)
    assert engle_granger(a, b) > 0.05


def test_cointegrated_series_is_detected():
    """b shares a's stochastic trend and adds a stationary wobble, so a fixed
    combination of the two is stationary even though each alone is not."""
    rng = np.random.default_rng(4)
    common = np.cumsum(rng.normal(0, 1.0, 1200)) + 100
    a = pd.Series(common + rng.normal(0, 0.5, 1200))
    b = pd.Series(0.5 * common + rng.normal(0, 0.5, 1200))
    assert engle_granger(a, b) < 0.05


def test_high_correlation_does_not_imply_cointegration():
    """The case the whole gate exists for.

    Both series are driven by the same shocks, so their DAILY CHANGES correlate
    near-perfectly. But one carries an extra drift, so the gap between their
    LEVELS grows without bound and no fixed combination is stationary.
    """
    rng = np.random.default_rng(5)
    shocks = rng.normal(0, 1.0, 1500)
    a = pd.Series(np.cumsum(shocks) + 100)
    b = pd.Series(np.cumsum(shocks + 0.08) + 100)   # same shocks, extra drift

    assert a.diff().corr(b.diff()) > 0.99            # correlated
    assert engle_granger(a, b) > 0.05                # but not cointegrated


def test_select_pairs_returns_only_passing_pairs_and_respects_the_cap():
    rng = np.random.default_rng(6)
    n = 900
    common = np.cumsum(rng.normal(0, 1.0, n)) + 200
    df = pd.DataFrame({
        "COINT_A": common + rng.normal(0, 0.6, n),
        "COINT_B": 0.7 * common + rng.normal(0, 0.6, n),
        "COINT_C": 1.3 * common + rng.normal(0, 0.6, n),
        "DRIFT_D": np.cumsum(rng.normal(0.15, 1.0, n)) + 200,
        "DRIFT_E": np.cumsum(rng.normal(-0.10, 1.0, n)) + 200,
    })
    out = select_pairs(df, pvalue_threshold=0.05, max_pairs=2,
                       min_observations=100, min_spread_std=1e-6)
    assert len(out) <= 2
    assert all(p.pvalue <= 0.05 for p in out)
    # Sorted strongest-evidence-first.
    assert out == sorted(out, key=lambda p: p.pvalue)


def test_select_pairs_returns_nothing_on_a_short_window():
    df = pd.DataFrame(np.random.default_rng(7).normal(100, 1, (50, 3)),
                      columns=list("ABC"))
    assert select_pairs(df, 0.05, 5, min_observations=200,
                        min_spread_std=1e-6) == []
