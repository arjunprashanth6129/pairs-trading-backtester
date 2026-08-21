"""The hedge ratio has to recover a known relationship, or nothing downstream
means anything: it sets the size of the second leg and therefore whether the
market exposure actually cancels.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_backtester.pairs import fit_pair, half_life, hedge_ratio, spread


def test_recovers_known_beta_and_alpha():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 500)) + 100)
    y = 3.0 + 2.5 * x + rng.normal(0, 0.01, 500)
    beta, alpha = hedge_ratio(y, x)
    assert beta == pytest.approx(2.5, abs=1e-3)
    assert alpha == pytest.approx(3.0, abs=1e-2)


def test_spread_of_a_perfect_relationship_is_flat():
    x = pd.Series(np.linspace(100, 200, 300))
    y = 5.0 + 1.5 * x
    beta, alpha = hedge_ratio(y, x)
    sp = spread(y, x, beta, alpha)
    assert sp.std(ddof=1) == pytest.approx(0.0, abs=1e-6)


def test_hedge_ratio_is_not_symmetric():
    """beta(y~x) is 1/beta(x~y) only without noise. With noise OLS attenuates,
    which is why the pair is oriented by a fixed rule rather than arbitrarily.
    """
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50)
    y = 2.0 * x + rng.normal(0, 5.0, 400)
    b1, _ = hedge_ratio(y, x)
    b2, _ = hedge_ratio(x, y)
    assert not np.isclose(b1, 1.0 / b2, rtol=0.02)


def test_half_life_detects_fast_and_slow_reversion():
    rng = np.random.default_rng(2)

    def ou(k, n=4000):
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] - k * s[i - 1] + rng.normal(0, 0.1)
        return pd.Series(s)

    fast, slow = half_life(ou(0.20)), half_life(ou(0.02))
    assert 0 < fast < slow
    assert fast == pytest.approx(np.log(2) / 0.20, rel=0.35)


def test_fit_pair_rejects_a_degenerate_spread():
    x = pd.Series(np.linspace(10, 20, 300))
    df = pd.DataFrame({"A": 2 * x, "B": x})   # exactly collinear
    assert fit_pair(df, "A", "B", min_spread_std=1e-6) is None
