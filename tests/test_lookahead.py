"""The no-lookahead guard.

Every test here is designed to FAIL if information from the trading window
reaches the parameters that trade it. This is the property that decides whether
any other number in the project means anything, so it is tested from four
independent angles rather than asserted once.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_backtester import data
from pairs_backtester.backtest import run_pair
from pairs_backtester.pairs import fit_pair, select_pairs, spread, zscore


@pytest.fixture
def frame():
    rng = np.random.default_rng(11)
    n = 1500
    idx = pd.bdate_range("2016-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 1.0, n)) + 300
    return pd.DataFrame({
        "AAA": common + rng.normal(0, 1.0, n),
        "BBB": 0.8 * common + rng.normal(0, 1.0, n),
        "CCC": np.cumsum(rng.normal(0.05, 1.0, n)) + 300,
    }, index=idx)


# ---------------------------------------------------------------- windows
def test_windows_never_share_a_single_bar(frame):
    wins = data.make_windows(frame.index, 12, 6, 6)
    assert len(wins) > 1
    for w in wins:
        f = data.slice_formation(frame, w)
        t = data.slice_trading(frame, w)
        assert len(f.index.intersection(t.index)) == 0
        assert f.index.max() < t.index.min()


def test_trading_windows_do_not_overlap_each_other(frame):
    """Each out-of-sample day is scored once, so the record is a single
    continuous series rather than an average over re-used days."""
    wins = data.make_windows(frame.index, 12, 6, 6)
    for a, b in zip(wins, wins[1:]):
        assert a.trading_end < b.trading_start


# ---------------------------------------------------------------- parameters
def test_fitted_parameters_are_blind_to_the_trading_window(frame):
    """Corrupt the future and the fitted parameters must not move.

    This is the strongest available statement of the property: if any trading
    data reached fit_pair, replacing it with noise would change the answer.
    """
    wins = data.make_windows(frame.index, 12, 6, 6)
    w = wins[0]

    before = fit_pair(data.slice_formation(frame, w), "AAA", "BBB")

    corrupted = frame.copy()
    mask = corrupted.index >= w.trading_start
    rng = np.random.default_rng(99)
    corrupted.loc[mask, :] = rng.normal(1e5, 1e4, corrupted.loc[mask, :].shape)

    after = fit_pair(data.slice_formation(corrupted, w), "AAA", "BBB")

    assert before is not None and after is not None
    assert before.beta == pytest.approx(after.beta, rel=1e-12)
    assert before.alpha == pytest.approx(after.alpha, rel=1e-12)
    assert before.mean == pytest.approx(after.mean, rel=1e-12)
    assert before.std == pytest.approx(after.std, rel=1e-12)


def test_pair_selection_is_blind_to_the_trading_window(frame):
    """Selection, not just fitting, must be decided before the future exists."""
    wins = data.make_windows(frame.index, 12, 6, 6)
    w = wins[0]

    picked = select_pairs(data.slice_formation(frame, w), 0.10, 5, 100, 1e-6)

    corrupted = frame.copy()
    mask = corrupted.index >= w.trading_start
    corrupted.loc[mask, :] = np.random.default_rng(7).normal(
        5e4, 5e3, corrupted.loc[mask, :].shape)
    picked_again = select_pairs(data.slice_formation(corrupted, w),
                                0.10, 5, 100, 1e-6)

    assert [p.key() for p in picked] == [p.key() for p in picked_again]
    assert [round(p.pvalue, 12) for p in picked] == \
           [round(p.pvalue, 12) for p in picked_again]


# ---------------------------------------------------------------- z-score
def test_trading_zscore_is_not_recentred_on_trading_data(frame):
    """The tell-tale signature of the classic bug.

    A z-score computed against the trading window's own mean is centred on zero
    by construction. Computed against fixed formation parameters it is not, and
    that offset is the strategy honestly not knowing where the spread has moved.
    """
    wins = data.make_windows(frame.index, 12, 6, 6)
    w = wins[0]
    p = fit_pair(data.slice_formation(frame, w), "AAA", "BBB")
    t = data.slice_trading(frame, w)

    sp = spread(t["AAA"], t["BBB"], p.beta, p.alpha)
    z_correct = zscore(sp, p.mean, p.std)
    z_leaked = (sp - sp.mean()) / sp.std(ddof=1)

    assert abs(float(z_leaked.mean())) < 1e-9        # leaked version: centred
    assert abs(float(z_correct.mean())) > 1e-6       # correct version: is not
    assert not np.allclose(z_correct, z_leaked)


def test_zscore_signature_rejects_a_deliberately_leaked_parameter_set(frame):
    """Refit the parameters on the trading window - the bug - and show the
    guard above catches it."""
    wins = data.make_windows(frame.index, 12, 6, 6)
    w = wins[0]
    t = data.slice_trading(frame, w)

    leaked = fit_pair(t, "AAA", "BBB")               # fitted on the FUTURE
    sp = spread(t["AAA"], t["BBB"], leaked.beta, leaked.alpha)
    z = zscore(sp, leaked.mean, leaked.std)

    assert abs(float(z.mean())) < 1e-9, (
        "a z-score fitted on its own window is centred; if this ever fails the "
        "leak detector itself is broken")


# ---------------------------------------------------------------- execution
def test_pnl_uses_no_future_price(frame):
    """Truncating the data after a date must not change P&L before that date."""
    wins = data.make_windows(frame.index, 12, 6, 6)
    w = wins[0]
    p = fit_pair(data.slice_formation(frame, w), "AAA", "BBB")
    t = data.slice_trading(frame, w)

    full = run_pair(t, p, 0, 2.0, 0.5, 4.0, 60, 20.0, 100_000.0)
    cut_at = len(t) // 2
    part = run_pair(t.iloc[:cut_at], p, 0, 2.0, 0.5, 4.0, 60, 20.0, 100_000.0)

    # Compare the overlapping stretch, excluding the truncated run's final bar,
    # where the forced window-end close legitimately differs.
    a = full.ret_net.iloc[:cut_at - 1]
    b = part.ret_net.iloc[:cut_at - 1]
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)
