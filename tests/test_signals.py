"""Entry, exit, stop and timeout rules, tested on hand-written z paths so no
prices, costs or capital can obscure what the rule did.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_backtester.pairs import zscore
from pairs_backtester.signals import generate, position_series

E, X, S, H = 2.0, 0.5, 4.0, 60


def z(*vals):
    return pd.Series(list(vals), dtype=float)


def test_no_trade_when_the_spread_never_stretches():
    assert generate(z(0, 0.5, -0.9, 1.2, -1.5, 0.3), E, X, S, H) == []


def test_short_the_spread_when_z_is_high():
    t = generate(z(0, 2.4, 1.0, 0.1, 0), E, X, S, H)
    assert len(t) == 1 and t[0].side == -1 and t[0].reason == "reverted"


def test_long_the_spread_when_z_is_low():
    t = generate(z(0, -2.4, -1.0, -0.1, 0), E, X, S, H)
    assert len(t) == 1 and t[0].side == 1 and t[0].reason == "reverted"


def test_execution_lags_the_signal_by_one_bar():
    """The threshold is crossed at index 1; the fill must book at index 2.

    Filling at index 1 would mean acting on a close before observing it, which
    is a lookahead that specifically flatters this strategy because entries
    cluster on large moves.
    """
    t = generate(z(0, 2.5, 1.0, 0.0), E, X, S, H)
    assert t[0].entry_i == 2


def test_stop_loss_fires_when_the_spread_keeps_diverging():
    t = generate(z(0, 2.2, 3.0, 4.5, 5.0), E, X, S, H)
    assert len(t) == 1 and t[0].reason == "stop"


def test_timeout_closes_a_trade_that_never_reverts():
    path = [0, 2.5] + [1.5] * 30          # entered, drifts, never reverts
    t = generate(z(*path), E, X, S, max_holding_days=10)
    assert len(t) == 1 and t[0].reason == "timeout"
    assert t[0].exit_i - t[0].entry_i == 10


def test_open_position_is_closed_at_the_window_end():
    """Dropping an unclosed trade would discard exactly the ones that went
    worst, which is the losing half of a survivorship bias."""
    t = generate(z(0, 2.5, 2.4, 2.3, 2.2), E, X, S, H)
    assert len(t) == 1 and t[0].reason == "window_end"


def test_only_one_position_is_open_at_a_time():
    t = generate(z(0, 2.5, 3.0, 2.8, 0.1, 2.6, 0.0), E, X, S, H)
    for a, b in zip(t, t[1:]):
        assert a.exit_i <= b.entry_i


def test_position_series_matches_the_trades():
    trades = generate(z(0, 2.5, 1.0, 0.0, -2.6, -1.0, 0.0), E, X, S, H)
    pos = position_series(trades, 7)
    for tr in trades:
        assert set(np.unique(pos[tr.entry_i:tr.exit_i])) == {float(tr.side)}


def test_nan_z_values_are_skipped_not_traded():
    assert generate(z(np.nan, np.nan, 0.1, 0.2), E, X, S, H) == []


def test_zscore_uses_the_parameters_it_is_given():
    sp = pd.Series([10.0, 11.0, 12.0])
    out = zscore(sp, mean=10.0, std=2.0)
    assert list(out) == pytest.approx([0.0, 0.5, 1.0])


def test_zscore_is_undefined_when_the_spread_is_flat():
    assert zscore(pd.Series([5.0, 5.0]), mean=5.0, std=0.0).isna().all()
