"""Turn a z-score path into a list of trades.

Kept separate from the P&L accounting so the entry and exit rules can be tested
against a hand-written z-score series with no prices, costs or capital involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Why the spread is traded rather than either leg on its own:
#
# Both banks move with the sector. Buying one and shorting the hedge-ratio
# multiple of the other cancels most of that shared direction, because whatever
# the sector does enters both legs and nets out. What is left is the difference
# between the two, which is the only thing the strategy has an opinion about.
# That is why the position is long one and short the other rather than a bet on
# either name, and why the resulting beta to the market should be near zero.

LONG, SHORT, FLAT = 1, -1, 0


@dataclass(frozen=True)
class Trade:
    """One round trip, in integer positions into the trading window."""

    entry_i: int
    exit_i: int
    side: int        # +1 long the spread, -1 short the spread
    entry_z: float
    exit_z: float
    reason: str      # reverted | stop | timeout | window_end


def generate(
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_days: int,
) -> list[Trade]:
    """Walk the z path once, opening and closing a single position at a time.

    Execution is deliberately lagged by one bar. The signal is decided on bar i
    using the close of bar i, and the trade is booked at the close of bar i+1,
    because acting on bar i's close requires knowing bar i's close. Same-bar
    execution is a small but real lookahead and it flatters this kind of
    strategy specifically, since entries cluster on large moves.
    """
    vals = z.to_numpy(dtype=float)
    n = len(vals)
    trades: list[Trade] = []

    pos = FLAT
    entry_i = -1
    entry_val = np.nan

    for i in range(n - 1):
        v = vals[i]
        if not np.isfinite(v):
            continue
        book = i + 1  # the bar the decision is executed on

        if pos == FLAT:
            # A spread already past the stop is not an entry. Without this check
            # a pair that breaks down is entered, stopped the next bar, entered
            # again, and so on, paying four legs of cost on every loop.
            if abs(v) >= stop_z:
                continue
            if v >= entry_z:
                pos, entry_i, entry_val = SHORT, book, v
            elif v <= -entry_z:
                pos, entry_i, entry_val = LONG, book, v
            continue

        held = book - entry_i
        reason = None
        if abs(v) <= exit_z:
            reason = "reverted"
        elif abs(v) >= stop_z:
            reason = "stop"
        elif held >= max_holding_days:
            reason = "timeout"

        if reason:
            trades.append(Trade(entry_i, book, pos, entry_val, v, reason))
            pos, entry_i, entry_val = FLAT, -1, np.nan

    # A position still open when the window ends is closed at the last bar. The
    # alternative - dropping it - would silently discard exactly the trades that
    # went worst, which is the losing half of a survivorship bias.
    if pos != FLAT and entry_i < n - 1:
        trades.append(Trade(entry_i, n - 1, pos, entry_val, float(vals[-1]),
                            "window_end"))

    return trades


def position_series(trades: list[Trade], n: int) -> np.ndarray:
    """Expand trades into a per-bar position vector, for P&L and diagnostics."""
    pos = np.zeros(n, dtype=float)
    for t in trades:
        pos[t.entry_i:t.exit_i] = t.side
    return pos
