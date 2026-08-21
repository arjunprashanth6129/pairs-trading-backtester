"""P&L accounting for a pair, a window, and the whole out-of-sample record.

Sign convention throughout: `side = +1` is long the spread (buy the y leg, sell
beta units of x), taken when z is very negative and the spread is expected to
rise. `side = -1` is the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pairs import PairParams, spread, zscore
from .signals import Trade, generate, position_series


@dataclass
class PairResult:
    params: PairParams
    window_index: int
    spread: pd.Series
    z: pd.Series
    position: pd.Series
    ret_gross: pd.Series   # daily return on capital_per_pair, before costs
    ret_net: pd.Series     # after costs
    trades: pd.DataFrame


def run_pair(
    trading: pd.DataFrame,
    params: PairParams,
    window_index: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_days: int,
    bps_per_leg: float,
    capital: float,
) -> PairResult:
    """Trade one pair across one trading window.

    Position sizing fixes GROSS notional at `capital`: one unit of the y leg plus
    |beta| units of the x leg. Sizing on gross rather than net is the honest
    choice for a long-short book, because both legs are real positions that must
    be financed and both incur cost.
    """
    y, x = trading[params.y], trading[params.x]
    sp = spread(y, x, params.beta, params.alpha)
    z = zscore(sp, params.mean, params.std)
    trades = generate(z, entry_z, exit_z, stop_z, max_holding_days)

    n = len(trading)
    pos = position_series(trades, n)
    units = np.zeros(n)
    cost = np.zeros(n)
    rate = bps_per_leg / 10_000.0
    yv, xv = y.to_numpy(float), x.to_numpy(float)
    per_unit = np.abs(yv) + abs(params.beta) * np.abs(xv)

    rows = []
    for t in trades:
        u = capital / per_unit[t.entry_i] if per_unit[t.entry_i] > 0 else 0.0
        units[t.entry_i:t.exit_i] = u
        # Both legs are charged, on entry and again on exit: four chargeable
        # legs per round trip. Pairs trading turns over often, so this is the
        # line item most likely to decide whether the strategy works.
        c_in = rate * u * per_unit[t.entry_i]
        c_out = rate * u * per_unit[t.exit_i]
        cost[t.entry_i] += c_in
        cost[t.exit_i] += c_out

        d_sp = float(sp.iloc[t.exit_i] - sp.iloc[t.entry_i])
        gross = t.side * u * d_sp
        rows.append({
            "window": window_index,
            "pair": params.key(),
            "side": "long_spread" if t.side > 0 else "short_spread",
            "entry_date": trading.index[t.entry_i],
            "exit_date": trading.index[t.exit_i],
            "holding_days": t.exit_i - t.entry_i,
            "entry_z": t.entry_z,
            "exit_z": t.exit_z,
            "reason": t.reason,
            "pnl_gross": gross,
            "cost": c_in + c_out,
            "pnl_net": gross - c_in - c_out,
        })

    # P&L over the interval t-1 -> t accrues only if the position was open at
    # t-1, so both position and size are lagged one bar against the price change.
    d = np.zeros(n)
    d[1:] = np.diff(sp.to_numpy(float))
    pnl_gross = np.zeros(n)
    pnl_gross[1:] = pos[:-1] * units[:-1] * d[1:]

    idx = trading.index
    return PairResult(
        params=params,
        window_index=window_index,
        spread=sp,
        z=z,
        position=pd.Series(pos, index=idx),
        ret_gross=pd.Series(pnl_gross / capital, index=idx),
        ret_net=pd.Series((pnl_gross - cost) / capital, index=idx),
        trades=pd.DataFrame(rows),
    )


def combine(results: list[PairResult], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Equal-weight the selected pairs within each window.

    Capital is split evenly across the pairs chosen for a window, and a pair
    holding no position contributes zero that day rather than being dropped from
    the average. Dropping it would quietly raise the portfolio return on exactly
    the days the strategy had no conviction.
    """
    if not results:
        return pd.DataFrame(
            {"ret_gross": 0.0, "ret_net": 0.0}, index=index).astype(float)

    g = pd.concat([r.ret_gross.rename(i) for i, r in enumerate(results)], axis=1)
    n = pd.concat([r.ret_net.rename(i) for i, r in enumerate(results)], axis=1)
    out = pd.DataFrame({
        "ret_gross": g.reindex(index).fillna(0.0).mean(axis=1),
        "ret_net": n.reindex(index).fillna(0.0).mean(axis=1),
    })
    return out.fillna(0.0)
