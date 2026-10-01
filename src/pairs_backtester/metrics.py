"""Performance and market-neutrality statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity(returns: pd.Series) -> pd.Series:
    """Compound a daily return series into a growth-of-1 curve."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def summarise(
    returns: pd.Series,
    trading_days: int = 252,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    r = returns.fillna(0.0)
    eq = equity(r)
    years = len(r) / trading_days if len(r) else np.nan

    total = float(eq.iloc[-1] - 1.0) if len(eq) else np.nan
    ann = float(eq.iloc[-1] ** (1 / years) - 1.0) if years and years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(trading_days)) if len(r) > 1 else np.nan
    # Excess over the risk-free rate, so the Sharpe is comparable to the other
    # projects in this portfolio, which use the same 6% rupee rate.
    sharpe = float((ann - rf_annual) / vol) if vol and np.isfinite(vol) and vol > 0 else np.nan

    return {
        "total_return_%": total * 100,
        "ann_return_%": ann * 100,
        "ann_vol_%": vol * 100,
        "sharpe": sharpe,
        "max_drawdown_%": float(drawdown(eq).min() * 100),
        "days": int(len(r)),
    }


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "win_rate_%": np.nan,
                "avg_holding_days": np.nan, "avg_pnl_net": np.nan,
                "total_cost": 0.0}
    return {
        "n_trades": int(len(trades)),
        "win_rate_%": float((trades["pnl_net"] > 0).mean() * 100),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "avg_pnl_net": float(trades["pnl_net"].mean()),
        "total_cost": float(trades["cost"].sum()),
    }


def market_neutrality(
    returns: pd.Series, benchmark_returns: pd.Series
) -> dict[str, float]:
    """Beta and correlation against the broad market.

    This is the claim the long-short structure is supposed to deliver, so it is
    measured rather than asserted. Beta is the OLS slope of strategy returns on
    market returns: near zero means the market's direction explains almost none
    of the strategy's day-to-day movement.
    """
    a, b = returns.align(benchmark_returns, join="inner")
    a, b = a.fillna(0.0), b.fillna(0.0)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 30 or b.std() == 0:
        return {"beta_to_market": np.nan, "corr_to_market": np.nan,
                "r_squared": np.nan, "overlap_days": int(len(a))}
    beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "beta_to_market": beta,
        "corr_to_market": corr,
        "r_squared": corr ** 2,
        "overlap_days": int(len(a)),
    }


def cost_sensitivity(
    gross_returns: pd.Series,
    cost_1bp: pd.Series,
    bps_grid: list[float],
    trading_days: int = 252,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Re-price the same trades at a range of cost assumptions.

    The trades never change, because the signal does not depend on costs. Cost
    is linear in the rate, so the daily cost line at 1 bp per leg, scaled, gives
    the exact net return at any other rate. At the configured rate this
    reproduces the backtest's own net figures.
    """
    rows = []
    for bps in bps_grid:
        net = gross_returns - bps * cost_1bp
        m = summarise(net, trading_days, rf_annual)
        rows.append({"bps_per_leg": bps, **m})
    return pd.DataFrame(rows)
