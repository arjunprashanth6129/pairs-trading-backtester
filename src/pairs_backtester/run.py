"""One entrypoint: data -> pair selection -> backtest -> report."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, data, metrics
from .backtest import combine, run_pair
from .pairs import select_pairs


def main(argv: list[str] | None = None) -> int:
    cfg = config.load()
    np.random.seed(cfg["run"]["seed"])
    cache = cfg.path("cache_dir")

    print("1. data")
    prices = data.fetch_prices(
        cfg.tickers + [cfg.benchmark],
        cfg["universe"]["start"], cfg["universe"]["end"], cache)
    bench = prices[cfg.benchmark].dropna()
    prices = data.align(prices.drop(columns=[cfg.benchmark]))
    print(f"  {prices.shape[1]} tickers x {len(prices)} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    w = cfg["windows"]
    windows = data.make_windows(
        prices.index, w["formation_months"], w["trading_months"],
        w["step_months"])
    print(f"  {len(windows)} formation->trading windows")

    print("\n2. pair selection (cointegration, formation windows only)")
    s, b, sel_cfg = cfg["signals"], cfg["backtest"], cfg["selection"]
    frames, all_trades, chosen = [], [], []

    for win in windows:
        f = data.slice_formation(prices, win)
        t = data.slice_trading(prices, win)
        selected = select_pairs(
            f, sel_cfg["pvalue_threshold"], sel_cfg["max_pairs"],
            sel_cfg["min_observations"], sel_cfg["min_spread_std"])
        for pp in selected:
            chosen.append({"window": win.index, **win.as_dict(),
                           "pair": pp.key(), **pp.as_dict()})

        results = [
            run_pair(t, pp, win.index, s["entry_z"], s["exit_z"], s["stop_z"],
                     s["max_holding_days"], cfg["costs"]["bps_per_leg"],
                     b["capital_per_pair"])
            for pp in selected
        ]
        frames.append(combine(results, t.index))
        all_trades += [r.trades for r in results if len(r.trades)]

    picks = pd.DataFrame(chosen)
    n_per = picks.groupby("window").size() if len(picks) else pd.Series(dtype=int)
    print(f"  {len(picks)} pair-windows selected, "
          f"{n_per.mean():.1f} per window on average")
    if len(picks):
        print(f"  median p-value {picks['pvalue'].median():.4f}  "
              f"median half-life {picks['half_life'].replace([np.inf], np.nan).median():.1f}d")

    print("\n3. backtest")
    port = pd.concat(frames).sort_index()
    trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    print(f"  {len(port)} out-of-sample days, {len(trades)} trades")

    print("\n4. report")
    for name, col in (("before costs", "ret_gross"), ("after costs", "ret_net")):
        m = metrics.summarise(port[col], b["trading_days_per_year"],
                              b["risk_free_annual"])
        print(f"  {name:12} total {m['total_return_%']:7.2f}%  "
              f"ann {m['ann_return_%']:6.2f}%  vol {m['ann_vol_%']:5.2f}%  "
              f"sharpe {m['sharpe']:6.2f}  maxDD {m['max_drawdown_%']:6.2f}%")
    ts = metrics.trade_stats(trades)
    print(f"  trades {ts['n_trades']}  win rate {ts['win_rate_%']:.1f}%  "
          f"avg hold {ts['avg_holding_days']:.1f}d")
    mn = metrics.market_neutrality(port["ret_net"], bench.pct_change())
    print(f"  beta to market {mn['beta_to_market']:+.4f}  "
          f"corr {mn['corr_to_market']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
