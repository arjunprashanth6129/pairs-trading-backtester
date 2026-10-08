"""One entrypoint: data -> selection -> backtest -> evaluation -> artifacts.

`python -m pairs_backtester.run` regenerates everything in results/ and
docs/figures/, plus a manifest recording the config and commit that produced it.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config, data, export_site, manifest, metrics, plots
from .backtest import combine, pick_example, run_pair
from .pairs import select_pairs


def main(argv: list[str] | None = None) -> int:
    cfg = config.load()
    np.random.seed(cfg["run"]["seed"])
    results, figures = cfg.path("results_dir"), cfg.path("figures_dir")
    s, b, sel = cfg["signals"], cfg["backtest"], cfg["selection"]

    # ---------------------------------------------------------------- data
    print("1. data")
    raw = data.fetch_prices(cfg.tickers + [cfg.benchmark],
                            cfg["universe"]["start"], cfg["universe"]["end"],
                            cfg.path("cache_dir"))
    bench_px = raw[cfg.benchmark].dropna()
    prices = data.align(raw.drop(columns=[cfg.benchmark]))
    print(f"  {prices.shape[1]} tickers x {len(prices)} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    w = cfg["windows"]
    windows = data.make_windows(prices.index, w["formation_months"],
                               w["trading_months"], w["step_months"])
    print(f"  {len(windows)} windows: {w['formation_months']}m formation -> "
          f"{w['trading_months']}m trading, stepping {w['step_months']}m")

    # ------------------------------------------------------ select and trade
    print("\n2. selection and backtest")
    frames, all_trades, chosen, examples = [], [], [], []
    for win in windows:
        f, t = data.slice_formation(prices, win), data.slice_trading(prices, win)
        picked = select_pairs(f, sel["pvalue_threshold"], sel["max_pairs"],
                              sel["min_observations"], sel["min_spread_std"])
        for pp in picked:
            chosen.append({**win.as_dict(), "pair": pp.key(), **pp.as_dict()})

        res = [run_pair(t, pp, win.index, s["entry_z"], s["exit_z"],
                        s["stop_z"], s["max_holding_days"],
                        cfg["costs"]["bps_per_leg"], b["capital_per_pair"])
               for pp in picked]
        frames.append(combine(res, t.index))
        all_trades += [r.trades for r in res if len(r.trades)]
        examples += res

    picks = pd.DataFrame(chosen)
    port = pd.concat(frames).sort_index()
    trades = (pd.concat(all_trades).sort_values("entry_date")
              if all_trades else pd.DataFrame())
    n_pairs_avg = float(picks.groupby("trading_start").size().mean()) if len(picks) else 1.0
    print(f"  {len(picks)} pair-windows selected "
          f"({n_pairs_avg:.1f} per window), {len(trades)} trades")
    print(f"  median p-value {picks['pvalue'].median():.4f}, median half-life "
          f"{picks['half_life'].replace([np.inf], np.nan).median():.1f}d")

    # ---------------------------------------------------------- evaluation
    print("\n3. evaluation")
    bench_ret = bench_px.pct_change().reindex(port.index).fillna(0.0)
    # Buy and hold the traded universe, equally weighted, over the same days.
    uni_ret = prices.pct_change().reindex(port.index).mean(axis=1).fillna(0.0)

    summary = {
        "gross": metrics.summarise(port["ret_gross"], b["trading_days_per_year"],
                                   b["risk_free_annual"]),
        "net": metrics.summarise(port["ret_net"], b["trading_days_per_year"],
                                 b["risk_free_annual"]),
        "benchmark_universe": metrics.summarise(
            uni_ret, b["trading_days_per_year"], b["risk_free_annual"]),
        "benchmark_nifty": metrics.summarise(
            bench_ret, b["trading_days_per_year"], b["risk_free_annual"]),
        "trades": metrics.trade_stats(trades),
        "neutrality_net": metrics.market_neutrality(port["ret_net"], bench_ret),
        "neutrality_gross": metrics.market_neutrality(port["ret_gross"], bench_ret),
        "selection": {
            "pair_windows": int(len(picks)),
            "avg_pairs_per_window": n_pairs_avg,
            "median_pvalue": float(picks["pvalue"].median()) if len(picks) else None,
            "median_half_life_days": float(
                picks["half_life"].replace([np.inf], np.nan).median()) if len(picks) else None,
            # A pair is two banks. Its orientation (which leg is y) can change between
            # windows, so count unordered pairs rather than the oriented labels.
            "unique_pairs": int(pd.Series([tuple(sorted(yx)) for yx in zip(picks["y"], picks["x"])])
                                .nunique()) if len(picks) else 0,
        },
    }

    for name in ("gross", "net"):
        m = summary[name]
        print(f"  {name:6} total {m['total_return_%']:8.2f}%  ann "
              f"{m['ann_return_%']:6.2f}%  vol {m['ann_vol_%']:5.2f}%  "
              f"sharpe {m['sharpe']:6.2f}  maxDD {m['max_drawdown_%']:7.2f}%")
    ts, mn = summary["trades"], summary["neutrality_net"]
    print(f"  {ts['n_trades']} trades, win rate {ts['win_rate_%']:.1f}%, "
          f"avg hold {ts['avg_holding_days']:.1f}d")
    print(f"  beta to market {mn['beta_to_market']:+.4f}, "
          f"corr {mn['corr_to_market']:+.4f}, R2 {mn['r_squared']:.4f}")

    sens = metrics.cost_sensitivity(
        port["ret_gross"], port["cost_1bp"],
        [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30], b["trading_days_per_year"],
        b["risk_free_annual"])
    if len(sens):
        pos = sens[sens["ann_return_%"] > 0]
        be = pos["bps_per_leg"].max() if len(pos) else 0.0
        print(f"  break-even cost is between {be:.1f} and "
              f"{sens[sens['ann_return_%'] <= 0]['bps_per_leg'].min():.1f} "
              f"bps per leg (assumed {cfg['costs']['bps_per_leg']:.0f})")

    # ------------------------------------------------------------ artifacts
    print("\n4. artifacts")
    picks.to_csv(results / "selected_pairs.csv", index=False)
    trades.to_csv(results / "trades.csv", index=False)
    port.to_csv(results / "daily_returns.csv")
    sens.to_csv(results / "cost_sensitivity.csv", index=False)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  results/ selected_pairs.csv trades.csv daily_returns.csv "
          f"cost_sensitivity.csv summary.json")

    plots.use_style()
    ex = pick_example(examples)
    if ex is not None:
        plots.plot_pair_example(ex, s["entry_z"], s["exit_z"], s["stop_z"],
                                figures / "pair-example.png")
    if len(trades):
        plots.plot_exits(trades, figures / "pnl-by-exit.png")
    plots.plot_equity(port, figures / "equity.png")
    plots.plot_annual(port, figures / "annual.png")
    if len(sens):
        plots.plot_cost_sensitivity(sens, cfg["costs"]["bps_per_leg"],
                                    len(trades), figures / "cost-sensitivity.png")

    manifest.write(results / "manifest.json", cfg.raw, {
        "windows": [x.as_dict() for x in windows],
        "n_trades": int(len(trades)),
        "oos_days": int(len(port)),
        "headline": {"gross": summary["gross"], "net": summary["net"]},
    })
    print("  results/manifest.json")
    export_site.main()

    verdict = "yes" if summary["net"]["ann_return_%"] > 0 else "no"
    print(f"\nDid it make money out of sample after costs? {verdict}. "
          f"Was it market-neutral? beta {mn['beta_to_market']:+.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
