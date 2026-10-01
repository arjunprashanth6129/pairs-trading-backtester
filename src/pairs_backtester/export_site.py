"""Write one compact JSON for the dashboard that presents these results.

The site recomputes nothing: it renders what this pipeline committed. Series are
downsampled here rather than in the browser, because shipping 2,347 daily points
per line to render a chart a few hundred pixels wide is bytes for no pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, metrics
from .backtest import pick_example, run_pair
from .pairs import select_pairs


def _curve(s: pd.Series, every: int = 3) -> list[dict]:
    s = s.iloc[::every]
    return [{"d": str(i.date()), "v": round(float(v), 5)} for i, v in s.items()]


def build(cfg=None) -> dict:
    cfg = cfg or config.load()
    results = cfg.path("results_dir")
    s, b, sel = cfg["signals"], cfg["backtest"], cfg["selection"]

    summary = json.loads((results / "summary.json").read_text())
    picks = pd.read_csv(results / "selected_pairs.csv")
    trades = pd.read_csv(results / "trades.csv", parse_dates=["entry_date", "exit_date"])
    port = pd.read_csv(results / "daily_returns.csv", index_col=0, parse_dates=True)
    sens = pd.read_csv(results / "cost_sensitivity.csv")

    raw = data.fetch_prices(cfg.tickers + [cfg.benchmark],
                            cfg["universe"]["start"], cfg["universe"]["end"],
                            cfg.path("cache_dir"))
    prices = data.align(raw.drop(columns=[cfg.benchmark]))
    uni = prices.pct_change().reindex(port.index).mean(axis=1).fillna(0.0)

    eq_g = metrics.equity(port["ret_gross"])
    eq_n = metrics.equity(port["ret_net"])
    eq_b = metrics.equity(uni)
    dd_n = metrics.drawdown(eq_n)

    # Rebuild the example pair so the dashboard can draw the mechanism.
    w = cfg["windows"]
    windows = data.make_windows(prices.index, w["formation_months"],
                                w["trading_months"], w["step_months"])
    candidates = []
    for win in windows:
        f, t = data.slice_formation(prices, win), data.slice_trading(prices, win)
        for pp in select_pairs(f, sel["pvalue_threshold"], sel["max_pairs"],
                               sel["min_observations"], sel["min_spread_std"]):
            candidates.append(run_pair(t, pp, win.index, s["entry_z"], s["exit_z"],
                                       s["stop_z"], s["max_holding_days"],
                                       cfg["costs"]["bps_per_leg"],
                                       b["capital_per_pair"]))
    best_res = pick_example(candidates)

    example = None
    if best_res is not None:
        p = best_res.params
        example = {
            "pair": p.key(), "y": p.y, "x": p.x,
            "beta": round(p.beta, 4), "pvalue": round(p.pvalue, 5),
            "half_life": round(p.half_life, 2),
            "mean": round(p.mean, 4), "std": round(p.std, 4),
            "window": best_res.window_index,
            "series": [
                {"d": str(i.date()),
                 "spread": round(float(sp), 3),
                 "z": round(float(z), 3)}
                for i, sp, z in zip(best_res.z.index, best_res.spread, best_res.z)
                if np.isfinite(z)
            ],
            "trades": [
                {"entry": str(r["entry_date"].date()), "exit": str(r["exit_date"].date()),
                 "side": r["side"], "entry_z": round(r["entry_z"], 3),
                 "exit_z": round(r["exit_z"], 3), "reason": r["reason"],
                 "pnl_net": round(r["pnl_net"], 2)}
                for _, r in best_res.trades.iterrows()
            ],
        }

    by_reason = trades["reason"].value_counts().to_dict() if len(trades) else {}

    # Where the gross P&L came from, by how each trade ended. This is the table
    # the result turns on: reversions earn, stops give it back.
    exits = []
    for reason in ("reverted", "stop", "timeout", "window_end"):
        g = trades[trades["reason"] == reason]
        if len(g):
            exits.append({
                "reason": reason, "n": int(len(g)),
                "gross": round(float(g["pnl_gross"].sum()), 0),
                "cost": round(float(g["cost"].sum()), 0),
                "net": round(float(g["pnl_net"].sum()), 0),
                "median_hold": float(g["holding_days"].median()),
            })

    # Calendar-year returns, compounded from the daily series.
    annual = []
    for year, g in port.groupby(port.index.year):
        annual.append({
            "year": int(year),
            "days": int(len(g)),
            "gross": round(float(((1 + g["ret_gross"]).prod() - 1) * 100), 2),
            "net": round(float(((1 + g["ret_net"]).prod() - 1) * 100), 2),
        })

    # Did a stronger cointegration test pick better pairs?
    joined = trades.merge(picks.rename(columns={"index": "window"})
                          [["window", "pair", "pvalue", "half_life"]],
                          on=["window", "pair"], how="left")
    by_pvalue = []
    for lo, hi in ((0.0, 0.01), (0.01, 0.03), (0.03, sel["pvalue_threshold"])):
        g = joined[(joined["pvalue"] > lo) & (joined["pvalue"] <= hi)]
        if len(g):
            by_pvalue.append({
                "lo": lo, "hi": hi, "n": int(len(g)),
                "stop_rate": round(float((g["reason"] == "stop").mean()), 4),
                "reverted_rate": round(float((g["reason"] == "reverted").mean()), 4),
                "avg_gross": round(float(g["pnl_gross"].mean()), 1),
            })

    rev = joined[joined["reason"] == "reverted"]
    reversion = {
        "n": int(len(rev)),
        "median_hold": float(rev["holding_days"].median()) if len(rev) else None,
        "median_half_life": round(float(rev["half_life"].median()), 2) if len(rev) else None,
        "median_hold_over_half_life": round(
            float((rev["holding_days"] / rev["half_life"]).median()), 2) if len(rev) else None,
        "median_entry_abs_z": round(float(rev["entry_z"].abs().median()), 3) if len(rev) else None,
    }
    top_pairs = (picks.groupby("pair")
                 .agg(times_selected=("pair", "size"),
                      median_pvalue=("pvalue", "median"),
                      median_half_life=("half_life", "median"))
                 .sort_values("times_selected", ascending=False)
                 .head(10).reset_index())

    return {
        "meta": {
            "universe": cfg.tickers,
            "traded": list(prices.columns),
            "benchmark": cfg.benchmark,
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "oos_start": str(port.index[0].date()),
            "oos_end": str(port.index[-1].date()),
            "n_windows": len(windows),
            "formation_months": w["formation_months"],
            "trading_months": w["trading_months"],
            "entry_z": s["entry_z"], "exit_z": s["exit_z"], "stop_z": s["stop_z"],
            "max_holding_days": s["max_holding_days"],
            "bps_per_leg": cfg["costs"]["bps_per_leg"],
            "pvalue_threshold": sel["pvalue_threshold"],
            "max_pairs": sel["max_pairs"],
            "candidate_pairs": len(prices.columns) * (len(prices.columns) - 1) // 2,
        },
        "summary": summary,
        "equity": {
            "gross": _curve(eq_g), "net": _curve(eq_n), "benchmark": _curve(eq_b),
            "drawdown_net": _curve(dd_n),
        },
        "cost_sensitivity": sens.round(4).to_dict(orient="records"),
        "trades_by_reason": by_reason,
        "exits": exits,
        "annual": annual,
        "by_pvalue": by_pvalue,
        "reversion": reversion,
        "capital_per_pair": b["capital_per_pair"],
        "top_pairs": top_pairs.round(4).to_dict(orient="records"),
        "example": example,
    }


def main() -> int:
    cfg = config.load()
    payload = build(cfg)
    out = cfg.path("results_dir") / "dashboard.json"
    out.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print(f"wrote {out.relative_to(config.ROOT)} "
          f"({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
