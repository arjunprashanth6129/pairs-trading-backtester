"""Price data: fetch, cache, align, and cut into formation/trading windows.

The window machinery lives here rather than in the backtest because the split is
a property of the timeline, not of any one strategy. Producing the windows in one
place, once, means every later stage receives slices that are disjoint by
construction instead of trusting each caller to slice correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Window:
    """One formation period and the trading period that follows it.

    `formation_end` is exclusive of `trading_start`: the trading window begins on
    the next observation, so no single bar is ever in both. That is asserted in
    tests/test_lookahead.py rather than assumed.
    """

    index: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp

    def label(self) -> str:
        return (f"{self.trading_start.date()} to {self.trading_end.date()}")

    def as_dict(self) -> dict[str, str]:
        return {
            "index": self.index,
            "formation_start": str(self.formation_start.date()),
            "formation_end": str(self.formation_end.date()),
            "trading_start": str(self.trading_start.date()),
            "trading_end": str(self.trading_end.date()),
        }


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily auto-adjusted closes, one column per ticker.

    auto_adjust=True folds splits and dividends back into the history. Without
    it a 2:1 split reads as a 50% overnight crash, which for this project would
    manufacture an enormous fake divergence in a spread and a fake trade.

    Cached to CSV so the pipeline is reproducible offline and a re-run does not
    depend on what Yahoo returns today. CSV rather than parquet on purpose: it
    costs one less binary dependency and the cache stays readable, which matters
    for a repository whose whole claim is that the numbers can be checked.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"prices_{start}_{end}_{len(tickers)}.csv"

    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise SystemExit("yfinance returned nothing - check the network.")

    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.sort_index()
    closes.index.name = "date"
    closes.to_csv(cache)
    return closes


def align(prices: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    """Drop thin columns, then keep only dates where every survivor traded.

    Two steps, in this order, on purpose. Dropping sparse tickers first means one
    late listing cannot delete years of history for everyone else; requiring a
    complete row afterwards means every pair is compared on identical dates,
    which a cointegration test needs.
    """
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(prices.columns) - set(keep))
    out = prices[keep].dropna(how="any")
    if dropped:
        print(f"  dropped {len(dropped)} thin series: {', '.join(dropped)}")
    return out


def make_windows(
    index: pd.DatetimeIndex,
    formation_months: int,
    trading_months: int,
    step_months: int,
) -> list[Window]:
    """Roll a formation->trading pair forward across the timeline.

    Successive trading windows tile the timeline without overlapping, so every
    trading day is scored exactly once and the out-of-sample record is a single
    continuous series rather than an average over re-used days.
    """
    if len(index) == 0:
        return []

    windows: list[Window] = []
    first, last = index[0], index[-1]
    f_start = first
    i = 0

    while True:
        f_end = f_start + pd.DateOffset(months=formation_months)
        t_end = f_end + pd.DateOffset(months=trading_months)
        if t_end > last + pd.Timedelta(days=1):
            break

        # Snap to actual observations. The formation window is everything
        # strictly before the first trading bar.
        f_mask = (index >= f_start) & (index < f_end)
        t_mask = (index >= f_end) & (index < t_end)
        if f_mask.sum() > 0 and t_mask.sum() > 0:
            windows.append(Window(
                index=i,
                formation_start=index[f_mask][0],
                formation_end=index[f_mask][-1],
                trading_start=index[t_mask][0],
                trading_end=index[t_mask][-1],
            ))
            i += 1
        f_start = f_start + pd.DateOffset(months=step_months)

    return windows


def slice_formation(prices: pd.DataFrame, w: Window) -> pd.DataFrame:
    return prices.loc[w.formation_start:w.formation_end]


def slice_trading(prices: pd.DataFrame, w: Window) -> pd.DataFrame:
    return prices.loc[w.trading_start:w.trading_end]
