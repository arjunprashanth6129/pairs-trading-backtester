"""Figures for the report, on the same palette as the site that presents them.

Blue and amber carry the polarity in every chart (gain and loss, before and
after costs). Green against red separates by a colour distance of about 3.5
under deuteranopia; this pair separates by about 31. Everything else is a
neutral, and every text label is in ink rather than in a series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Mirrors the site's design tokens.
PAPER = "#ffffff"
INK = "#0f172a"
INK_MUTED = "#475569"
INK_DIM = "#64748b"
LINE = "#e2e8f0"
LINE_STRONG = "#cbd5e1"
BLUE = "#2563eb"
AMBER = "#b45309"
SLATE = "#64748b"
SLATE_DARK = "#334155"

DPI = 200
WIDTH = 10.0


def use_style() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "lines.solid_capstyle": "round",
        "axes.unicode_minus": False,
    })


def style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(PAPER)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE_STRONG)
    ax.tick_params(colors=INK_DIM, length=0, pad=6)


def heading(fig, title: str, subtitle: str | None = None) -> None:
    """Title and one-line subtitle, left-aligned to the figure edge."""
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=13,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.0, 0.985, subtitle, ha="left", va="top", fontsize=10,
                 color=INK_MUTED)


def legend_row(fig, handles, y: float = 0.935) -> None:
    lg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, y),
                    ncol=len(handles), frameon=False, handlelength=1.6,
                    columnspacing=1.6, borderaxespad=0)
    for t in lg.get_texts():
        t.set_color(INK_MUTED)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def _short(ticker: str) -> str:
    return ticker.replace(".NS", "")


# ----------------------------------------------------------------------
def plot_pair_example(res, entry_z: float, exit_z: float, stop_z: float,
                      out: Path) -> None:
    """One pair's z-score through one trading window, with every trade shaded."""
    p = res.params
    z = res.z
    fig, ax = plt.subplots(figsize=(WIDTH, 4.9))
    fig.subplots_adjust(top=0.84)
    style(ax, "none")

    for _, tr in res.trades.iterrows():
        colour = BLUE if tr["pnl_net"] > 0 else AMBER
        ax.axvspan(tr["entry_date"], tr["exit_date"], color=colour, alpha=0.1,
                   linewidth=0, zorder=1)

    for lv, ls, lw, col in ((stop_z, "-", 1.0, SLATE_DARK),
                            (entry_z, "--", 1.0, SLATE),
                            (exit_z, ":", 1.3, SLATE)):
        for sign in (1, -1):
            ax.axhline(sign * lv, color=col, linestyle=ls, linewidth=lw, zorder=2)
    ax.axhline(0, color=LINE_STRONG, linewidth=1, zorder=2)

    ax.plot(z.index, z.to_numpy(), color=INK, linewidth=1.5, zorder=4)

    if len(res.trades):
        t = res.trades
        ax.scatter(t["entry_date"], z.loc[t["entry_date"]], s=48, marker="o",
                   facecolor=PAPER, edgecolor=INK, linewidth=1.6, zorder=5)
        ax.scatter(t["exit_date"], z.loc[t["exit_date"]], s=48, marker="o",
                   color=INK, edgecolor=PAPER, linewidth=1.5, zorder=5)

    # Direct labels for the bands, in the right margin.
    for lv, name in ((stop_z, "stop"), (entry_z, "entry"), (exit_z, "exit")):
        for sign in (1, -1):
            ax.annotate(f"{name} {sign * lv:+g}", xy=(1.0, sign * lv),
                        xycoords=("axes fraction", "data"), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_MUTED, annotation_clip=False)

    lim = max(stop_z + 0.6, float(np.nanmax(np.abs(z.to_numpy()))) + 0.4)
    ax.set_ylim(-lim, lim)
    ax.set_yticks(np.arange(-stop_z, stop_z + 1, 2))
    ax.set_ylabel("z-score")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    pad = pd.Timedelta(days=3)
    ax.set_xlim(z.index[0] - pad, z.index[-1] + pad)

    n_rev = int((res.trades["reason"] == "reverted").sum())
    n_stop = int((res.trades["reason"] == "stop").sum())
    heading(fig, f"{_short(p.y)} against {p.beta:.2f} x {_short(p.x)}, "
                 f"{z.index[0]:%b %Y} to {z.index[-1]:%b %Y}",
            f"{len(res.trades)} trades: {n_rev} reverted, {n_stop} stopped out. "
            f"Every band was fixed on the previous 12 months "
            f"(p = {p.pvalue:.3f}, half-life {p.half_life:.1f} days).")
    legend_row(fig, [
        Patch(color=BLUE, alpha=0.25, label="trade closed in profit"),
        Patch(color=AMBER, alpha=0.3, label="trade closed at a loss"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markeredgewidth=1.6, label="entry"),
        Line2D([], [], marker="o", linestyle="none", color=INK, label="exit"),
    ])
    save(fig, out)


def plot_exits(trades: pd.DataFrame, out: Path) -> None:
    """Gross P&L by how each trade ended, then costs, as a waterfall."""
    labels = {"reverted": "Reverted", "stop": "Stopped out",
              "timeout": "Timed out", "window_end": "Window ended"}
    steps = []
    for r in ("reverted", "stop", "timeout", "window_end"):
        g = trades[trades["reason"] == r]
        if len(g):
            steps.append((f"{labels[r]}\n{len(g)} trades",
                          g["pnl_gross"].sum() / 1000, False))
    gross = trades["pnl_gross"].sum() / 1000
    cost = trades["cost"].sum() / 1000
    steps.append(("Gross P&L", gross, True))
    steps.append((f"Costs\n{len(trades)} round trips", -cost, False))
    steps.append(("Net P&L", gross - cost, True))

    fig, ax = plt.subplots(figsize=(WIDTH, 4.9))
    fig.subplots_adjust(top=0.86)
    style(ax)
    level, lows, highs, w = 0.0, [], [], 0.56
    for i, (_, v, total) in enumerate(steps):
        lo, hi = (0.0, v) if total else (level, level + v)
        colour = SLATE if total else (BLUE if v >= 0 else AMBER)
        ax.bar(i, hi - lo, bottom=lo, width=w, color=colour, linewidth=0, zorder=3)
        lows.append(min(lo, hi))
        highs.append(max(lo, hi))
        up = max(lo, hi) > 0   # label above the bar unless it sits below zero
        ax.annotate(f"{v:+,.0f}k", (i, max(lo, hi) if up else min(lo, hi)),
                    xytext=(0, 5 if up else -5), textcoords="offset points",
                    ha="center", va="bottom" if up else "top", fontsize=9.5,
                    color=INK, fontweight="bold")
        level = v if total else level + v
        if i < len(steps) - 1:
            ax.plot([i + w / 2, i + 1 - w / 2], [level, level], color=LINE_STRONG,
                    linewidth=1, zorder=2)

    ax.axhline(0, color=SLATE, linewidth=1, zorder=2)
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], color=INK_MUTED)
    ax.set_ylabel("Rupees, thousands")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    pad = 0.12 * (max(highs) - min(lows))
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    heading(fig, "Where the money came from, and where it went",
            "Every out-of-sample trade, on Rs 1 lakh of gross notional per pair. "
            "Reversions earned it; stops gave nearly all of it back.")
    save(fig, out)


def plot_equity(port: pd.DataFrame, out: Path) -> None:
    """Growth of 1, before and after costs, with the after-cost drawdown."""
    from .metrics import drawdown, equity

    eq_g, eq_n = equity(port["ret_gross"]), equity(port["ret_net"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 6.4), sharex=True,
                                   height_ratios=[1.7, 1])
    fig.subplots_adjust(top=0.87, hspace=0.12)

    style(ax1)
    ax1.axhline(1.0, color=SLATE, linewidth=1)
    ax1.plot(eq_g.index, eq_g.to_numpy(), color=BLUE, linewidth=2)
    ax1.plot(eq_n.index, eq_n.to_numpy(), color=AMBER, linewidth=2)
    for s, name in ((eq_g, "before costs"), (eq_n, "after costs")):
        ax1.annotate(f"{s.iloc[-1]:.2f}x {name}", xy=(s.index[-1], s.iloc[-1]),
                     xytext=(8, 0), textcoords="offset points", va="center",
                     fontsize=9.5, color=INK, annotation_clip=False)
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}x")
    ax1.set_ylabel("Growth of 1")

    style(ax2)
    dn = drawdown(eq_n) * 100
    ax2.fill_between(dn.index, dn.to_numpy(), 0, color=AMBER, alpha=0.12,
                     linewidth=0)
    ax2.plot(dn.index, dn.to_numpy(), color=AMBER, linewidth=1.4)
    ax2.axhline(0, color=SLATE, linewidth=1)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax2.set_ylabel("Drawdown after costs")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(eq_g.index[0], eq_g.index[-1])

    heading(fig, "Out-of-sample equity, before and after costs",
            f"{len(port):,} trading days, each one traded with parameters "
            f"fixed before it began.")
    legend_row(fig, [Line2D([], [], color=BLUE, linewidth=2, label="before costs"),
                     Line2D([], [], color=AMBER, linewidth=2, label="after costs")],
               y=0.925)
    save(fig, out)


def plot_annual(port: pd.DataFrame, out: Path) -> None:
    """Calendar-year return before and after costs."""
    rows = []
    for y, g in port.groupby(port.index.year):
        rows.append((y, len(g), ((1 + g["ret_gross"]).prod() - 1) * 100,
                     ((1 + g["ret_net"]).prod() - 1) * 100))
    full = [r for r in rows if r[1] > 200]
    n_pos = sum(r[2] > 0 for r in full)
    worst = sorted(r[0] for r in sorted(full, key=lambda r: r[2])[:2])

    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    fig.subplots_adjust(top=0.82)
    style(ax)
    x = np.arange(len(rows))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, [r[2] for r in rows], width=w, color=BLUE,
           linewidth=0, zorder=3)
    ax.bar(x + w / 2 + 0.01, [r[3] for r in rows], width=w, color=AMBER,
           linewidth=0, zorder=3)
    ax.axhline(0, color=SLATE, linewidth=1, zorder=2)
    ax.set_xticks(x, [f"{y}" if n > 200 else f"{y}\n(Jan-Jun)" for y, n, *_ in rows],
                  color=INK_MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("Return in the year")
    heading(fig, "Calendar-year return",
            f"Positive before costs in {n_pos} of {len(full)} full years. "
            f"The two worst, {worst[0]} and {worst[1]}, carry most of the loss.")
    legend_row(fig, [Patch(color=BLUE, label="before costs"),
                     Patch(color=AMBER, label="after costs")], y=0.905)
    save(fig, out)


def plot_cost_sensitivity(sens: pd.DataFrame, assumed_bps: float, n_trades: int,
                          out: Path) -> None:
    """Annualised return across a grid of cost assumptions."""
    x = sens["bps_per_leg"].to_numpy()
    y = sens["ann_return_%"].to_numpy()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    fig.subplots_adjust(top=0.85)
    style(ax, "both")
    ax.axhline(0, color=SLATE, linewidth=1)
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=PAPER, markeredgewidth=1.5, zorder=4)

    ax.axvline(assumed_bps, color=SLATE_DARK, linestyle="--", linewidth=1)
    at = float(np.interp(assumed_bps, x, y))
    ax.annotate(f"assumed {assumed_bps:g} bps: {at:+.2f}% a year",
                xy=(assumed_bps, at), xytext=(10, 6), textcoords="offset points",
                fontsize=9.5, color=INK)

    for i in range(len(x) - 1):
        if y[i] >= 0 >= y[i + 1]:
            cross = x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
            ax.plot([cross], [0], marker="o", markersize=9, color=INK,
                    markeredgecolor=PAPER, markeredgewidth=2, zorder=5)
            ax.annotate(f"break-even {cross:.1f} bps", xy=(cross, 0),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9.5, color=INK, fontweight="bold")
            break

    ax.set_xticks(np.arange(0, x.max() + 1, 5))
    ax.set_xlim(-0.8, x.max() + 0.8)
    ax.set_xlabel("Transaction cost, bps per leg")
    ax.set_ylabel("Annualised return")
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.1f}%" if abs(v) > 1e-9 else "0%")
    heading(fig, "Annualised return against the cost assumption",
            f"The same {n_trades} trades re-priced at each cost level. The trades "
            f"do not change, because the signal never sees costs.")
    save(fig, out)
