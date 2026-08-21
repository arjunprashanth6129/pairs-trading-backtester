"""Figures, on the same light palette as the dashboard that presents them.

Blue and amber rather than green and red. Most of these charts encode a
polarity, and green against red separates by a colour distance of only about 3.5
under deuteranopia, so roughly one man in twelve cannot tell the two apart. These
separate by roughly 25 in the worst case.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mirrors the site's design tokens so the repo figures and the web page read as
# one system.
PAPER = "#ffffff"
SURFACE = "#f8fafc"
INK = "#0f172a"
INK_MUTED = "#475569"
INK_DIM = "#64748b"
LINE = "#e2e8f0"
LINE_STRONG = "#cbd5e1"
BLUE = "#2563eb"
AMBER = "#b45309"
TEAL = "#0f766e"
VIOLET = "#6d28d9"
SLATE = "#64748b"

DPI = 200


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(PAPER)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=LINE, linewidth=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE_STRONG)
    ax.tick_params(colors=INK_DIM, labelsize=9)


def title(ax, text: str, deck: str | None = None, wrap: int = 96) -> None:
    lines = textwrap.wrap(deck, wrap) if deck else []
    ax.set_title(text, color=INK, loc="left", pad=10 + 11.5 * len(lines))
    for i, line in enumerate(lines):
        ax.annotate(line, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6 + 11.5 * (len(lines) - 1 - i)),
                    textcoords="offset points", ha="left", va="bottom",
                    color=INK_MUTED, fontsize=9.5, annotation_clip=False)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, **kw)
    for t in lg.get_texts():
        t.set_color(INK_MUTED)
    return lg


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ----------------------------------------------------------------------
def plot_pair_example(res, entry_z: float, exit_z: float, stop_z: float,
                      out: Path) -> None:
    """One pair's spread and z-score, with every entry and exit marked.

    This is the figure that shows the mechanism rather than the outcome: the
    bands are the fixed formation thresholds, and every marker is a decision the
    rules made without seeing anything to the right of it.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   height_ratios=[1, 1.15])
    p = res.params
    idx = res.spread.index

    style(ax1)
    ax1.plot(idx, res.spread.to_numpy(), color=BLUE, linewidth=1.4)
    ax1.axhline(p.mean, color=SLATE, linestyle="--", linewidth=1.1)
    ax1.annotate("formation mean", xy=(idx[0], p.mean), xytext=(6, 6),
                 textcoords="offset points", color=INK_DIM, fontsize=9)
    ax1.set_ylabel("Spread", color=INK_MUTED)
    title(ax1, f"{p.y} - {p.beta:.3f} x {p.x}",
          f"Hedge ratio, mean and standard deviation were all fixed on the "
          f"formation window (Engle-Granger p = {p.pvalue:.4f}, half-life "
          f"{p.half_life:.1f} days). Nothing here is re-estimated.")

    style(ax2)
    z = res.z.to_numpy()
    ax2.plot(idx, z, color=INK_MUTED, linewidth=1.2)
    ax2.axhline(0, color=LINE_STRONG, linewidth=1)
    for lv, lab, col in ((entry_z, "entry", AMBER), (-entry_z, None, AMBER),
                         (exit_z, "exit", TEAL), (-exit_z, None, TEAL),
                         (stop_z, "stop", VIOLET), (-stop_z, None, VIOLET)):
        ax2.axhline(lv, color=col, linestyle=":", linewidth=1.1)
        if lab:
            ax2.annotate(lab, xy=(idx[-1], lv), xytext=(6, 0),
                         textcoords="offset points", va="center", color=col,
                         fontsize=9, fontweight="bold", annotation_clip=False)

    if len(res.trades):
        for _, tr in res.trades.iterrows():
            ei, xi = tr["entry_date"], tr["exit_date"]
            win = tr["pnl_net"] > 0
            ax2.scatter([ei], [res.z.loc[ei]], s=46, marker="v" if tr["side"] ==
                        "short_spread" else "^", color=BLUE, zorder=5)
            ax2.scatter([xi], [res.z.loc[xi]], s=40, marker="o",
                        facecolor=PAPER if not win else TEAL,
                        edgecolors=TEAL if win else AMBER, linewidths=1.5,
                        zorder=5)

    ax2.set_ylabel("z-score", color=INK_MUTED)
    n_win = int((res.trades["pnl_net"] > 0).sum()) if len(res.trades) else 0
    title(ax2, "Entries and exits",
          f"Triangles open a position, circles close it; filled circles made "
          f"money. {len(res.trades)} trades, {n_win} profitable.")
    fig.tight_layout()
    save(fig, out)


def plot_equity(port: pd.DataFrame, benchmark: pd.Series, out: Path) -> None:
    """Growth of 1, before and after costs, against buying the universe."""
    from .metrics import drawdown, equity

    eq_g, eq_n = equity(port["ret_gross"]), equity(port["ret_net"])
    eq_b = equity(benchmark.reindex(port.index).fillna(0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                   height_ratios=[1.5, 1])
    style(ax1)
    ax1.plot(eq_b.index, eq_b.to_numpy(), color=SLATE, linewidth=1.4,
             linestyle=":", label="Buy and hold the universe")
    ax1.plot(eq_g.index, eq_g.to_numpy(), color=BLUE, linewidth=1.9,
             label="Strategy, before costs")
    ax1.plot(eq_n.index, eq_n.to_numpy(), color=AMBER, linewidth=2.1,
             label="Strategy, after costs")
    ax1.axhline(1.0, color=LINE_STRONG, linewidth=1)
    ax1.set_ylabel("Growth of 1.0", color=INK_MUTED)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}x")
    legend(ax1, loc="upper left")
    title(ax1, "Out-of-sample equity",
          "Every day here was traded with parameters fixed before it began. The "
          "gap between the two strategy lines is transaction cost, and it is "
          "the whole story of this project.")

    style(ax2)
    dn = drawdown(eq_n)
    ax2.fill_between(dn.index, dn.to_numpy() * 100, 0, color=AMBER, alpha=0.18,
                     linewidth=0)
    ax2.plot(dn.index, dn.to_numpy() * 100, color=AMBER, linewidth=1.3)
    ax2.axhline(0, color=LINE_STRONG, linewidth=1)
    ax2.set_ylabel("Drawdown", color=INK_MUTED)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    title(ax2, "Drawdown, after costs")
    fig.tight_layout()
    save(fig, out)


def plot_cost_sensitivity(sens: pd.DataFrame, assumed_bps: float,
                          out: Path) -> None:
    """Where the strategy breaks even on cost."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    style(ax)
    x = sens["bps_per_leg"].to_numpy()
    y = sens["ann_return_%"].to_numpy()
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color=INK_MUTED, linewidth=1.2)
    ax.axvline(assumed_bps, color=AMBER, linestyle="--", linewidth=1.4)
    ax.annotate(f"assumed\n{assumed_bps:.0f} bps", xy=(assumed_bps, y.min()),
                xytext=(8, 12), textcoords="offset points", color=AMBER,
                fontsize=9, fontweight="bold")

    # Linear crossing between the two points that straddle zero.
    cross = None
    for i in range(len(x) - 1):
        if y[i] >= 0 >= y[i + 1]:
            cross = x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
            break
    if cross is not None:
        ax.plot([cross], [0], marker="o", markersize=9, color=TEAL, zorder=5)
        ax.annotate(f"break-even {cross:.1f} bps per leg", xy=(cross, 0),
                    xytext=(10, 14), textcoords="offset points", color=TEAL,
                    fontsize=9.5, fontweight="bold")

    ax.set_xlabel("Transaction cost, bps per leg", color=INK_MUTED)
    ax.set_ylabel("Annualised return", color=INK_MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    title(ax, "How much cost the strategy can carry",
          "The same trades, re-priced. The signal never saw costs, so varying "
          "the charge while holding the strategy fixed is the honest way to "
          "locate the break-even.")
    fig.tight_layout()
    save(fig, out)
