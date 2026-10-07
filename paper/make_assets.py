#!/usr/bin/env python3
"""Generate every number, table and figure in paper/main.tex from committed results.

Reads results/ (written by the pipeline) and writes:

    paper/numbers.tex        one \\newcommand per number quoted in the text
    paper/tables/*.tex       table bodies
    paper/figures/*.pdf      vector figures

The statistical tests in the paper (Newey-West, trade-level t, binomial and
chi-square) are computed here from results/trades.csv and
results/daily_returns.csv. The figures for the run before the entry-rule
correction are read from the results committed at 00591ed, the last commit
before the fix, so the comparison in the appendix is reproducible too.

Run from the repository root:

    python paper/make_assets.py
"""
from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIG, TAB = PAPER / "figures", PAPER / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
RES = ROOT / "results"
PRE_FIX = "00591ed"

BLUE, ORANGE, GREY, INK = "#1f5aa6", "#d9702a", "#8c8c8c", "#222222"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 1.0, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlelocation": "left",
})
WIDTH = 6.5

MACROS: dict[str, str] = {}


def num(x: float, d: int, signed: bool = False) -> str:
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return f"\\ensuremath{{{s}}}" if s.startswith(("-", "+")) else s


def pc(x: float, d: int = 2, signed: bool = False) -> str:
    return num(x, d, signed) + "\\%"


def inr(x: float, d: int = 0) -> str:
    """Rupee amount with standard thousands grouping, minus sign in maths."""
    s = f"{abs(x):,.{d}f}".replace(",", "{,}")
    return ("\\ensuremath{-}" if x < 0 else "") + s


def macro(name: str, value) -> None:
    assert name.isalpha(), name
    MACROS[name] = str(value)


def tsign(x: float, d: int) -> str:
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


summary = json.loads((RES / "summary.json").read_text())
dash = json.loads((RES / "dashboard.json").read_text())
meta = dash["meta"]
trades = pd.read_csv(RES / "trades.csv", parse_dates=["entry_date", "exit_date"])
pairs = pd.read_csv(RES / "selected_pairs.csv")
daily = pd.read_csv(RES / "daily_returns.csv", parse_dates=["date"], index_col="date")
costs = pd.read_csv(RES / "cost_sensitivity.csv")
cap = float(dash["capital_per_pair"])

# ---------------------------------------------------------------- design
macro("NUniverse", len(meta["universe"]))
macro("NTraded", len(meta["traded"]))
macro("NCandidates", meta["candidate_pairs"])
macro("NWindows", meta["n_windows"])
macro("FormMonths", meta["formation_months"])
macro("TradeMonths", meta["trading_months"])
macro("EntryZ", num(meta["entry_z"], 0))
macro("ExitZ", num(meta["exit_z"], 1))
macro("StopZ", num(meta["stop_z"], 0))
macro("MaxHold", meta["max_holding_days"])
macro("BpsLeg", f"{meta['bps_per_leg']:.0f}")
macro("PThreshold", num(meta["pvalue_threshold"], 2))
macro("MaxPairs", meta["max_pairs"])
macro("SampleStart", meta["start"])
macro("SampleEnd", meta["end"])
macro("OOSStart", meta["oos_start"])
macro("OOSEnd", meta["oos_end"])
g, n = summary["gross"], summary["net"]
macro("OOSDays", f"{g['days']:,}".replace(",", "{,}"))
macro("Capital", inr(cap))
macro("NConvStop", 0)

# ---------------------------------------------------------------- selection
sel = summary["selection"]
macro("PairWindows", sel["pair_windows"])
macro("UniquePairs", sel["unique_pairs"])
macro("MedianP", num(sel["median_pvalue"], 3))
macro("MedianHalfLife", num(sel["median_half_life_days"], 1))

# ---------------------------------------------------------------- performance
for tag, s in [("G", g), ("N", n), ("U", summary["benchmark_universe"]),
               ("M", summary["benchmark_nifty"])]:
    macro(f"Ret{tag}", pc(s["ann_return_%"], 2, signed=tag in "GN"))
    macro(f"Vol{tag}", pc(s["ann_vol_%"], 2))
    macro(f"Sharpe{tag}", num(s["sharpe"], 2))
    macro(f"DD{tag}", pc(abs(s["max_drawdown_%"]), 2))
    macro(f"Growth{tag}", num(1 + s["total_return_%"] / 100, 2))
nt = len(trades)
macro("NTrades", nt)
macro("MeanHold", num(trades["holding_days"].mean(), 1))
macro("MedianHold", f"{trades['holding_days'].median():.0f}")
macro("WinGross", pc(100 * (trades["pnl_gross"] > 0).mean(), 1))
macro("WinNet", pc(100 * (trades["pnl_net"] > 0).mean(), 1))
mg = trades["pnl_gross"].mean()
macro("MeanGrossPnL", inr(mg))
macro("MeanGrossBps", f"{100 * 100 * mg / cap:.0f}")
macro("MeanCost", inr(trades["cost"].mean()))
macro("MeanCostBps", f"{100 * 100 * trades['cost'].mean() / cap:.0f}")

# Newey-West t on mean daily returns, 10 lags
def nw_t(r: pd.Series, lags: int = 10) -> tuple[float, float]:
    fit = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


tg, pg = nw_t(daily["ret_gross"])
tn, pn = nw_t(daily["ret_net"])
macro("MeanDailyBps", num(1e4 * daily["ret_gross"].mean(), 2))
macro("NWtGross", num(tg, 2))
macro("NWpGross", num(pg, 2))
macro("NWtNet", num(tn, 2))
macro("NWpNet", num(pn, 2))
tt = stats.ttest_1samp(trades["pnl_gross"], 0)
macro("TradeT", num(tt.statistic, 2))

perf = [("Growth of 1", lambda s: f"{1 + s['total_return_%'] / 100:.2f}"),
        ("Annualised return (\\%)", lambda s: tsign(s["ann_return_%"], 2)),
        ("Annualised volatility (\\%)", lambda s: f"{s['ann_vol_%']:.2f}"),
        ("Sharpe ratio, $r_f = 6\\%$", lambda s: tsign(s["sharpe"], 2)),
        ("Maximum drawdown (\\%)", lambda s: f"{abs(s['max_drawdown_%']):.2f}")]
cols = [g, n, summary["benchmark_universe"], summary["benchmark_nifty"]]
rows = [rf"{name} & " + " & ".join(f(c) for c in cols) + r" \\" for name, f in perf]
(TAB / "performance.tex").write_text(r"""\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Pairs strategy} & \multicolumn{2}{c}{Long-only references} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
 & Before costs & After costs & 11 banks, equal weight & Nifty 50 \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- exits
reason_name = {"reverted": "Converged ($|z| \\le 0.5$)", "stop": "Stopped out ($|z| \\ge 4$)",
               "timeout": "Timed out (60 days)", "window_end": "Trading period ended"}
order = ["reverted", "stop", "timeout", "window_end"]
ex_rows = []
for r in order:
    t = trades[trades["reason"] == r]
    ex_rows.append(rf"{reason_name[r]} & {len(t)} & {100 * len(t) / nt:.1f} & "
                   rf"{inr(t['pnl_gross'].sum())} & {inr(t['cost'].sum())} & "
                   rf"{inr(t['pnl_net'].sum())} & {t['holding_days'].median():g} \\")
ex_rows.append(rf"\midrule Total & {nt} & 100.0 & {inr(trades['pnl_gross'].sum())} & "
               rf"{inr(trades['cost'].sum())} & {inr(trades['pnl_net'].sum())} & "
               rf"{trades['holding_days'].median():g} \\")
(TAB / "exits.tex").write_text(r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Exit & Trades & Share (\%) & Gross P\&L & Costs & Net P\&L & Median days held \\
\midrule
""" + "\n".join(ex_rows).replace("{,}", ",") + "\n\\bottomrule\n\\end{tabular}\n")

conv = trades[trades["reason"] == "reverted"]
stop = trades[trades["reason"] == "stop"]
macro("NConv", len(conv))
macro("NStop", len(stop))
macro("NTimeout", int((trades["reason"] == "timeout").sum()))
macro("NWindowEnd", int((trades["reason"] == "window_end").sum()))
macro("ShareStop", pc(100 * len(stop) / nt, 1))
macro("ShareStopRound", f"{round(100 * len(stop) / nt)}\\%")
macro("StopToConv", num(len(stop) / len(conv), 1))
macro("ConvGross", inr(conv["pnl_gross"].sum()))
macro("StopGross", inr(abs(stop["pnl_gross"].sum())))
macro("AvgConv", inr(conv["pnl_gross"].mean()))
macro("AvgStop", inr(abs(stop["pnl_gross"].mean())))
be = abs(stop["pnl_gross"].mean()) / (conv["pnl_gross"].mean() + abs(stop["pnl_gross"].mean()))
macro("BreakEvenMix", pc(100 * be, 1))
realised = len(conv) / (len(conv) + len(stop))
macro("RealisedMix", pc(100 * realised, 1))
MACROS["NConvStop"] = str(len(conv) + len(stop))
macro("BinomMixP", num(stats.binomtest(len(conv), len(conv) + len(stop), be).pvalue, 2))
macro("StopMedianHold", num(stop["holding_days"].median(), 1))
macro("StopWithinFour", int((stop["holding_days"] <= 4).sum()))

# ---------------------------------------------------------------- convergence speed
rv = dash["reversion"]
macro("ConvMedianHold", f"{rv['median_hold']:.0f}")
macro("ConvHalfLife", num(rv["median_half_life"], 1))
macro("ConvRatio", num(rv["median_hold_over_half_life"], 2))
macro("ConvEntryZ", num(rv["median_entry_abs_z"], 2))
decay = rv["median_entry_abs_z"] / meta["exit_z"]
macro("ConvDecay", num(decay, 1))
macro("ConvHalfLives", num(math.log2(decay), 1))

# ---------------------------------------------------------------- p-value groups and chi-square
key = pairs.assign(window=pairs["index"])[["window", "pair", "pvalue", "half_life"]]
tp = trades.merge(key, on=["window", "pair"], how="left")
assert tp["pvalue"].notna().all()
bins = [0, 0.01, 0.03, 0.05]
tp["group"] = pd.cut(tp["pvalue"], bins=bins, include_lowest=True)
grp_rows = []
labels = {0: "0.00 to 0.01", 1: "0.01 to 0.03", 2: "0.03 to 0.05"}
for gi, (gname, t) in enumerate(tp.groupby("group", observed=True)):
    grp_rows.append(rf"{labels[gi]} & {len(t)} & "
                    rf"{100 * (t['reason'] == 'stop').mean():.1f} & "
                    rf"{100 * (t['reason'] == 'reverted').mean():.1f} & {inr(t['pnl_gross'].mean())} \\")
(TAB / "pvalue.tex").write_text(r"""\begin{tabular}{@{}lrrrr@{}}
\toprule
Formation $p$-value & Trades & Stopped out (\%) & Converged (\%) & Mean gross P\&L \\
\midrule
""" + "\n".join(grp_rows) + "\n\\bottomrule\n\\end{tabular}\n")
ct = pd.crosstab(tp["group"], tp["reason"] == "stop")
chi2, chip, dof, _ = stats.chi2_contingency(ct, correction=False)
macro("ChiTwo", num(chi2, 2))
macro("ChiDof", dof)
macro("ChiP", num(chip, 2))

# ---------------------------------------------------------------- periods and concentration
ann = pd.DataFrame(dash["annual"])
full = ann[ann["days"] >= 240]
macro("NFullYears", len(full))
macro("NYearsPos", int((full["gross"] > 0).sum()))
win_pnl = trades.groupby("window")["pnl_gross"].sum().reindex(range(meta["n_windows"]), fill_value=0)
macro("NWindowsPos", int((win_pnl > 0).sum()))
y20 = ann[ann["year"] == 2020]["gross"].iloc[0]
macro("GrossTwentyTwenty", pc(y20, 2, signed=True))
worst_w = int(win_pnl.idxmin())
macro("WorstWindowLoss", inr(abs(win_pnl.min())))
macro("TotalGross", inr(trades["pnl_gross"].sum()))
ww = pairs[pairs["index"] == worst_w]
macro("WorstWindowStart", ww["trading_start"].iloc[0])
legs = pd.concat([pairs["y"], pairs["x"]]).groupby(pd.concat([pairs["index"], pairs["index"]])).value_counts()
macro("NConcentrated", int((legs.groupby(level=0).max() >= 3).sum()))
kotak = trades[(trades["window"] == worst_w) & (trades["pair"].str.contains("KOTAKBANK"))
               & (trades["entry_date"] == trades.loc[trades["window"] == worst_w, "entry_date"].mode()[0])]
macro("KotakSameDay", len(kotak))
macro("KotakPairsSelected", int(ww["pair"].str.contains("KOTAKBANK").sum()))
macro("KotakEntry", kotak["entry_date"].iloc[0].date().isoformat())
psb = kotak[~kotak["pair"].str.contains("FEDERALBNK")]
fed = kotak[kotak["pair"].str.contains("FEDERALBNK")]
assert len(psb) == 3 and len(fed) == 1 and psb["exit_date"].nunique() == 1
macro("KotakExit", psb["exit_date"].iloc[0].date().isoformat())
macro("KotakEach", inr(round(abs(psb["pnl_gross"].mean()), -2)))
macro("KotakFedExit", fed["exit_date"].iloc[0].date().isoformat())
macro("KotakFedLoss", inr(abs(fed["pnl_gross"].iloc[0])))
macro("KotakSameDayLoss", inr(abs(kotak["pnl_gross"].sum())))
axfed = trades[(trades["pair"] == "AXISBANK.NS~FEDERALBNK.NS") & (trades["entry_date"].dt.year == 2020)]
macro("AxisFedStops", int((axfed["reason"] == "stop").sum()))
macro("AxisFedLoss", inr(abs(axfed["pnl_gross"].sum())))
iid = trades[trades["pair"].str.contains("INDUSINDBK") & trades["pair"].str.contains("IDFCFIRSTB")
             & (trades["entry_date"].dt.year == 2020)]
macro("IndIdfcTrades", len(iid))
macro("IndIdfcLoss", inr(abs(iid["pnl_gross"].sum())))

# ---------------------------------------------------------------- costs and exposure
c = costs.set_index("bps_per_leg")["ann_return_%"]
x = c.index.to_numpy()
y_ = c.to_numpy()
i = int(np.where((y_[:-1] >= 0) & (y_[1:] < 0))[0][0])
breakeven = x[i] + (0 - y_[i]) * (x[i + 1] - x[i]) / (y_[i + 1] - y_[i])
macro("BreakEven", num(breakeven, 1))
macro("BreakEvenSimple", num(mg / cap * 1e4 / 2, 1))
ne = summary["neutrality_net"]
macro("Beta", num(ne["beta_to_market"], 3))
macro("BetaMove", num(abs(ne["beta_to_market"]), 2))
macro("CorrMkt", num(ne["corr_to_market"], 3))
macro("RsqMkt", num(100 * ne["r_squared"], 1))

# ---------------------------------------------------------------- example pair
exm = dash["example"]
macro("ExP", num(exm["pvalue"], 3))
macro("ExHalfLife", num(exm["half_life"], 1))
top = dash["top_pairs"][0]
macro("ExTimesSelected", top["times_selected"])

# ---------------------------------------------------------------- pre-correction run
def git_file(path: str) -> str:
    return subprocess.run(["git", "show", f"{PRE_FIX}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


old_sum = json.loads(git_file("results/summary.json"))
old_tr = pd.read_csv(io.StringIO(git_file("results/trades.csv")))
old_cs = pd.read_csv(io.StringIO(git_file("results/cost_sensitivity.csv"))).set_index("bps_per_leg")["ann_return_%"]
ox, oy = old_cs.index.to_numpy(), old_cs.to_numpy()
j = int(np.where((oy[:-1] >= 0) & (oy[1:] < 0))[0][0])
old_be = ox[j] + (0 - oy[j]) * (ox[j + 1] - ox[j]) / (oy[j + 1] - oy[j])
fix_rows = [
    ("Round trips", f"{len(old_tr):,}", f"{nt}"),
    ("Entered with $|z| \\ge 4$ already", f"{int((old_tr['entry_z'].abs() >= 4).sum())}",
     f"{int((trades['entry_z'].abs() >= 4).sum())}"),
    ("Stopped out", f"{int((old_tr['reason'] == 'stop').sum())}", f"{len(stop)}"),
    ("Median holding period (days)", f"{old_tr['holding_days'].median():.0f}",
     f"{trades['holding_days'].median():.0f}"),
    ("Annualised return before costs (\\%)", tsign(old_sum["gross"]["ann_return_%"], 2),
     tsign(g["ann_return_%"], 2)),
    ("Annualised return after costs (\\%)", tsign(old_sum["net"]["ann_return_%"], 2),
     tsign(n["ann_return_%"], 2)),
    ("Maximum drawdown after costs (\\%)", f"{abs(old_sum['net']['max_drawdown_%']):.2f}",
     f"{abs(n['max_drawdown_%']):.2f}"),
    ("Break-even cost per leg (bps)", f"{old_be:.1f}", f"{breakeven:.1f}"),
]
(TAB / "correction.tex").write_text(r"""\begin{tabular}{@{}lrr@{}}
\toprule
 & Before correction & After correction \\
\midrule
""" + "\n".join(rf"{a} & {b} & {c_} \\" for a, b, c_ in fix_rows) + "\n\\bottomrule\n\\end{tabular}\n")
macro("OldTrades", f"{len(old_tr):,}".replace(",", "{,}"))
macro("OldBeyondStop", int((old_tr["entry_z"].abs() >= 4).sum()))
macro("OldStops", int((old_tr["reason"] == "stop").sum()))
macro("OldNet", pc(old_sum["net"]["ann_return_%"], 2, signed=True))
macro("OldGross", pc(old_sum["gross"]["ann_return_%"], 2, signed=True))
macro("FixNetChange", num(n["ann_return_%"] - old_sum["net"]["ann_return_%"], 1))
macro("FixGrossChange", num(g["ann_return_%"] - old_sum["gross"]["ann_return_%"], 1))
macro("OldBreakEven", num(old_be, 1))

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def panel(ax, text):
    ax.set_title(text, fontsize=8.5, pad=6)


# Equity and drawdown, from the daily return series
eq_g = (1 + daily["ret_gross"]).cumprod()
eq_n = (1 + daily["ret_net"]).cumprod()
fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 3.6), sharex=True,
                           gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.2})
a.plot(eq_g.index, eq_g, color=BLUE, lw=1.0, label="Before costs")
a.plot(eq_n.index, eq_n, color=ORANGE, lw=1.0, label="After costs")
a.axhline(1, color=GREY, lw=0.6)
a.set_ylabel("Growth of 1")
a.legend(loc="upper left")
panel(a, "Panel A. Growth of one unit of capital")
dd = eq_n / eq_n.cummax() - 1
b.fill_between(dd.index, 100 * dd, 0, color=ORANGE, alpha=0.25, lw=0)
b.plot(dd.index, 100 * dd, color=ORANGE, lw=0.8)
b.set_ylabel("Drawdown (%)")
panel(b, "Panel B. Drawdown after costs")
fig.savefig(FIG / "equity.pdf")
plt.close(fig)

# Example pair: z-score with each trade shaded
ser = pd.DataFrame(exm["series"])
ser["d"] = pd.to_datetime(ser["d"])
fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
for lvl, ls in [(meta["entry_z"], (0, (3, 2))), (meta["exit_z"], (0, (1, 1.5))),
                (meta["stop_z"], "-")]:
    for sgn in (1, -1):
        ax.axhline(sgn * lvl, color=GREY, lw=0.6, ls=ls)
ax.axhline(0, color=GREY, lw=0.4)
for t in exm["trades"]:
    colour = BLUE if t["pnl_net"] > 0 else ORANGE
    ax.axvspan(pd.Timestamp(t["entry"]), pd.Timestamp(t["exit"]), color=colour, alpha=0.13, lw=0)
    ax.scatter([pd.Timestamp(t["entry"])], [t["entry_z"]], marker="v", s=18, color=INK, zorder=4)
    ax.scatter([pd.Timestamp(t["exit"])], [t["exit_z"]], marker="o", s=12, color=INK, zorder=4)
ax.plot(ser["d"], ser["z"], color=INK, lw=0.9, zorder=3)
for lvl, lab in [(meta["entry_z"], "entry"), (meta["exit_z"], "exit"), (meta["stop_z"], "stop")]:
    for sgn in (1, -1):
        ax.text(1.008, sgn * lvl, f"{sgn * lvl:+g}  {lab}", transform=ax.get_yaxis_transform(),
                fontsize=7, va="center", ha="left", color="#555555")
ax.set_yticks([-4, -2, 0, 2, 4])
ax.set_ylabel("Spread z-score")
ax.set_ylim(-5.2, 5.2)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
fig.savefig(FIG / "example.pdf")
plt.close(fig)

# Calendar years and cost sensitivity
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1]})
yrs = ann["year"].astype(str).where(ann["days"] >= 240, ann["year"].astype(str) + "*")
xx = np.arange(len(ann))
a.bar(xx - 0.19, ann["gross"], width=0.38, color=BLUE, label="Before costs")
a.bar(xx + 0.19, ann["net"], width=0.38, color=ORANGE, label="After costs")
a.axhline(0, color=INK, lw=0.6)
a.set_xticks(xx)
a.set_xticklabels(yrs, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
a.set_ylabel("Return (%)")
a.legend(loc="upper right")
a.grid(axis="x", visible=False)
panel(a, "Panel A. Calendar-year return")
b.plot(costs["bps_per_leg"], costs["ann_return_%"], color=INK, lw=1.0, marker="o", ms=2.5)
b.axhline(0, color=GREY, lw=0.6)
b.axvline(breakeven, color=BLUE, lw=0.7, ls=(0, (3, 2)))
b.axvline(meta["bps_per_leg"], color=ORANGE, lw=0.7, ls=(0, (3, 2)))
b.text(breakeven - 0.5, -0.9, f"break-even\n{breakeven:.1f} bps", fontsize=7,
       color=BLUE, va="center", ha="right")
b.text(meta["bps_per_leg"] + 0.5, 0.6, f"assumed\n{meta['bps_per_leg']:.0f} bps",
       fontsize=7, color=ORANGE, va="center", ha="left")
b.set_xlabel("Cost per leg (bps)")
b.set_ylabel("Annualised return after costs (%)")
panel(b, "Panel B. Sensitivity to the cost assumption")
fig.savefig(FIG / "years_costs.pdf")
plt.close(fig)

lines = ["% Generated by paper/make_assets.py. Do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS.items()]
(PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"{len(MACROS)} numbers, {len(list(TAB.glob('*.tex')))} tables, "
      f"{len(list(FIG.glob('*.pdf')))} figures written")
