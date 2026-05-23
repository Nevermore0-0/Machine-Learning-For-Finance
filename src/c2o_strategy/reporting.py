"""Output tables, figures and HTML reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig
from c2o_strategy.metrics import max_drawdown, sharpe_ratio


def compute_equal_weight_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the equal-weighted overnight/intraday/total stylised-fact series."""

    mask = panel["in_universe"] & panel["r_overnight"].notna() & panel["r_intraday"].notna()
    daily = (
        panel.loc[mask]
        .groupby("date")
        .agg(
            overnight=("r_overnight", "mean"),
            intraday=("r_intraday", "mean"),
            close_to_close=("r_close_close", "mean"),
            names=("instrument_id", "nunique"),
        )
        .reset_index()
    )
    return daily


def plot_equal_weight_decomposition(decomposition: pd.DataFrame, path: Path) -> None:
    """Plot cumulative growth of overnight, intraday and close-to-close streams."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column, label in [
        ("overnight", "Overnight"),
        ("intraday", "Intraday"),
        ("close_to_close", "Close-to-close"),
    ]:
        growth = (1.0 + decomposition[column].fillna(0.0)).cumprod()
        ax.plot(decomposition["date"], growth, label=label, linewidth=1.6)
    ax.set_title("Equal-weight universe return decomposition")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_strategy_cumulative(daily_by_aum: dict[float, pd.DataFrame], path: Path) -> None:
    """Plot cumulative net strategy returns for all required AUM levels."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for aum, daily in daily_by_aum.items():
        growth = (1.0 + daily["net_return"].fillna(0.0)).cumprod()
        ax.plot(daily["date"], growth, label=f"${aum/1_000_000:.0f}M", linewidth=1.5)
    ax.set_title("C2O strategy net cumulative return")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Portfolio AUM")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ic(ic_daily: pd.DataFrame, path: Path) -> None:
    """Plot rolling mean IC."""

    if ic_daily.empty:
        return
    data = ic_daily.sort_values("date").copy()
    data["rolling_63d_ic"] = data["ic"].rolling(63, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["date"], data["rolling_63d_ic"], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("63-day rolling Information Coefficient")
    ax.set_ylabel("Spearman IC")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_tearsheet(
    returns: pd.Series, benchmark: pd.Series, output_path: Path, title: str
) -> None:
    """Generate QuantStats HTML, falling back to a compact HTML if unavailable."""

    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index().dropna()
    benchmark = benchmark.copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark = benchmark.sort_index().dropna()

    try:
        import quantstats as qs

        qs.extend_pandas()
        qs.reports.html(returns, benchmark=benchmark, output=str(output_path), title=title)
        return
    except Exception as exc:  # pragma: no cover - exercised only without quantstats
        _write_fallback_html(returns, benchmark, output_path, title, exc)


def _write_fallback_html(
    returns: pd.Series, benchmark: pd.Series, output_path: Path, title: str, exc: Exception
) -> None:
    joined = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    strategy_growth = (1.0 + joined["strategy"]).cumprod()
    benchmark_growth = (1.0 + joined["benchmark"]).cumprod()
    drawdown = strategy_growth / strategy_growth.cummax() - 1.0

    plot_path = output_path.with_suffix(".fallback.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(strategy_growth.index, strategy_growth, label="Strategy")
    axes[0].plot(benchmark_growth.index, benchmark_growth, label="SP500_TR")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p><strong>QuantStats was unavailable or failed:</strong> {exc}</p>
<p>This fallback is not a substitute for the required QuantStats artefact, but
it keeps the reproduction command auditable until dependencies are installed.</p>
<ul>
<li>Strategy Sharpe: {sharpe_ratio(joined['strategy']):.3f}</li>
<li>Benchmark Sharpe: {sharpe_ratio(joined['benchmark']):.3f}</li>
<li>Strategy max drawdown: {max_drawdown(joined['strategy']):.2%}</li>
<li>Benchmark max drawdown: {max_drawdown(joined['benchmark']):.2%}</li>
</ul>
<img src="{plot_path.name}" alt="fallback cumulative return and drawdown">
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def plot_yearly_sharpe(daily: pd.DataFrame, path: Path) -> None:
    """Bar chart of net Sharpe ratio by calendar year."""
    from c2o_strategy.metrics import sharpe_ratio as sr

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    years, sharpes = [], []
    for year, chunk in data.groupby("year", sort=True):
        years.append(int(year))
        sharpes.append(sr(chunk["net_return"]))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colours = ["#2ecc71" if s >= 0 else "#e74c3c" for s in sharpes]
    ax.bar(years, sharpes, color=colours, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Annual net Sharpe ratio (250M AUM)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gross_to_net_waterfall(perf_row: dict, path: Path) -> None:
    """Waterfall from gross Sharpe to net Sharpe showing cost drag."""
    labels = ["Gross", "Commission", "Slippage", "Borrow", "Net"]
    gross = perf_row.get("gross_sharpe", 0)
    comm = perf_row.get("commission_sharpe_drag", perf_row.get("commission_drag", 0))
    slip = perf_row.get("slippage_sharpe_drag", perf_row.get("slippage_drag", 0))
    borr = perf_row.get("borrow_sharpe_drag", perf_row.get("borrow_drag", 0))
    net = perf_row.get("net_sharpe", 0)

    values = [gross, -comm, -slip, -borr, net]
    bottoms = [0, gross, gross - comm, gross - comm - slip, 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["#3498db", "#e74c3c", "#e74c3c", "#e74c3c", "#2ecc71"]
    ax.bar(labels, values, bottom=[0, 0, 0, 0, 0], color="none")  # invisible for alignment

    for i, (label, val, bot, col) in enumerate(zip(labels, values, bottoms, colours)):
        if i == 0 or i == 4:
            ax.bar(i, val, color=col, edgecolor="white", width=0.6)
        else:
            ax.bar(i, val, bottom=bot + val if val < 0 else bot, color=col,
                   edgecolor="white", width=0.6)

    # Redraw properly as a waterfall
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    running = gross
    positions = [gross]
    for drag in [comm, slip, borr]:
        positions.append(drag)
        running -= drag
    positions.append(net)

    bar_bottoms = [0, gross - comm, gross - comm - slip, gross - comm - slip - borr, 0]
    bar_heights = [gross, comm, slip, borr, net]
    for i, (lbl, h, b, c) in enumerate(zip(labels, bar_heights, bar_bottoms, colours)):
        ax2.bar(i, h, bottom=b, color=c, edgecolor="white", width=0.6)
        ax2.text(i, b + h + 0.02, f"{h:.3f}", ha="center", fontsize=9)

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_title("Gross-to-net Sharpe waterfall (250M AUM)")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(path, dpi=160)
    plt.close(fig)
    plt.close(fig2)


def plot_universe_count(panel: pd.DataFrame, path: Path) -> None:
    """Plot daily number of eligible names over time."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    eligible = data.loc[data["is_trade_eligible"]].groupby("date").size()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if "in_universe" in data.columns:
        in_univ = data.loc[data["in_universe"]].groupby("date").size()
        ax.plot(in_univ.index, in_univ.values, label="In universe", linewidth=1.2, alpha=0.7)
    ax.plot(eligible.index, eligible.values, label="Trade-eligible", linewidth=1.2)
    ax.set_ylabel("Number of names")
    ax.set_title("Daily universe and eligible set size")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_borrow_tier_distribution(panel: pd.DataFrame, path: Path) -> None:
    """Stacked area chart of borrow tier distribution over time."""
    data = panel.loc[panel["is_trade_eligible"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    counts = data.groupby(["date", "borrow_tier"]).size().unstack(fill_value=0)
    for tier in ["A", "B", "C"]:
        if tier not in counts.columns:
            counts[tier] = 0
    counts = counts[["A", "B", "C"]]
    fractions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(
        fractions.index,
        fractions["A"], fractions["B"], fractions["C"],
        labels=["Tier A (40 bps)", "Tier B (200 bps)", "Tier C (800 bps)"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_ylabel("Fraction of eligible names")
    ax.set_title("Borrow tier distribution over time")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_ablation(ablation: pd.DataFrame, path: Path, aum: float = 250_000_000.0) -> None:
    """Bar chart of Sharpe impact from removing each feature group."""
    data = ablation.loc[ablation["AUM"].eq(aum)].copy()
    if data.empty:
        return
    full = data.loc[data["model_variant"].eq("full_model"), "net_sharpe"]
    if full.empty:
        return
    full_sharpe = float(full.iloc[0])
    removes = data.loc[~data["model_variant"].eq("full_model")].copy()
    removes["sharpe_impact"] = removes["net_sharpe"] - full_sharpe
    removes = removes.sort_values("sharpe_impact")

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#e74c3c" if v < 0 else "#2ecc71" for v in removes["sharpe_impact"]]
    ax.barh(removes["removed_group"], removes["sharpe_impact"], color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in net Sharpe vs full model")
    ax.set_title("Feature group ablation impact (250M AUM)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_drawdown(daily: pd.DataFrame, path: Path) -> None:
    """Underwater chart of net strategy drawdowns."""
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    wealth = (1.0 + data["net_return"].fillna(0.0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(data["date"], dd, 0, alpha=0.5, color="#e74c3c")
    ax.set_ylabel("Drawdown")
    ax.set_title("Strategy drawdown (250M AUM)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ic_comparison(
    ic_linear: pd.DataFrame, ic_lgbm: pd.DataFrame, path: Path
) -> None:
    """Compare rolling IC of linear vs LightGBM model."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    window = 63

    if not ic_linear.empty:
        data_lin = ic_linear.sort_values("date").copy()
        ic_col = "ic" if "ic" in data_lin.columns else "IC"
        data_lin["rolling"] = data_lin[ic_col].rolling(window, min_periods=20).mean()
        ax.plot(data_lin["date"], data_lin["rolling"], label="Linear", linewidth=1.4)

    if not ic_lgbm.empty:
        data_lgb = ic_lgbm.sort_values("date").copy()
        ic_col_lgb = "ic_lgbm" if "ic_lgbm" in data_lgb.columns else "ic"
        data_lgb["rolling"] = data_lgb[ic_col_lgb].rolling(window, min_periods=20).mean()
        ax.plot(data_lgb["date"], data_lgb["rolling"], label="LightGBM", linewidth=1.4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("63-day rolling IC: linear vs LightGBM")
    ax.set_ylabel("Spearman IC")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_markdown(config: StrategyConfig, performance: pd.DataFrame, output_path: Path) -> None:
    """Write a compact run summary for report traceability."""

    lines = [
        "# C2O Strategy Reproduction Summary",
        "",
        f"- Start date: `{config.start_date}`",
        f"- Data cutoff: `{config.cutoff}`",
        f"- Development cutoff: `{config.development_cutoff}`",
        f"- Universe size: `{config.universe_size}`",
        f"- Minimum price: `${config.min_price:.2f}`",
        f"- ADV20 floor: `${config.min_adv20:,.0f}`",
        f"- Volatility band: `{config.vol_floor:.0%}` to `{config.vol_cap:.0%}` annualised",
        f"- Earnings exclusion window: `+/- {config.earnings_window_days}` trading day(s)",
        f"- Participation cap: `{config.participation_cap:.1%}` of ADV20",
        f"- Basket fraction: `{config.basket_fraction:.0%}` per side",
        "",
        "## Headline Metrics",
        "",
        performance.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
