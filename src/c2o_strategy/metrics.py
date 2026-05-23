"""Performance and audit metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(252.0))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualised_volatility(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))


def sharpe_ratio_lo_corrected(returns: pd.Series, max_lag: int = 5) -> float:
    """Lo (2002) autocorrelation-corrected Sharpe ratio.

    Adjusts the annualised Sharpe for serial correlation in daily returns
    up to ``max_lag`` lags. See Lo, "The Statistics of Sharpe Ratios",
    Financial Analysts Journal, 2002.
    """
    returns = returns.dropna()
    n = len(returns)
    if n < max_lag + 10:
        return np.nan
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma == 0:
        return np.nan

    # Compute autocorrelation correction factor
    rho_sum = 0.0
    centred = (returns - mu).values
    var = float(np.dot(centred, centred) / n)
    if var == 0:
        return np.nan
    for k in range(1, max_lag + 1):
        rho_k = float(np.dot(centred[k:], centred[:-k]) / (n * var))
        rho_sum += (max_lag + 1 - k) * rho_k

    # eta(q) = q * (1 + 2 * sum_k (1 - k/q) * rho_k)  where q = 252
    q = 252.0
    eta = q * (1.0 + 2.0 * rho_sum)
    if eta <= 0:
        return np.nan

    sr_annual = mu / sigma * np.sqrt(eta)
    return float(sr_annual)


def top_pct_concentration(returns: pd.Series, pct: float = 0.05) -> float:
    """Fraction of total cumulative return from the top ``pct`` days."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float(returns.sum())
    if total == 0:
        return np.nan
    n_top = max(1, int(np.ceil(len(returns) * pct)))
    top_sum = float(returns.nlargest(n_top).sum())
    return top_sum / total


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.fillna(0.0)
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def summarise_backtest(daily: pd.DataFrame, aum: float) -> dict[str, float]:
    gross = daily["gross_return"]
    after_commission = gross - daily["commission_cost"]
    after_slippage = after_commission - daily["slippage_cost"]
    net = daily["net_return"]

    gross_sharpe = sharpe_ratio(gross)
    after_commission_sharpe = sharpe_ratio(after_commission)
    after_slippage_sharpe = sharpe_ratio(after_slippage)
    net_sharpe = sharpe_ratio(net)

    return {
        "aum": aum,
        "net_annualised_return": annualised_return(net),
        "net_annualised_volatility": annualised_volatility(net),
        "net_sharpe": net_sharpe,
        "max_drawdown": max_drawdown(net),
        "daily_turnover": float(daily["turnover"].mean()),
        "average_gross_exposure": float(daily["gross_exposure"].mean()),
        "fraction_days_full_gross_exposure": float(daily["full_gross_exposure"].mean()),
        "fraction_days_cap_binding": float(daily["cap_binding"].mean()),
        "gross_sharpe": gross_sharpe,
        "commission_sharpe_drag": gross_sharpe - after_commission_sharpe,
        "slippage_sharpe_drag": after_commission_sharpe - after_slippage_sharpe,
        "borrow_sharpe_drag": after_slippage_sharpe - net_sharpe,
        "avg_commission_bps_per_day": float(daily["commission_cost"].mean() * 10_000),
        "avg_slippage_bps_per_day": float(daily["slippage_cost"].mean() * 10_000),
        "avg_borrow_bps_per_day": float(daily["borrow_cost"].mean() * 10_000),
        "days": int(len(daily)),
    }


def stress_window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "late_2018": ("2018-10-01", "2018-12-31"),
        "covid_q1_2020": ("2020-01-01", "2020-03-31"),
        "drawdown_2022": ("2022-01-01", "2022-12-31"),
    }
    rows = []
    for name, (start, end) in windows.items():
        chunk = daily.loc[pd.to_datetime(daily["date"]).between(start, end)]
        if chunk.empty:
            continue
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "net_return": float((1.0 + chunk["net_return"]).prod() - 1.0),
                "net_sharpe": sharpe_ratio(chunk["net_return"]),
                "max_drawdown": max_drawdown(chunk["net_return"]),
                "avg_gross_exposure": float(chunk["gross_exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)
