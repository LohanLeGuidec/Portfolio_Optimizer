"""Mesures de risque : drawdown, VaR, CVaR, Calmar, bêta."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_prices
from .performance import cagr, compute_returns
from .utils import map_tickers_to_benchmarks


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    return equity_curve / equity_curve.cummax() - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    return float(drawdown_series(equity_curve).min())


def historical_var(portfolio_returns, alpha: float = 0.05) -> float:
    """VaR historique : quantile alpha des rendements (valeur négative = perte)."""
    return float(np.percentile(portfolio_returns, 100 * alpha))


def historical_cvar(portfolio_returns, alpha: float = 0.05) -> float:
    """CVaR (Expected Shortfall) : moyenne des rendements sous la VaR."""
    portfolio_returns = pd.Series(portfolio_returns)
    var_level = historical_var(portfolio_returns, alpha)
    return float(portfolio_returns[portfolio_returns <= var_level].mean())


def calmar_ratio(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    equity = (1 + portfolio_returns).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return float(cagr(portfolio_returns, periods_per_year) / abs(mdd))


def portfolio_risks(returns: pd.DataFrame, weights, alpha: float = 0.05) -> dict:
    """Max drawdown, VaR et CVaR d'un portefeuille à poids fixes."""
    port_ret = returns @ np.asarray(weights)
    equity = (1 + port_ret).cumprod()
    return {
        "max_drawdown": max_drawdown(equity),
        "var": historical_var(port_ret, alpha),
        "cvar": historical_cvar(port_ret, alpha),
    }


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def beta_auto_for_all(returns: pd.DataFrame, start_date: str) -> pd.Series:
    """
    Bêta de chaque actif contre l'indice de son marché (déduit de la devise).
    `returns` doit être en devise locale. Chaque indice n'est téléchargé qu'une fois.
    """
    benchmark_map = map_tickers_to_benchmarks(list(returns.columns))
    benchmarks = sorted(set(benchmark_map.values()))
    bench_returns = compute_returns(load_prices(benchmarks, start=start_date))

    return pd.Series({
        ticker: beta(returns[ticker], bench_returns[bench])
        if bench in bench_returns.columns else np.nan
        for ticker, bench in benchmark_map.items()
    })
