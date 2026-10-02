"""
Analyse complète d'un univers d'actifs, indépendante de l'interface.

Utilisée à la fois par l'application Django (optimizer/) et par la version
Streamlit (streamlit_app.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .backtest import walk_forward_backtest
from .data_loader import convert_prices_to_currency, load_prices
from .optimization import (
    efficient_frontier_random,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_performance,
    risk_parity_weights,
)
from .performance import annualized_return, annualized_volatility, compute_returns, sharpe_ratio_per_asset
from .risk import beta_auto_for_all, calmar_ratio, drawdown_series, historical_cvar, historical_var, max_drawdown
from .simulation import monte_carlo_simulation, simulate_portfolio_values
from .statistics import relative_metrics, sharpe_statistics, var_backtest
from .utils import get_currency_from_ticker, get_risk_free_rate, seed_everything

LOCAL_CCY = "LOCAL"  # pas de conversion : chaque actif reste dans sa devise
DEFAULT_BENCHMARK = "URTH"  # ETF iShares MSCI World (dividendes inclus via les prix ajustés)
VAR_LEVELS = (0.95, 0.99)
VAR_WINDOW = 250

STRATEGIES = {
    "Equal Weight": "equal",
    "Min Variance": "min_variance",
    "Risk Parity": "risk_parity",
    "Max Sharpe": "max_sharpe",
}


def run_analysis(
    tickers,
    start: str,
    base_ccy: str = "EUR",
    seed: int = 42,
    n_sim: int = 500,
    horizon: int = 252,
    n_portfolios: int = 2000,
    wf_window: int = 252,
    wf_rebal: int = 21,
    wf_cost: float = 10.0,
    benchmark: str | None = DEFAULT_BENCHMARK,
) -> dict:
    """
    Télécharge les prix, optimise les 4 stratégies, calcule risques, frontière,
    Monte Carlo et backtest walk-forward. Renvoie un dictionnaire de résultats
    (DataFrames, Series, tableaux numpy) prêt à être affiché.
    """
    tickers = list(tickers)
    if len(tickers) < 2:
        raise ValueError("Indiquez au moins deux tickers.")

    seed_everything(seed)
    prices_local = load_prices(tickers, start=start)

    missing = [t for t in tickers if t not in prices_local.columns or prices_local[t].isna().all()]
    if missing:
        raise ValueError(f"Tickers introuvables ou sans données : {', '.join(missing)}")

    prices = prices_local if base_ccy == LOCAL_CCY else convert_prices_to_currency(prices_local, base_ccy)
    returns = compute_returns(prices)
    returns_local = compute_returns(prices_local)

    if returns.shape[0] < 2:
        raise ValueError("Pas assez de données sur la période sélectionnée.")

    # Indice de référence, converti dans la même devise que le portefeuille
    bench_returns, bench_error = None, None
    if benchmark:
        try:
            bench_px = load_prices([benchmark], start=start)
            if benchmark not in bench_px.columns or bench_px[benchmark].isna().all():
                raise ValueError("aucune donnée")
            if base_ccy != LOCAL_CCY:
                bench_px = convert_prices_to_currency(bench_px, base_ccy)
            bench_px = bench_px[benchmark].reindex(prices.index.union(bench_px.index)).ffill()
            bench_returns = bench_px.pct_change().reindex(returns.index).dropna()
            if len(bench_returns) < 30:
                raise ValueError("historique trop court")
        except Exception as exc:  # noqa: BLE001 — l'analyse continue sans indice
            bench_returns, bench_error = None, f"Indice {benchmark} indisponible : {exc}"

    period_start = str(returns.index[0].date())
    period_end = str(returns.index[-1].date())

    # Taux sans risque moyen sur la période analysée
    if base_ccy == LOCAL_CCY:
        rf_assets = pd.Series({
            t: get_risk_free_rate(get_currency_from_ticker(t), period_start, period_end) for t in tickers
        })
        rf = float(rf_assets.mean())
    else:
        rf = get_risk_free_rate(base_ccy, period_start, period_end)
        rf_assets = pd.Series(rf, index=tickers)

    mean_ret = returns.mean()
    cov = LedoitWolf().fit(returns.values).covariance_

    n = len(tickers)
    portfolios = {
        "Equal Weight": np.ones(n) / n,
        "Min Variance": min_variance_portfolio(mean_ret, cov),
        "Risk Parity": risk_parity_weights(cov),
        "Max Sharpe": max_sharpe_portfolio(mean_ret, cov, rf=rf),
    }

    port_returns = {name: returns @ w for name, w in portfolios.items()}
    equity_curves = {name: (1 + pr).cumprod() for name, pr in port_returns.items()}
    drawdowns = {name: drawdown_series(eq) for name, eq in equity_curves.items()}

    # Tableau de synthèse par portefeuille (valeurs numériques, formatage côté interface)
    summary_rows = {}
    for name, w in portfolios.items():
        ret_p, vol_p = portfolio_performance(w, mean_ret, cov)
        pr = port_returns[name]
        sh = sharpe_statistics(pr, rf)
        summary_rows[name] = {
            "sharpe_low": sh["low"],
            "sharpe_high": sh["high"],
            "psr": sh["psr"],
            "return": ret_p,
            "volatility": vol_p,
            "sharpe": (ret_p - rf) / vol_p if vol_p else np.nan,
            "max_drawdown": max_drawdown(equity_curves[name]),
            "var95": historical_var(pr),
            "cvar95": historical_cvar(pr),
            "calmar": calmar_ratio(pr),
        }
    summary = pd.DataFrame(summary_rows).T

    # Performance relative à l'indice (in-sample)
    benchmark_stats, relative = None, pd.DataFrame()
    if bench_returns is not None:
        bench_equity = (1 + bench_returns).cumprod()
        sh = sharpe_statistics(bench_returns, rf)
        benchmark_stats = {
            "return": annualized_return(bench_returns),
            "volatility": annualized_volatility(bench_returns),
            "sharpe": sh["sharpe"], "sharpe_low": sh["low"], "sharpe_high": sh["high"], "psr": sh["psr"],
            "max_drawdown": max_drawdown(bench_equity),
            "equity": bench_equity,
            "drawdown": drawdown_series(bench_equity),
        }
        relative = pd.DataFrame({
            name: relative_metrics(pr, bench_returns, rf) for name, pr in port_returns.items()
        }).T

    # Backtest de la VaR historique glissante (prévision à J+1 sur 250 jours passés)
    var_tests = {}
    if len(returns) > VAR_WINDOW + 30:
        for name, pr in port_returns.items():
            var_tests[name] = {level: var_backtest(pr, level, VAR_WINDOW) for level in VAR_LEVELS}

    rets_arr, vols_arr, _, sharpes_arr = efficient_frontier_random(
        mean_ret, cov, n_portfolios=n_portfolios, rf=rf, seed=seed
    )

    # Monte Carlo : on ne garde que les statistiques (les trajectoires complètes
    # pèsent plusieurs dizaines de Mo)
    sims = monte_carlo_simulation(returns, n_sim=n_sim, horizon=horizon, seed=seed)
    monte_carlo = {}
    for name, w in portfolios.items():
        values = simulate_portfolio_values(sims, w)
        final = values[:, -1]
        monte_carlo[name] = {
            "mean": values.mean(axis=0),
            "p10": np.percentile(values, 10, axis=0),
            "p90": np.percentile(values, 90, axis=0),
            "final": final,
            "median_final": float(np.median(final)),
            "p10_final": float(np.percentile(final, 10)),
            "p90_final": float(np.percentile(final, 90)),
            "prob_loss": float((final < 1).mean()),
            "prob_gain50": float((final > 1.5).mean()),
        }
    del sims

    # Bêta en devise locale contre l'indice du marché de cotation
    betas = beta_auto_for_all(returns_local, start)

    # Backtest walk-forward (out-of-sample)
    wf_returns, wf_errors = {}, {}
    for name, method in STRATEGIES.items():
        try:
            wf_returns[name] = walk_forward_backtest(
                returns, window=wf_window, rebalance_freq=wf_rebal, method=method, rf=rf, cost_bps=wf_cost,
            )
        except Exception as exc:  # noqa: BLE001 — on affiche l'erreur sans bloquer les autres
            wf_returns[name] = pd.Series(dtype=float)
            wf_errors[name] = str(exc)
    wf_returns = {k: v for k, v in wf_returns.items() if not v.empty}
    wf_equity_curves = {name: (1 + r).cumprod() for name, r in wf_returns.items()}

    wf_rows, wf_relative = {}, {}
    for name, r in wf_returns.items():
        is_ret = (1 + port_returns[name].loc[r.index]).prod() - 1  # mêmes dates que l'OOS
        sh = sharpe_statistics(r, rf)
        if bench_returns is not None:
            wf_relative[name] = relative_metrics(r, bench_returns, rf)
        wf_rows[name] = {
            "sharpe": sh["sharpe"],
            "sharpe_low": sh["low"],
            "sharpe_high": sh["high"],
            "psr": sh["psr"],
            "total_is": is_ret,
            "total_oos": wf_equity_curves[name].iloc[-1] - 1,
            "return": annualized_return(r),
            "volatility": annualized_volatility(r),
            "max_drawdown": max_drawdown(wf_equity_curves[name]),
            "var95": historical_var(r),
            "cvar95": historical_cvar(r),
        }
    wf_summary = pd.DataFrame(wf_rows).T

    wf_benchmark = None
    if bench_returns is not None and wf_returns:
        oos_index = next(iter(wf_returns.values())).index
        b = bench_returns.reindex(oos_index).fillna(0.0)
        b_eq = (1 + b).cumprod()
        sh = sharpe_statistics(b, rf)
        wf_benchmark = {
            "total": b_eq.iloc[-1] - 1, "return": annualized_return(b), "volatility": annualized_volatility(b),
            "sharpe": sh["sharpe"], "sharpe_low": sh["low"], "sharpe_high": sh["high"], "psr": sh["psr"],
            "max_drawdown": max_drawdown(b_eq), "equity": b_eq,
        }

    return dict(
        params=dict(
            tickers=tickers, start=start, base_ccy=base_ccy, seed=seed, n_sim=n_sim, horizon=horizon,
            n_portfolios=n_portfolios, wf_window=wf_window, wf_rebal=wf_rebal, wf_cost=wf_cost,
            benchmark=benchmark,
        ),
        benchmark=benchmark if bench_returns is not None else None,
        benchmark_error=bench_error,
        benchmark_returns=bench_returns,
        benchmark_stats=benchmark_stats,
        relative=relative,
        var_tests=var_tests,
        wf_relative=pd.DataFrame(wf_relative).T,
        wf_benchmark=wf_benchmark,
        tickers=list(returns.columns),
        period=(returns.index[0], returns.index[-1]),
        prices=prices,
        returns=returns,
        mean_ret=mean_ret,
        cov=cov,
        rf=rf,
        portfolios=portfolios,
        port_returns=port_returns,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
        summary=summary,
        asset_metrics=pd.DataFrame({
            "return": annualized_return(returns),
            "volatility": annualized_volatility(returns),
            "sharpe": sharpe_ratio_per_asset(returns, rf=rf_assets),
            "beta": betas,
        }),
        frontier=dict(returns=rets_arr, vols=vols_arr, sharpes=sharpes_arr),
        monte_carlo=monte_carlo,
        wf_returns=wf_returns,
        wf_equity_curves=wf_equity_curves,
        wf_summary=wf_summary,
        wf_errors=wf_errors,
    )
