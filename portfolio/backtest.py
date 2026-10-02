"""Backtest walk-forward (out-of-sample)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .optimization import max_sharpe_portfolio, min_variance_portfolio, risk_parity_weights


def _target_weights(train: pd.DataFrame, method: str, rf: float) -> np.ndarray:
    n = train.shape[1]
    if method == "equal":
        return np.ones(n) / n
    mean_ret = train.mean()
    cov = LedoitWolf().fit(train.values).covariance_
    if method == "max_sharpe":
        return max_sharpe_portfolio(mean_ret, cov, rf=rf)
    if method == "min_variance":
        return min_variance_portfolio(mean_ret, cov)
    if method == "risk_parity":
        return risk_parity_weights(cov)
    raise ValueError(f"Méthode inconnue : {method}")


def walk_forward_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    rebalance_freq: int = 21,
    method: str = "max_sharpe",
    rf: float = 0.0,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Backtest walk-forward : à chaque date de rebalancement, les poids cibles sont
    estimés sur les `window` derniers jours uniquement (pas de look-ahead bias).

    Entre deux rebalancements, les poids dérivent avec les prix (buy-and-hold).
    À chaque rebalancement, un coût de `cost_bps` points de base est appliqué
    sur le turnover (somme des |variations de poids|).

    Renvoie les rendements journaliers nets du portefeuille.
    """
    if len(returns) <= window:
        return pd.Series(dtype=float)

    cost = cost_bps / 10_000
    values = returns.values
    n_assets = values.shape[1]

    w_current = np.zeros(n_assets)  # pas de position avant le premier rebalancement
    out = np.empty(len(returns) - window)

    for i, t in enumerate(range(window, len(returns))):
        trade_cost = 0.0
        if (t - window) % rebalance_freq == 0:
            w_target = _target_weights(returns.iloc[t - window:t], method, rf)
            turnover = np.abs(w_target - w_current).sum()
            trade_cost = turnover * cost
            w_current = w_target

        r_t = values[t]
        gross = float(w_current @ r_t)
        out[i] = (1 - trade_cost) * (1 + gross) - 1

        # Dérive des poids avec les prix
        w_current = w_current * (1 + r_t) / (1 + gross)

    return pd.Series(out, index=returns.index[window:], name=method)
