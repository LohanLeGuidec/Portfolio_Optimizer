"""Simulation Monte Carlo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def monte_carlo_simulation(
    returns: pd.DataFrame,
    n_sim: int = 1000,
    horizon: int = 252,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Rendements journaliers simulés selon une loi normale multivariée
    (moyennes historiques, covariance Ledoit-Wolf).
    Renvoie un tableau (n_sim, horizon, n_actifs).
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    sigma = LedoitWolf().fit(returns.values).covariance_
    return rng.multivariate_normal(mu, sigma, size=(n_sim, horizon))


def simulate_portfolio_values(sims: np.ndarray, weights) -> np.ndarray:
    """Valeur (base 1) de chaque trajectoire simulée : tableau (n_sim, horizon)."""
    return np.cumprod(1 + sims @ np.asarray(weights, dtype=float), axis=1)


def monte_carlo_portfolio_paths(sims: np.ndarray, portfolios: dict) -> pd.DataFrame:
    """Trajectoire moyenne pour chaque portefeuille {nom: poids}."""
    return pd.DataFrame({
        name: simulate_portfolio_values(sims, w).mean(axis=0) for name, w in portfolios.items()
    })
