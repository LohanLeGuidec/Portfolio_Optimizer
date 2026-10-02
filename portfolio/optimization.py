"""Allocation : Markowitz, risk parity, max Sharpe, Black-Litterman."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from .utils import normalize_weights


def _to_numpy(mat):
    return mat.values if hasattr(mat, "values") else np.asarray(mat)


def _long_only_optimize(objective, n: int, w0=None) -> np.ndarray:
    """Minimise `objective` sous contraintes somme = 1 et 0 <= w <= 1."""
    w0 = np.ones(n) / n if w0 is None else w0
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        warnings.warn(f"Optimisation non convergée : {res.message}", stacklevel=2)
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


# 1) Performance -------------------------------------------------------------

def portfolio_performance(weights, mean_returns, cov_matrix, periods_per_year: int = 252):
    """Rendement et volatilité annualisés à partir de données journalières."""
    w = np.asarray(weights, dtype=float)
    cov = _to_numpy(cov_matrix)
    port_ret = float(w @ _to_numpy(mean_returns)) * periods_per_year
    port_vol = float(np.sqrt(w @ cov @ w * periods_per_year))
    return port_ret, port_vol


# 2) Variance minimale ------------------------------------------------------

def min_variance_portfolio(mean_returns, cov_matrix, long_only: bool = True) -> np.ndarray:
    """
    Portefeuille de variance minimale.
    - long_only=True (défaut) : pas de vente à découvert, comparable aux autres stratégies.
    - long_only=False : solution analytique Σ⁻¹1 / 1'Σ⁻¹1 (poids négatifs possibles).
    """
    cov = _to_numpy(cov_matrix)
    n = cov.shape[0]
    if not long_only:
        ones = np.ones(n)
        inv_cov_ones = np.linalg.solve(cov, ones)
        return inv_cov_ones / (ones @ inv_cov_ones)
    return _long_only_optimize(lambda w: w @ cov @ w, n)


# 3) Risk parity ------------------------------------------------------------

def portfolio_volatility(weights, cov_matrix) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ _to_numpy(cov_matrix) @ w))


def risk_contributions(weights, cov_matrix) -> np.ndarray:
    """Contribution de chaque actif à la volatilité (somme = volatilité totale)."""
    w = np.asarray(weights, dtype=float)
    cov = _to_numpy(cov_matrix)
    return w * (cov @ w) / portfolio_volatility(w, cov)


def risk_parity_objective(weights, cov_matrix) -> float:
    rc = risk_contributions(weights, cov_matrix)
    return float(np.sum((rc - rc.mean()) ** 2))


def risk_parity_weights(cov_matrix, budgets=None) -> np.ndarray:
    """
    Equal Risk Contribution (ou budgets de risque `budgets`), formulation convexe
    de Spinu (2013) :  min  ½ y'Σy − Σ b_i log(y_i),  y > 0,  puis w = y / Σy.
    La solution est unique et toujours long-only, contrairement à l'itération
    multiplicative w ← w · (σ/n) / RC qui peut diverger.
    """
    cov = _to_numpy(cov_matrix)
    n = cov.shape[0]
    b = np.ones(n) / n if budgets is None else normalize_weights(budgets)

    # Mise à l'échelle : ne change pas la solution, améliore le conditionnement
    cov_s = cov / np.mean(np.diag(cov))

    def objective(y):
        return 0.5 * y @ cov_s @ y - b @ np.log(y)

    def gradient(y):
        return cov_s @ y - b / y

    y0 = 1 / np.sqrt(np.diag(cov_s))
    res = minimize(
        objective, y0, jac=gradient, method="L-BFGS-B",
        bounds=[(1e-12, None)] * n, options={"maxiter": 10_000, "gtol": 1e-12, "ftol": 1e-15},
    )
    if not res.success:
        warnings.warn(f"Risk parity non convergé : {res.message}", stacklevel=2)
    return res.x / res.x.sum()


# 4) Max Sharpe -------------------------------------------------------------

def max_sharpe_portfolio(mean_ret, cov_matrix, rf: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Portefeuille tangent long-only (rf annuel, données journalières)."""
    mu = _to_numpy(mean_ret)
    cov = _to_numpy(cov_matrix)

    def neg_sharpe(w):
        ret = w @ mu * periods_per_year
        vol = np.sqrt(w @ cov @ w * periods_per_year)
        return -(ret - rf) / vol

    return _long_only_optimize(neg_sharpe, len(mu))


# 5) Black-Litterman --------------------------------------------------------

def implied_equilibrium_returns(cov_matrix_annual, market_weights, risk_aversion: float = 2.5) -> np.ndarray:
    """Rendements d'équilibre π = δ Σ w_marché (annuels si Σ est annuelle)."""
    return risk_aversion * _to_numpy(cov_matrix_annual) @ np.asarray(market_weights, dtype=float)


def black_litterman_posterior(
    cov_matrix,
    market_weights,
    P,
    Q,
    tau: float = 0.05,
    omega=None,
    risk_aversion: float = 2.5,
    view_confidence: float = 0.5,
    periods_per_year: int = 252,
):
    """
    Rendements et covariance a posteriori, en ANNUEL.

    cov_matrix : covariance des rendements journaliers (annualisée en interne).
    Q : rendements attendus des vues, annuels (ex. 0.03 = 3 %/an).
    view_confidence : entre 0 et 1. 0.5 = Ω de He & Litterman (diag(P τΣ P')) ;
        plus proche de 1 = vues plus influentes, plus proche de 0 = moins influentes.
    """
    sigma = _to_numpy(cov_matrix) * periods_per_year
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).ravel()

    pi = implied_equilibrium_returns(sigma, market_weights, risk_aversion)
    tau_sigma = tau * sigma

    if omega is None:
        c = float(np.clip(view_confidence, 1e-4, 1 - 1e-4))
        omega = np.diag(np.diag(P @ tau_sigma @ P.T)) * (1 - c) / c

    middle = np.linalg.inv(P @ tau_sigma @ P.T + omega)
    mu_bl = pi + tau_sigma @ P.T @ middle @ (Q - P @ pi)
    sigma_bl = sigma + tau_sigma - tau_sigma @ P.T @ middle @ P @ tau_sigma
    return mu_bl, sigma_bl


def black_litterman_weights(
    cov_matrix,
    market_weights,
    P,
    Q,
    tau: float = 0.05,
    omega=None,
    risk_aversion: float = 2.5,
    view_confidence: float = 0.5,
    periods_per_year: int = 252,
):
    """
    Poids optimaux long-only : max  w'μ_BL − δ/2 · w'Σ_BL w  avec somme = 1, w >= 0.
    Renvoie (poids, μ_BL annuel, Σ_BL annuelle).
    """
    mu_bl, sigma_bl = black_litterman_posterior(
        cov_matrix, market_weights, P, Q, tau, omega, risk_aversion, view_confidence, periods_per_year
    )

    def neg_utility(w):
        return -(w @ mu_bl - 0.5 * risk_aversion * w @ sigma_bl @ w)

    w_bl = _long_only_optimize(neg_utility, len(mu_bl), w0=normalize_weights(market_weights))
    return w_bl, mu_bl, sigma_bl


# 6) Frontière efficiente (tirages aléatoires) -----------------------------

def efficient_frontier_random(
    mean_returns,
    cov_matrix,
    n_portfolios: int = 5000,
    periods_per_year: int = 252,
    seed: int | None = 42,
    rf: float = 0.0,
):
    """Portefeuilles long-only tirés uniformément sur le simplexe (Dirichlet)."""
    rng = np.random.default_rng(seed)
    mu = _to_numpy(mean_returns)
    cov = _to_numpy(cov_matrix)

    weights = rng.dirichlet(np.ones(len(mu)), size=n_portfolios)
    rets = weights @ mu * periods_per_year
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights) * periods_per_year)
    sharpes = (rets - rf) / vols
    return rets, vols, weights, sharpes
