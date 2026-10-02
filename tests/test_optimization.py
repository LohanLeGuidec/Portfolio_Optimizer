import numpy as np
import pytest

from portfolio.backtest import walk_forward_backtest
from portfolio.optimization import (
    black_litterman_posterior,
    black_litterman_weights,
    efficient_frontier_random,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
)
from portfolio.simulation import monte_carlo_simulation, simulate_portfolio_values


def _is_long_only(w):
    return np.all(w >= -1e-9) and np.isclose(w.sum(), 1.0)


def test_min_variance_is_long_only_and_beats_random(returns, cov):
    w = min_variance_portfolio(returns.mean(), cov)
    assert _is_long_only(w)
    _, vols, _, _ = efficient_frontier_random(returns.mean(), cov, n_portfolios=2000)
    assert portfolio_volatility(w, cov) * np.sqrt(252) <= vols.min() + 1e-9


def test_min_variance_analytic_matches_closed_form(cov):
    w = min_variance_portfolio(None, cov, long_only=False)
    inv = np.linalg.inv(cov)
    expected = inv.sum(axis=1) / inv.sum()
    np.testing.assert_allclose(w, expected)


def test_risk_parity_equalizes_contributions(cov):
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    assert _is_long_only(w)
    np.testing.assert_allclose(rc, rc.mean(), rtol=1e-6)


def test_max_sharpe_beats_random_portfolios(returns, cov):
    mu = returns.mean()
    w = max_sharpe_portfolio(mu, cov, rf=0.0)
    assert _is_long_only(w)
    _, _, _, sharpes = efficient_frontier_random(mu, cov, n_portfolios=2000)
    best = (w @ mu * 252) / (portfolio_volatility(w, cov) * np.sqrt(252))
    assert best >= sharpes.max() - 1e-6


def test_black_litterman_without_strong_view_stays_near_market(cov):
    w_mkt = np.array([0.25, 0.25, 0.25, 0.25])
    P = np.array([[1, -1, 0, 0]])
    # Vue égale à l'écart d'équilibre → le postérieur doit rester l'équilibre
    pi = 2.5 * (cov * 252) @ w_mkt
    Q = P @ pi
    mu_bl, _ = black_litterman_posterior(cov, w_mkt, P, Q)
    np.testing.assert_allclose(mu_bl, pi, atol=1e-12)


def test_black_litterman_view_scale_is_annual(cov):
    """Une vue de 3 %/an ne doit pas produire des rendements de plusieurs centaines de %."""
    w_mkt = np.ones(4) / 4
    P = np.array([[1, -1, 0, 0]])
    w, mu_bl, _ = black_litterman_weights(cov, w_mkt, P, np.array([0.03]))
    assert _is_long_only(w)
    assert np.all(np.abs(mu_bl) < 0.5)
    assert mu_bl[0] - mu_bl[1] < 0.03 + 1e-9  # le postérieur est un compromis


def test_black_litterman_confidence_moves_weights(cov):
    w_mkt = np.ones(4) / 4
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.10])
    w_low, _, _ = black_litterman_weights(cov, w_mkt, P, Q, view_confidence=0.1)
    w_high, _, _ = black_litterman_weights(cov, w_mkt, P, Q, view_confidence=0.9)
    assert w_high[0] - w_high[1] > w_low[0] - w_low[1]


@pytest.mark.parametrize("method", ["equal", "min_variance", "risk_parity", "max_sharpe"])
def test_walk_forward_backtest_runs(returns, method):
    r = walk_forward_backtest(returns, window=252, rebalance_freq=21, method=method)
    assert len(r) == len(returns) - 252
    assert r.index[0] == returns.index[252]
    assert np.isfinite(r).all()


def test_transaction_costs_reduce_performance(returns):
    gross = walk_forward_backtest(returns, method="min_variance", cost_bps=0)
    net = walk_forward_backtest(returns, method="min_variance", cost_bps=50)
    assert (1 + net).prod() < (1 + gross).prod()


def test_monte_carlo_shapes_and_reproducibility(returns):
    s1 = monte_carlo_simulation(returns, n_sim=50, horizon=30, seed=1)
    s2 = monte_carlo_simulation(returns, n_sim=50, horizon=30, seed=1)
    assert s1.shape == (50, 30, 4)
    np.testing.assert_array_equal(s1, s2)
    values = simulate_portfolio_values(s1, np.ones(4) / 4)
    assert values.shape == (50, 30)


def test_risk_parity_stable_on_near_identical_assets():
    """Régression : l'ancienne itération multiplicative divergeait dans ce cas
    (poids négatifs, somme des contributions incohérente)."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0003, 0.015, size=(2000, 7))
    cov = np.cov(r, rowvar=False)
    w = risk_parity_weights(cov)
    assert _is_long_only(w)
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc, rc.mean(), rtol=1e-6)


def test_efficient_frontier_dominates_random_portfolios(returns, cov):
    from portfolio.optimization import efficient_frontier_curve

    mu = returns.mean()
    c_rets, c_vols = efficient_frontier_curve(mu, cov, n_points=25)
    r_rets, r_vols, _, _ = efficient_frontier_random(mu, cov, n_portfolios=2000)
    assert len(c_rets) >= 20
    assert np.all(np.diff(c_rets) > 0)  # rendement croissant le long de la courbe
    # Aucun portefeuille aléatoire n'est au-dessus de la frontière à volatilité égale
    # Tolérance de 1 pb : l'interpolation linéaire passe légèrement sous une courbe concave
    frontier_ret_at = np.interp(r_vols, c_vols, c_rets, left=np.nan, right=np.nan)
    ok = ~np.isnan(frontier_ret_at)
    assert np.all(r_rets[ok] <= frontier_ret_at[ok] + 1e-4)
    # Extrémités exactes : variance minimale au début, actif le plus rentable seul à la fin
    w_mv = min_variance_portfolio(mu, cov)
    assert c_vols[0] == pytest.approx(np.sqrt(w_mv @ cov @ w_mv * 252))
    best = np.argmax(mu.values)
    assert c_rets[-1] == pytest.approx(mu.values[best] * 252)
    assert c_vols[-1] == pytest.approx(np.sqrt(cov[best, best] * 252))


def test_max_sharpe_is_the_tangency_point(returns, cov):
    """La Capital Market Line touche la frontière au portefeuille Max Sharpe :
    aucun point de la frontière n'a un meilleur Sharpe."""
    from portfolio.optimization import efficient_frontier_curve

    mu, rf = returns.mean(), 0.02
    w = max_sharpe_portfolio(mu, cov, rf=rf)
    ms_ret = w @ mu * 252
    ms_vol = np.sqrt(w @ cov @ w * 252)
    c_rets, c_vols = efficient_frontier_curve(mu, cov)
    assert (ms_ret - rf) / ms_vol >= ((c_rets - rf) / c_vols).max() - 1e-6
    assert ms_vol == pytest.approx(np.interp(ms_ret, c_rets, c_vols), rel=1e-3)
