import numpy as np
import pandas as pd
import pytest

from portfolio.performance import cagr, compute_returns, sharpe_ratio_per_asset
from portfolio.risk import beta, calmar_ratio, historical_cvar, historical_var, max_drawdown


def test_max_drawdown_known_value():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.6 / 1.2 - 1)


def test_var_and_cvar():
    r = pd.Series(np.arange(-50, 50) / 1000)  # -5 % … +4,9 %
    var = historical_var(r, alpha=0.05)
    cvar = historical_cvar(r, alpha=0.05)
    assert var == pytest.approx(np.percentile(r, 5))
    assert cvar <= var  # la CVaR est toujours au moins aussi mauvaise que la VaR


def test_compute_returns_keeps_days_when_one_market_is_closed():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"US": [100, 101, 102, 103], "FR": [50, np.nan, 51, 52]}, index=idx)
    returns = compute_returns(prices)
    assert len(returns) == 3  # l'ancienne version supprimait le jour férié
    assert returns.loc[idx[1], "FR"] == 0.0


def test_cagr():
    r = pd.Series([0.01] * 252)
    assert cagr(r) == pytest.approx(1.01 ** 252 - 1)


def test_calmar_positive_for_rising_series():
    r = pd.Series([0.01, -0.02, 0.015, 0.01] * 50)
    assert calmar_ratio(r) > 0


def test_beta_of_scaled_series():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 500))
    asset = 1.5 * bench
    assert beta(asset, bench) == pytest.approx(1.5)


def test_sharpe_with_explicit_rf(returns):
    s = sharpe_ratio_per_asset(returns, rf=0.0)
    expected = returns.mean() / returns.std() * np.sqrt(252)
    pd.testing.assert_series_equal(s, expected)
