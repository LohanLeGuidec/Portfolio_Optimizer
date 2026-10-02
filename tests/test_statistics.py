import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio.statistics import (
    christoffersen_independence_test,
    kupiec_test,
    relative_metrics,
    rolling_historical_var,
    sharpe_statistics,
    var_backtest,
)


def _normal_returns(n=2000, mu=0.0005, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.bdate_range("2015-01-01", periods=n))


# ── Sharpe ────────────────────────────────────────────────────────────────


def test_sharpe_matches_definition_and_ci_contains_it():
    r = _normal_returns()
    s = sharpe_statistics(r, rf=0.0)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert s["low"] < s["sharpe"] < s["high"]


def test_sharpe_se_matches_lo_formula_for_gaussian_returns():
    # Pour des rendements gaussiens, SE ≈ sqrt((1 + SR²/2) / T) (Lo, 2002)
    r = _normal_returns(n=20_000, seed=3)
    s = sharpe_statistics(r)
    sr = r.mean() / r.std()
    expected = np.sqrt((1 + 0.5 * sr**2) / len(r)) * np.sqrt(252)
    assert s["se"] == pytest.approx(expected, rel=0.05)


def test_ci_shrinks_with_more_data():
    short = sharpe_statistics(_normal_returns(n=500))
    long = sharpe_statistics(_normal_returns(n=5000))
    assert (long["high"] - long["low"]) < (short["high"] - short["low"])


def test_psr_high_for_strong_strategy_and_low_for_losing_one():
    assert sharpe_statistics(_normal_returns(mu=0.002))["psr"] > 0.99
    assert sharpe_statistics(_normal_returns(mu=-0.001))["psr"] < 0.05


# ── VaR ───────────────────────────────────────────────────────────────────


def test_rolling_var_uses_only_past_data():
    r = _normal_returns(n=300)
    var = rolling_historical_var(r, level=0.95, window=250)
    first_day = var.index[0]
    assert first_day == r.index[250]
    assert var.iloc[0] == pytest.approx(r.iloc[:250].quantile(0.05))


def test_kupiec_accepts_exact_frequency_and_rejects_too_many_exceptions():
    assert kupiec_test(50, 1000, 0.95)["p_value"] == pytest.approx(1.0)
    assert kupiec_test(100, 1000, 0.95)["p_value"] < 0.001
    lr = kupiec_test(70, 1000, 0.95)["lr"]
    p, phat = 0.05, 0.07
    expected = -2 * ((930 * np.log(1 - p) + 70 * np.log(p)) - (930 * np.log(1 - phat) + 70 * np.log(phat)))
    assert lr == pytest.approx(expected)


def test_kupiec_with_zero_exceptions():
    res = kupiec_test(0, 500, 0.99)
    assert res["lr"] == pytest.approx(-2 * 500 * np.log(0.99))


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(1)
    independent = rng.random(2000) < 0.05
    clustered = np.zeros(2000, dtype=bool)
    for start in range(0, 2000, 200):
        clustered[start:start + 10] = True  # 10 exceptions consécutives tous les 200 jours
    assert christoffersen_independence_test(independent)["p_value"] > 0.01
    assert christoffersen_independence_test(clustered)["p_value"] < 0.001


def test_var_backtest_on_gaussian_returns_is_not_rejected():
    r = _normal_returns(n=3000, seed=11)
    res = var_backtest(r, level=0.95, window=250)
    assert res["n_obs"] == 2750
    assert abs(res["rate"] - 0.05) < 0.015
    assert not res["rejected"]


def test_var_backtest_rejects_when_volatility_jumps():
    calm = _normal_returns(n=1500, sigma=0.005, seed=5)
    storm = _normal_returns(n=500, sigma=0.03, seed=6)
    r = pd.Series(np.concatenate([calm.values, storm.values]), index=pd.bdate_range("2015-01-01", periods=2000))
    res = var_backtest(r, level=0.99, window=250)
    assert res["rejected"]


# ── Relatif à l'indice ──────────────────────────────────────────────────────


def test_relative_metrics_of_leveraged_benchmark():
    bench = _normal_returns(seed=2)
    port = 1.5 * bench
    m = relative_metrics(port, bench, rf=0.0)
    assert m["beta"] == pytest.approx(1.5)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["correlation"] == pytest.approx(1.0)


def test_relative_metrics_alpha_and_information_ratio():
    bench = _normal_returns(seed=4)
    rng = np.random.default_rng(8)
    noise = pd.Series(rng.normal(0.0002, 0.002, len(bench)), index=bench.index)
    port = bench + noise
    m = relative_metrics(port, bench, rf=0.02)
    assert m["beta"] == pytest.approx(1.0, abs=0.05)
    assert m["alpha"] == pytest.approx(noise.mean() * 252, abs=0.01)
    assert m["tracking_error"] == pytest.approx(noise.std() * np.sqrt(252), rel=1e-6)
    assert m["information_ratio"] == pytest.approx(noise.mean() / noise.std() * np.sqrt(252), rel=1e-6)


def test_chi2_reference():
    # sanity check de la loi utilisée pour les p-values
    assert stats.chi2.sf(3.841, df=1) == pytest.approx(0.05, abs=1e-3)
