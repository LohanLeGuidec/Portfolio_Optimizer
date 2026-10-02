import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns() -> pd.DataFrame:
    """Rendements journaliers synthétiques (4 actifs, ~3 ans), sans réseau."""
    rng = np.random.default_rng(0)
    n_days, n_assets = 750, 4
    vols = np.array([0.010, 0.015, 0.020, 0.025])
    corr = np.full((n_assets, n_assets), 0.3) + 0.7 * np.eye(n_assets)
    cov = corr * np.outer(vols, vols)
    data = rng.multivariate_normal(np.full(n_assets, 0.0004), cov, size=n_days)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(data, index=index, columns=["A", "B", "C", "D"])


@pytest.fixture
def cov(returns) -> np.ndarray:
    return returns.cov().values


def _fake_download(tickers, start=None, end=None, **kwargs):
    """Remplace yfinance.download : marche aléatoire déterministe par ticker."""
    import zlib

    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    idx = pd.bdate_range(pd.Timestamp(start) if start is not None else "2015-01-01", "2024-12-31")
    cols = {}
    for t in tickers:
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        fx = t.endswith("=X")
        p = (1.1 if fx else 100) * np.exp(np.cumsum(rng.normal(0.0003, 0.005 if fx else 0.015, len(idx))))
        s = pd.Series(p, index=idx)
        if t.endswith(".PA"):  # quelques jours fériés propres à Paris
            s.iloc[rng.choice(len(idx), 20, replace=False)] = np.nan
        cols[("Close", t)] = s
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _fake_fred(name, source, start=None, end=None):
    idx = pd.date_range("2015-01-01", "2024-12-31", freq="MS")
    return pd.DataFrame({name: np.linspace(0.5, 3.5, len(idx))}, index=idx)


@pytest.fixture
def fake_market(monkeypatch):
    """Coupe tout accès réseau : Yahoo Finance et FRED sont simulés."""
    import pandas_datareader.data as web
    import yfinance

    from portfolio.utils import get_risk_free_rate

    monkeypatch.setattr(yfinance, "download", _fake_download)
    monkeypatch.setattr(web, "DataReader", _fake_fred)
    get_risk_free_rate.cache_clear()
    yield
    get_risk_free_rate.cache_clear()
