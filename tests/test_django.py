"""Tests de l'interface Django (réseau simulé, voir conftest.fake_market)."""
import re

import pytest

PARAMS = {
    "tickers": "AAPL\nMC.PA\nMSFT\nOR.PA",
    "start": "2019-01-01",
    "base_ccy": "EUR",
    "benchmark": "URTH",
    "seed": "42",
    "n_sim": "100",
    "horizon": "63",
    "n_portfolios": "500",
    "wf_window": "252",
    "wf_rebal": "21",
    "wf_cost": "10",
}


@pytest.fixture(autouse=True)
def memory_cache(settings):
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}


def _run_key(html: str) -> str:
    return re.search(r'name="run" value="([a-f0-9]+)"', html).group(1)


def test_home_page_without_analysis(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Lancer l'analyse" in response.content.decode()
    assert "Résultats" not in response.content.decode()


def test_invalid_tickers_show_form_error(client):
    response = client.get("/", {**PARAMS, "tickers": "AAPL"})
    assert response.status_code == 200
    assert "au moins deux tickers" in response.content.decode()


def test_full_analysis_page(client, fake_market):
    response = client.get("/", PARAMS)
    html = response.content.decode()
    assert response.status_code == 200
    assert "Erreur" not in html
    for name in ("Equal Weight", "Min Variance", "Risk Parity", "Max Sharpe"):
        assert name in html
    assert html.count('class="chart"') >= 15
    assert "Performance relative à l&#x27;indice URTH" in html or "Performance relative à l'indice URTH" in html
    assert "Backtest de la VaR historique" in html
    assert "IC 95 % du Sharpe" in html


def test_analysis_without_benchmark(client, fake_market):
    html = client.get("/", {**PARAMS, "benchmark": ""}).content.decode()
    assert "Erreur" not in html
    assert "Performance relative" not in html
    assert "Backtest de la VaR historique" in html


def test_black_litterman_and_exports(client, fake_market):
    html = client.get("/", PARAMS).content.decode()
    run = _run_key(html)

    response = client.post("/black-litterman/", {
        "run": run, "ref": "Equal Weight", "tau": "0.05", "confidence": "0.5",
        "view_a": ["AAPL"], "view_b": ["MSFT"], "view_q": ["3"],
    })
    body = response.content.decode()
    assert response.status_code == 200
    assert "Poids Black-Litterman" in body
    assert "alert-error" not in body

    csv = client.get(f"/export/{run}/weights.csv")
    assert csv.status_code == 200
    assert csv["Content-Type"].startswith("text/csv")
    assert "Max Sharpe" in csv.content.decode()


def test_black_litterman_expired_run(client):
    response = client.post("/black-litterman/", {"run": "inconnu", "view_a": ["A"], "view_b": ["B"], "view_q": ["1"]})
    assert "expiré" in response.content.decode()


def test_plotly_js_is_served(client):
    response = client.get("/plotly.js")
    assert response.status_code == 200
    assert len(response.content) > 1_000_000
