"""Utilitaires : poids, graines aléatoires, devises, benchmarks et taux sans risque."""
from __future__ import annotations

import random
import warnings
from functools import lru_cache

import numpy as np

# Taux utilisé si le taux sans risque ne peut pas être récupéré (pas de réseau,
# série FRED indisponible, devise non couverte).
DEFAULT_RISK_FREE_RATE = 0.03

# Suffixes Yahoo Finance → devise de cotation.
# Un ticker sans suffixe (AAPL, MSFT…) est considéré comme coté en USD.
SUFFIX_TO_CURRENCY: dict[str, str] = {
    # Zone euro
    "PA": "EUR",  # Euronext Paris
    "DE": "EUR",  # Xetra
    "F": "EUR",   # Francfort
    "AS": "EUR",  # Euronext Amsterdam
    "BR": "EUR",  # Euronext Bruxelles
    "MI": "EUR",  # Milan
    "MC": "EUR",  # Madrid
    "LS": "EUR",  # Lisbonne
    "VI": "EUR",  # Vienne
    "HE": "EUR",  # Helsinki
    "IR": "EUR",  # Dublin
    # Autres
    "L": "GBP",   # Londres (prix cotés en pence, GBp)
    "SW": "CHF",  # SIX Swiss Exchange
    "T": "JPY",   # Tokyo
    "TO": "CAD",  # Toronto
    "V": "CAD",   # TSX Venture
    "AX": "AUD",  # Australie
    "NS": "INR",  # NSE Inde
    "BO": "INR",  # BSE Inde
    "SS": "CNY",  # Shanghai
    "SZ": "CNY",  # Shenzhen
    "HK": "HKD",  # Hong Kong
    "SA": "BRL",  # São Paulo
}

CURRENCY_TO_BENCHMARK: dict[str, str] = {
    "USD": "^GSPC",      # S&P 500
    "EUR": "^STOXX50E",  # Euro Stoxx 50
    "GBP": "^FTSE",      # FTSE 100
    "CHF": "^SSMI",      # Swiss Market Index
    "JPY": "^N225",      # Nikkei 225
    "CAD": "^GSPTSE",    # S&P/TSX Composite
    "AUD": "^AXJO",      # S&P/ASX 200
    "INR": "^NSEI",      # Nifty 50
    "CNY": "000001.SS",  # Shanghai Composite
    "HKD": "^HSI",       # Hang Seng
    "BRL": "^BVSP",      # Bovespa
}

# Séries FRED de taux courts (3 mois) par devise.
# DGS3MO est quotidienne ; les séries OCDE IR3TIB01… sont mensuelles.
CURRENCY_TO_FRED_RATE: dict[str, str] = {
    "USD": "DGS3MO",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "CHF": "IR3TIB01CHM156N",
    "JPY": "IR3TIB01JPM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
    "INR": "IR3TIB01INM156N",
    "CNY": "IR3TIB01CNM156N",
}


def normalize_weights(weights) -> np.ndarray:
    """Normalise un vecteur de poids pour qu'il somme à 1."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if np.isclose(total, 0.0):
        raise ValueError("La somme des poids est nulle.")
    return weights / total


def seed_everything(seed: int = 42) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)


def get_currency_from_ticker(ticker: str) -> str:
    """Devise de cotation d'un ticker Yahoo Finance, d'après son suffixe."""
    if "." not in ticker:
        return "USD"
    suffix = ticker.rsplit(".", 1)[1].upper()
    return SUFFIX_TO_CURRENCY.get(suffix, "USD")


def is_quoted_in_pence(ticker: str) -> bool:
    """Les actions de Londres (.L) sont cotées en pence (GBp), pas en livres."""
    return ticker.upper().endswith(".L")


def get_benchmark_ticker_from_currency(currency: str) -> str:
    try:
        return CURRENCY_TO_BENCHMARK[currency]
    except KeyError:
        raise ValueError(f"Devise non supportée : {currency}") from None


def map_tickers_to_benchmarks(tickers: list[str]) -> dict[str, str]:
    return {t: get_benchmark_ticker_from_currency(get_currency_from_ticker(t)) for t in tickers}


@lru_cache(maxsize=64)
def get_risk_free_rate(currency: str, start: str | None = None, end: str | None = None) -> float:
    """
    Taux sans risque annuel (en décimal) pour une devise.

    - Avec `start` : moyenne du taux sur la période [start, end], cohérente avec
      des rendements calculés sur cette même période.
    - Sans `start` : dernière valeur disponible.

    Le résultat est mis en cache : un seul appel réseau par (devise, période).
    En cas d'échec, renvoie DEFAULT_RISK_FREE_RATE avec un avertissement.
    """
    series_id = CURRENCY_TO_FRED_RATE.get(currency)
    if series_id is None:
        warnings.warn(
            f"Pas de taux sans risque FRED pour {currency} : "
            f"utilisation de {DEFAULT_RISK_FREE_RATE:.1%}.",
            stacklevel=2,
        )
        return DEFAULT_RISK_FREE_RATE

    try:
        import pandas_datareader.data as web

        data = web.DataReader(series_id, "fred", start=start, end=end).iloc[:, 0].dropna()
        if data.empty:
            raise ValueError("série vide")
        value = data.mean() if start is not None else data.iloc[-1]
        return float(value) / 100
    except Exception as exc:  # réseau, série retirée, etc.
        warnings.warn(
            f"Taux sans risque {currency} ({series_id}) indisponible ({exc}) : "
            f"utilisation de {DEFAULT_RISK_FREE_RATE:.1%}.",
            stacklevel=2,
        )
        return DEFAULT_RISK_FREE_RATE


def get_risk_free_rate_from_ticker(ticker: str, start: str | None = None, end: str | None = None) -> float:
    return get_risk_free_rate(get_currency_from_ticker(ticker), start, end)
