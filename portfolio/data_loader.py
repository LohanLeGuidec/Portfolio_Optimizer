"""Téléchargement des prix et conversion dans une devise de référence."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from .utils import get_currency_from_ticker, is_quoted_in_pence


def load_prices(tickers, start=None, end=None) -> pd.DataFrame:
    """
    Prix de clôture ajustés (dividendes et splits) depuis Yahoo Finance.
    Renvoie toujours un DataFrame avec une colonne par ticker.
    """
    tickers = list(tickers)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data is None or data.empty:
        raise ValueError("Aucune donnée téléchargée depuis Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
        field = "Close" if "Close" in fields else "Adj Close"
        prices = data[field]
    else:
        field = "Close" if "Close" in data.columns else "Adj Close"
        prices = data[[field]].rename(columns={field: tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Garde l'ordre demandé par l'utilisateur
    return prices.reindex(columns=[t for t in tickers if t in prices.columns])


def convert_prices_to_currency(prices: pd.DataFrame, base_currency: str) -> pd.DataFrame:
    """
    Convertit chaque colonne de prix dans `base_currency` avec les taux de change
    Yahoo (paires du type EURUSD=X). Les prix de Londres sont d'abord passés de
    pence en livres.
    """
    converted = prices.copy()

    for ticker in converted.columns:
        if is_quoted_in_pence(ticker):
            converted[ticker] = converted[ticker] / 100

    currencies = {t: get_currency_from_ticker(t) for t in converted.columns}
    needed = sorted({c for c in currencies.values() if c != base_currency})
    if not needed:
        return converted

    pairs = {c: f"{c}{base_currency}=X" for c in needed}
    fx = load_prices(list(pairs.values()), start=converted.index.min(), end=None)
    fx = fx.reindex(converted.index).ffill().bfill()

    missing = [p for p in pairs.values() if p not in fx.columns or fx[p].isna().all()]
    if missing:
        raise ValueError(f"Taux de change introuvables : {', '.join(missing)}")

    for ticker, cur in currencies.items():
        if cur != base_currency:
            converted[ticker] = converted[ticker] * fx[pairs[cur]]

    return converted
