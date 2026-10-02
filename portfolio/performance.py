"""Rendements et mesures de performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_risk_free_rate_from_ticker


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Rendements simples ou logarithmiques.

    Les prix manquants (jour férié sur une seule des places de cotation) sont
    reportés depuis la veille : on garde la journée, avec un rendement nul pour
    l'actif dont la bourse était fermée, au lieu de supprimer la ligne entière.
    """
    prices = prices.ffill()
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.iloc[1:].dropna(how="any")


def annualized_return(returns, periods_per_year: int = 252):
    """Rendement moyen arithmétique annualisé."""
    return returns.mean() * periods_per_year


def cagr(returns, periods_per_year: int = 252):
    """Rendement annuel composé (géométrique)."""
    n = len(returns)
    if n == 0:
        return np.nan
    growth = (1 + returns).prod()
    return growth ** (periods_per_year / n) - 1


def annualized_volatility(returns, periods_per_year: int = 252):
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio_per_asset(
    returns: pd.DataFrame,
    rf=None,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Ratio de Sharpe annualisé par actif.

    `rf` peut être un taux unique, une Series indexée par ticker, ou None
    (taux de la devise de chaque actif, moyenné sur la période des rendements).
    """
    if rf is None:
        start = str(returns.index[0].date())
        end = str(returns.index[-1].date())
        rf = pd.Series({t: get_risk_free_rate_from_ticker(t, start, end) for t in returns.columns})
    elif np.isscalar(rf):
        rf = pd.Series(float(rf), index=returns.columns)

    excess = returns - rf / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)
