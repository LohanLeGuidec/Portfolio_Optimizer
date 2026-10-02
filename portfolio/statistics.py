"""
Validation statistique : significativité du ratio de Sharpe, backtest de la VaR
et performance relative à un indice de référence.

Références :
- Lo (2002), « The Statistics of Sharpe Ratios », Financial Analysts Journal.
- Mertens (2002) : correction de l'erreur standard pour l'asymétrie et le kurtosis.
- Bailey & López de Prado (2012), « The Sharpe Ratio Efficient Frontier » (PSR).
- Kupiec (1995) : test de couverture inconditionnelle (proportion of failures).
- Christoffersen (1998) : test d'indépendance des exceptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ── Ratio de Sharpe ──────────────────────────────────────────────────────────


def sharpe_statistics(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Ratio de Sharpe annualisé avec son intervalle de confiance.

    L'erreur standard suit Mertens (2002), qui tient compte de l'asymétrie (γ3)
    et du kurtosis (γ4) des rendements : des queues épaisses élargissent
    l'intervalle par rapport à l'hypothèse gaussienne.

    `psr` (Probabilistic Sharpe Ratio) est la probabilité que le vrai Sharpe
    soit positif compte tenu de la longueur de l'historique.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 30:
        return {"sharpe": np.nan, "low": np.nan, "high": np.nan, "se": np.nan, "psr": np.nan, "n": n}

    excess = r - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return {"sharpe": np.nan, "low": np.nan, "high": np.nan, "se": np.nan, "psr": np.nan, "n": n}

    sr = excess.mean() / sd                       # Sharpe par période
    skew = stats.skew(excess, bias=False)
    kurt = stats.kurtosis(excess, fisher=False, bias=False)  # kurtosis « brut » (3 si gaussien)
    var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / (n - 1)
    se = np.sqrt(max(var_sr, 0.0))

    z = stats.norm.ppf(0.5 + confidence / 2)
    scale = np.sqrt(periods_per_year)
    return {
        "sharpe": float(sr * scale),
        "low": float((sr - z * se) * scale),
        "high": float((sr + z * se) * scale),
        "se": float(se * scale),
        "psr": float(stats.norm.cdf(sr / se)) if se > 0 else np.nan,
        "n": n,
    }


# ── Backtest de la VaR ───────────────────────────────────────────────────────


def rolling_historical_var(returns: pd.Series, level: float = 0.95, window: int = 250) -> pd.Series:
    """
    VaR historique prévue pour chaque jour t à partir des `window` jours précédents
    uniquement (t-window … t-1) : aucune information du jour testé n'est utilisée.
    """
    alpha = 1 - level
    return returns.rolling(window).quantile(alpha).shift(1).dropna()


def kupiec_test(exceptions: int, n_obs: int, level: float) -> dict:
    """
    Test de Kupiec (POF) : la fréquence observée des exceptions est-elle compatible
    avec la fréquence théorique 1 - level ?  LR ~ χ²(1) sous H0.
    """
    p = 1 - level
    x, n = int(exceptions), int(n_obs)
    if n == 0:
        return {"lr": np.nan, "p_value": np.nan}
    phat = x / n

    def loglik(prob):
        # Convention 0·log(0) = 0
        out = 0.0
        if n - x > 0:
            out += (n - x) * np.log(1 - prob)
        if x > 0:
            out += x * np.log(prob)
        return out

    lr = -2 * (loglik(p) - (loglik(phat) if 0 < phat < 1 else 0.0))
    lr = max(lr, 0.0)
    return {"lr": float(lr), "p_value": float(stats.chi2.sf(lr, df=1))}


def christoffersen_independence_test(hits: np.ndarray) -> dict:
    """
    Test d'indépendance de Christoffersen : une exception aujourd'hui rend-elle
    une exception demain plus probable (clustering) ?  LR ~ χ²(1) sous H0.
    """
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        return {"lr": np.nan, "p_value": np.nan}
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(count_stay, count_move, prob):
        out = 0.0
        if count_stay:
            out += count_stay * np.log(1 - prob) if prob < 1 else -np.inf
        if count_move:
            out += count_move * np.log(prob) if prob > 0 else -np.inf
        return out

    if pi in (0.0, 1.0):
        return {"lr": 0.0, "p_value": 1.0}
    lr = -2 * (ll(n00 + n10, n01 + n11, pi) - (ll(n00, n01, pi0) + ll(n10, n11, pi1)))
    lr = max(float(lr), 0.0)
    return {"lr": lr, "p_value": float(stats.chi2.sf(lr, df=1))}


def var_backtest(returns: pd.Series, level: float = 0.95, window: int = 250, significance: float = 0.05) -> dict:
    """
    Backtest d'une VaR historique glissante : nombre d'exceptions, tests de
    Kupiec et de Christoffersen, verdict au seuil `significance`.
    """
    returns = pd.Series(returns).dropna()
    var = rolling_historical_var(returns, level, window)
    realized = returns.loc[var.index]
    hits = (realized < var).to_numpy()
    n, x = len(hits), int(hits.sum())

    kupiec = kupiec_test(x, n, level)
    indep = christoffersen_independence_test(hits)
    rejected = (kupiec["p_value"] < significance) or (indep["p_value"] < significance)
    return {
        "level": level,
        "n_obs": n,
        "exceptions": x,
        "expected": n * (1 - level),
        "rate": x / n if n else np.nan,
        "kupiec_lr": kupiec["lr"],
        "kupiec_p": kupiec["p_value"],
        "christoffersen_p": indep["p_value"],
        "rejected": bool(rejected),
        "var_series": var,
        "hits": pd.Series(hits, index=var.index),
    }


# ── Performance relative à un indice ─────────────────────────────────────────


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Écart-type des rendements sous le seuil `mar` (annuel), annualisé."""
    shortfall = np.minimum(returns - mar / periods_per_year, 0.0)
    return float(np.sqrt((shortfall**2).mean()) * np.sqrt(periods_per_year))


def relative_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Métriques de gestion active par rapport à un indice :
    bêta, alpha de Jensen, tracking error, information ratio, ratio de Treynor,
    ratio de Sortino, corrélation et excès de rendement annualisé.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {k: np.nan for k in ("beta", "alpha", "tracking_error", "information_ratio",
                                    "treynor", "sortino", "correlation", "excess_return")}
    rp, rb = aligned.iloc[:, 0], aligned.iloc[:, 1]
    rf_d = rf / periods_per_year

    cov = np.cov(rp, rb, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha = ((rp - rf_d) - beta * (rb - rf_d)).mean() * periods_per_year

    active = rp - rb
    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    excess = active.mean() * periods_per_year
    ann_ret = rp.mean() * periods_per_year
    dd = downside_deviation(rp, rf, periods_per_year)

    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "tracking_error": float(te),
        "information_ratio": float(excess / te) if te > 0 else np.nan,
        "treynor": float((ann_ret - rf) / beta) if beta != 0 else np.nan,
        "sortino": float((ann_ret - rf) / dd) if dd > 0 else np.nan,
        "correlation": float(np.corrcoef(rp, rb)[0, 1]),
        "excess_return": float(excess),
    }
