# Portfolio Optimizer

Application web (Django) d'allocation de portefeuille multi-stratégies : Markowitz, risk parity, max Sharpe, Black-Litterman, simulation Monte Carlo et backtest walk-forward hors échantillon.

*Lohan Le Guidec — Master 1 MBFA, Ingénierie Économique et Financière, Université de Rennes*

## Démonstration

https://github.com/user-attachments/assets/19f77039-a2ae-4b77-a3b6-1e993cb51713

## Fonctionnalités

| Onglet | Contenu |
|---|---|
| Vue d'ensemble | Rendement, volatilité, Sharpe et bêta par actif, matrice de corrélation |
| Portefeuilles | Equal Weight, Min Variance, Risk Parity, Max Sharpe : performances avec IC du Sharpe et PSR, performance relative à l'indice (alpha, bêta, tracking error, information ratio, Treynor, Sortino), allocations, contributions au risque |
| Frontière efficiente | Portefeuilles aléatoires (tirage uniforme sur le simplexe) et positionnement des stratégies |
| Risque & Drawdown | Drawdowns, VaR / CVaR historiques, ratio de Calmar, backtest de la VaR (Kupiec, Christoffersen) |
| Monte Carlo | Trajectoires simulées (loi normale multivariée, covariance Ledoit-Wolf), bandes P10–P90 |
| Black-Litterman | Vues relatives entre actifs, confiance réglable, poids optimaux long-only |
| Backtest OOS | Walk-forward : réestimation périodique sur fenêtre glissante, coûts de transaction, comparaison in-sample / out-of-sample sur les mêmes dates, alpha et information ratio hors échantillon |

## Méthodologie

- **Données** : prix ajustés Yahoo Finance. Les jours fériés propres à une place sont reportés (rendement nul) plutôt que supprimés.
- **Devise** : les prix peuvent être convertis dans une devise de référence (EUR par défaut) pour intégrer le risque de change. Les actions de Londres sont converties de pence en livres.
- **Covariance** : estimateur de Ledoit-Wolf (shrinkage), plus stable que la covariance empirique.
- **Taux sans risque** : taux 3 mois de la devise (FRED), **moyenné sur la période analysée**.
- **Contraintes** : toutes les stratégies sont long-only (0 ≤ w ≤ 1, Σw = 1), pour une comparaison homogène.
- **Risk parity** : formulation convexe de Spinu (2013), solution unique.
- **Black-Litterman** : modèle en annuel ; Ω = diag(P τΣ P') · (1 − c) / c, où c est la confiance dans les vues (c = 0,5 correspond à He & Litterman).
- **Bêta** : calculé en devise locale contre l'indice du marché de cotation (Euro Stoxx 50 pour Paris, S&P 500 pour les États-Unis, etc.).
- **Indice de référence** : ETF (URTH = MSCI World par défaut) plutôt qu'un indice de prix, pour comparer à dividendes réinvestis ; converti dans la devise de référence.
- **Significativité du Sharpe** : erreur standard de Mertens (2002), qui corrige la formule de Lo (2002) pour l'asymétrie et le kurtosis ; Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).
- **Backtest de la VaR** : VaR historique glissante sur 250 séances, prévue sans information du jour testé ; test de couverture de Kupiec (1995) et test d'indépendance de Christoffersen (1998), à 95 % et 99 %.
- **Backtest** : aucun look-ahead. Les poids dérivent avec les prix entre deux rebalancements, et un coût proportionnel au turnover est appliqué.

## Installation

```powershell
python -m venv .venv            # ou : uv venv
.venv\Scripts\activate
pip install -r requirements\dev.txt   # ou : uv pip install -r requirements\dev.txt
```

## Lancement

```powershell
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000. Aucune base de données n'est nécessaire (pas de `migrate`) : les résultats d'analyse sont mis en cache sur disque pendant 1 h dans `.cache/`.

### Production

Variables d'environnement (ou fichier `.env` à la racine) :

```
DJANGO_DEBUG=0
DJANGO_SECRET_KEY=<une longue chaîne aléatoire>
DJANGO_ALLOWED_HOSTS=mon-domaine.fr
```

puis `python manage.py collectstatic` et un serveur WSGI (gunicorn, waitress…) sur `config.wsgi`.

## Tests

```powershell
pytest
```

Les tests (calcul et pages Django) utilisent des données simulées et ne nécessitent pas de connexion.

## Structure

```
manage.py               Point d'entrée Django
config/                 Réglages et URLs du projet Django
optimizer/              Application web : formulaire, vues, graphiques Plotly, templates, CSS/JS
portfolio/
  analysis.py           Analyse complète, indépendante de l'interface
  statistics.py         IC du Sharpe, PSR, backtest de la VaR (Kupiec, Christoffersen), alpha / TE / IR
  data_loader.py        Téléchargement des prix, conversion de devise
  performance.py        Rendements, CAGR, volatilité, Sharpe
  risk.py               Drawdown, VaR, CVaR, Calmar, bêta
  optimization.py       Min variance, risk parity, max Sharpe, Black-Litterman, frontière
  simulation.py         Monte Carlo
  backtest.py           Backtest walk-forward
  utils.py              Devises, benchmarks, taux sans risque
tests/                  Tests pytest
requirements/           base.txt (application) et dev.txt (+ tests)
```

## Limites

- Les rendements simulés suivent une loi normale : les queues épaisses sont sous-estimées.
- **Biais de sélection de l'univers** : choisir aujourd'hui des titres qui ont bien performé (NVDA, MSFT…) gonfle tous les résultats, y compris hors échantillon. Un univers fixé ex ante (composants d'un indice à la date de départ) éliminerait ce biais.
- Quatre stratégies sont comparées sans correction pour tests multiples (Deflated Sharpe Ratio).
- Les rendements espérés historiques sont des estimateurs bruités ; le backtest hors échantillon sert précisément à mesurer cet effet.
- Données Yahoo Finance non garanties (usage pédagogique).
