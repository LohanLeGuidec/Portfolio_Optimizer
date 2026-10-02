"""Outils d'analyse et d'optimisation de portefeuille."""

from .backtest import walk_forward_backtest
from .data_loader import convert_prices_to_currency, load_prices
from .optimization import (
    black_litterman_posterior,
    black_litterman_weights,
    efficient_frontier_random,
    implied_equilibrium_returns,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_performance,
    portfolio_volatility,
    risk_contributions,
    risk_parity_objective,
    risk_parity_weights,
)
from .performance import (
    annualized_return,
    annualized_volatility,
    cagr,
    compute_returns,
    sharpe_ratio_per_asset,
)
from .risk import (
    beta,
    beta_auto_for_all,
    calmar_ratio,
    drawdown_series,
    historical_cvar,
    historical_var,
    max_drawdown,
    portfolio_risks,
)
from .simulation import (
    monte_carlo_portfolio_paths,
    monte_carlo_simulation,
    simulate_portfolio_values,
)
from .utils import (
    DEFAULT_RISK_FREE_RATE,
    get_benchmark_ticker_from_currency,
    get_currency_from_ticker,
    get_risk_free_rate,
    get_risk_free_rate_from_ticker,
    map_tickers_to_benchmarks,
    normalize_weights,
    seed_everything,
)
from .analysis import LOCAL_CCY, STRATEGIES, run_analysis
from .statistics import (
    christoffersen_independence_test,
    kupiec_test,
    relative_metrics,
    rolling_historical_var,
    sharpe_statistics,
    var_backtest,
)
