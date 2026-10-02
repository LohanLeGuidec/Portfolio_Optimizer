import numpy as np
import pytest

from portfolio.utils import (
    get_benchmark_ticker_from_currency,
    get_currency_from_ticker,
    map_tickers_to_benchmarks,
    normalize_weights,
)


@pytest.mark.parametrize(
    "ticker, currency",
    [
        ("AAPL", "USD"),
        ("MC.PA", "EUR"),
        ("SAP.DE", "EUR"),
        ("ASML.AS", "EUR"),
        ("HSBA.L", "GBP"),
        ("NESN.SW", "CHF"),
        ("7203.T", "JPY"),
        ("RY.TO", "CAD"),
        ("BHP.AX", "AUD"),
        ("RELIANCE.NS", "INR"),
        ("PETR4.SA", "BRL"),
    ],
)
def test_currency_from_yahoo_suffix(ticker, currency):
    assert get_currency_from_ticker(ticker) == currency


def test_french_stocks_use_euro_stoxx_benchmark():
    mapping = map_tickers_to_benchmarks(["MC.PA", "AAPL"])
    assert mapping == {"MC.PA": "^STOXX50E", "AAPL": "^GSPC"}


def test_unknown_currency_raises():
    with pytest.raises(ValueError):
        get_benchmark_ticker_from_currency("XYZ")


def test_normalize_weights():
    np.testing.assert_allclose(normalize_weights([1, 1, 2]), [0.25, 0.25, 0.5])
    with pytest.raises(ValueError):
        normalize_weights([0, 0])
