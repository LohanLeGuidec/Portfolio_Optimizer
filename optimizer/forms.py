import datetime as dt
import re

from django import forms

from portfolio import LOCAL_CCY
from portfolio.analysis import DEFAULT_BENCHMARK

DEFAULT_TICKERS = "AAPL\nEL.PA\nMC.PA\nAI.PA\nOR.PA\nMSFT\nNVDA"
TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,20}$")
MAX_TICKERS = 25


def _choices(values, fmt="{}"):
    return [(v, fmt.format(v)) for v in values]


class AnalysisForm(forms.Form):
    tickers = forms.CharField(
        label="Tickers Yahoo Finance",
        help_text="Un par ligne (ex. AAPL, MC.PA, SAP.DE)",
        widget=forms.Textarea(attrs={"rows": 7, "spellcheck": "false"}),
        initial=DEFAULT_TICKERS,
    )
    start = forms.DateField(
        label="Date de début",
        initial=dt.date(2018, 1, 1),
        widget=forms.DateInput(attrs={"type": "date"}, format="%Y-%m-%d"),
    )
    base_ccy = forms.ChoiceField(
        label="Devise de référence",
        choices=[("EUR", "EUR"), ("USD", "USD"), ("GBP", "GBP"), ("CHF", "CHF"),
                 (LOCAL_CCY, "Locale (sans conversion)")],
        initial="EUR",
        help_text="Les prix sont convertis dans cette devise (risque de change inclus).",
    )
    benchmark = forms.CharField(
        label="Indice de référence",
        initial=DEFAULT_BENCHMARK,
        required=False,
        max_length=20,
        help_text="ETF Yahoo (dividendes inclus) : URTH = MSCI World, EXSA.DE = Stoxx 600, SPY = S&P 500. "
                  "Vide = pas d'indice.",
    )
    seed = forms.IntegerField(label="Seed", initial=42, min_value=0, max_value=2**31 - 1)

    n_sim = forms.TypedChoiceField(label="Simulations", coerce=int, initial=500,
                                   choices=_choices([100, 500, 1000, 2000, 5000]))
    horizon = forms.IntegerField(label="Horizon (jours)", initial=252, min_value=21, max_value=756)

    n_portfolios = forms.TypedChoiceField(label="Portefeuilles aléatoires", coerce=int, initial=2000,
                                          choices=_choices([500, 1000, 2000, 5000]))

    wf_window = forms.TypedChoiceField(label="Fenêtre d'estimation", coerce=int, initial=252,
                                       choices=_choices([126, 189, 252, 378, 504], "{} jours"))
    wf_rebal = forms.TypedChoiceField(label="Rebalancement", coerce=int, initial=21,
                                      choices=[(5, "Hebdomadaire (5 j)"), (21, "Mensuel (21 j)"),
                                               (63, "Trimestriel (63 j)")])
    wf_cost = forms.FloatField(label="Coûts de transaction (pb)", initial=10.0, min_value=0, max_value=200,
                               help_text="Points de base par unité de turnover")

    def clean_tickers(self):
        raw = self.cleaned_data["tickers"]
        tickers = list(dict.fromkeys(t.strip().upper() for t in re.split(r"[\s,;]+", raw) if t.strip()))
        invalid = [t for t in tickers if not TICKER_RE.match(t)]
        if invalid:
            raise forms.ValidationError(f"Tickers invalides : {', '.join(invalid)}")
        if len(tickers) < 2:
            raise forms.ValidationError("Indiquez au moins deux tickers.")
        if len(tickers) > MAX_TICKERS:
            raise forms.ValidationError(f"{MAX_TICKERS} tickers maximum.")
        return tickers

    def clean_benchmark(self):
        bench = self.cleaned_data.get("benchmark", "").strip().upper()
        if bench and not TICKER_RE.match(bench):
            raise forms.ValidationError("Ticker d'indice invalide.")
        return bench or None

    def clean_start(self):
        start = self.cleaned_data["start"]
        if start >= dt.date.today() - dt.timedelta(days=60):
            raise forms.ValidationError("Choisissez une date de début d'au moins deux mois dans le passé.")
        return start
