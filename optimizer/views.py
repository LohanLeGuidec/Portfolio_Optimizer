import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from django.core.cache import cache
from django.http import Http404, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from portfolio import (
    LOCAL_CCY,
    black_litterman_weights,
    historical_var,
    portfolio_performance,
    risk_contributions,
    run_analysis,
)

from . import charts
from .forms import AnalysisForm

CACHE_PREFIX = "analysis:"


# ── Formatage ────────────────────────────────────────────────────────────────

def pct(x, digits=2):
    # « − » typographique plutôt que le tiret « - »
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}%}".replace("-", "−")


def num(x, digits=2):
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}".replace("-", "−")


def ci(low, high, digits=2):
    if pd.isna(low) or pd.isna(high):
        return "—"
    return f"[{num(low, digits)} ; {num(high, digits)}]"


def table(columns, rows, first_col_label=True):
    return {"columns": columns, "rows": rows, "first_col_label": first_col_label}


def frame_table(df: pd.DataFrame, fmt, index_label="", index_fmt=str):
    return table(
        [index_label, *df.columns],
        [[index_fmt(idx), *(fmt(v) for v in row)] for idx, row in zip(df.index, df.values)],
    )


# ── Cache des analyses ───────────────────────────────────────────────────────

def _run_key(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _get_results(run_key: str):
    if not run_key or not run_key.isalnum():
        return None
    return cache.get(CACHE_PREFIX + run_key)


# ── Contexte de la page de résultats ─────────────────────────────────────────

def build_results_context(R: dict, run_key: str) -> dict:
    tickers = R["tickers"]
    names = list(R["portfolios"])
    summary = R["summary"]
    params = R["params"]

    kpis = [{
        "name": name,
        "color": charts.COLORS[name],
        "ret": pct(summary.loc[name, "return"]),
        "vol": pct(summary.loc[name, "volatility"]),
        "sharpe": num(summary.loc[name, "sharpe"]),
        "mdd": pct(summary.loc[name, "max_drawdown"]),
    } for name in names]

    am = R["asset_metrics"]
    assets_table = table(
        ["Actif", "Rendement ann.", "Volatilité ann.", "Sharpe", "Bêta"],
        [[t, pct(r["return"]), pct(r["volatility"]), num(r["sharpe"]), num(r["beta"])] for t, r in am.iterrows()],
    )

    bench = R.get("benchmark")
    bench_label = f"Indice ({bench})" if bench else None
    bs = R.get("benchmark_stats")

    perf_rows = [[n, pct(summary.loc[n, "return"]), pct(summary.loc[n, "volatility"]),
                  num(summary.loc[n, "sharpe"]), ci(summary.loc[n, "sharpe_low"], summary.loc[n, "sharpe_high"]),
                  pct(summary.loc[n, "psr"], 1), pct(summary.loc[n, "max_drawdown"])] for n in names]
    if bs:
        perf_rows.append([bench_label, pct(bs["return"]), pct(bs["volatility"]), num(bs["sharpe"]),
                          ci(bs["sharpe_low"], bs["sharpe_high"]), pct(bs["psr"], 1), pct(bs["max_drawdown"])])
    perf_table = table(
        ["Portefeuille", "Rendement ann.", "Volatilité ann.", "Sharpe", "IC 95 % du Sharpe", "PSR",
         "Max drawdown"],
        perf_rows,
    )

    def relative_table(df):
        return table(
            ["Portefeuille", "Alpha de Jensen", "Bêta", "Tracking error", "Information ratio", "Treynor",
             "Sortino", "Corrélation"],
            [[n, pct(r["alpha"]), num(r["beta"]), pct(r["tracking_error"]), num(r["information_ratio"]),
              # Treynor n'a pas de sens si le bêta est quasi nul (division par ~0)
              pct(r["treynor"]) if abs(r["beta"]) >= 0.1 else "—",
              num(r["sortino"]), num(r["correlation"])] for n, r in df.iterrows()],
        )

    var_rows = []
    for n, tests in R.get("var_tests", {}).items():
        for level, t in sorted(tests.items()):
            var_rows.append([
                n, f"{level:.0%}", f"{t['exceptions']} / {t['expected']:.0f}", pct(t["rate"]),
                num(t["kupiec_p"], 3), num(t["christoffersen_p"], 3),
                "Rejetée" if t["rejected"] else "Validée",
            ])
    var_table = table(
        ["Portefeuille", "Niveau", "Exceptions (obs. / att.)", "Taux", "p-value Kupiec",
         "p-value Christoffersen", "Modèle"],
        var_rows,
    )
    risk_table = table(
        ["Portefeuille", "Max drawdown", "VaR 95 %", "CVaR 95 %", "Volatilité ann.", "Calmar"],
        [[n, pct(summary.loc[n, "max_drawdown"]), pct(summary.loc[n, "var95"]), pct(summary.loc[n, "cvar95"]),
          pct(summary.loc[n, "volatility"]), num(summary.loc[n, "calmar"])] for n in names],
    )
    weights_df = pd.DataFrame(R["portfolios"], index=tickers)
    weights_table = frame_table(weights_df, pct, "Actif")

    mc = R["monte_carlo"]
    mc_table = table(
        ["Portefeuille", "Médiane", "P10", "P90", "P(perte)", "P(+50 %)"],
        [[n, num(m["median_final"], 3), num(m["p10_final"], 3), num(m["p90_final"], 3),
          pct(m["prob_loss"], 1), pct(m["prob_gain50"], 1)] for n, m in mc.items()],
    )

    def date_fmt(d):
        return d.strftime("%d/%m/%Y")

    prices_tail = R["prices"].tail(30).iloc[::-1]
    returns_tail = R["returns"].tail(30).iloc[::-1]
    cov_df = pd.DataFrame(R["cov"] * 252, index=tickers, columns=tickers)

    wf = R["wf_summary"]
    wb = R.get("wf_benchmark")
    wf_comp_rows = [[n, pct(r["total_is"]), pct(r["total_oos"]), pct(r["total_oos"] - r["total_is"])]
                    for n, r in wf.iterrows()]
    if wb:
        wf_comp_rows.append([bench_label, "—", pct(wb["total"]), "—"])
    wf_comp_table = table(["Portefeuille", "In-sample", "Out-of-sample", "Écart"], wf_comp_rows)

    wf_risk_rows = [[n, pct(r["return"]), pct(r["volatility"]), num(r["sharpe"]),
                     ci(r["sharpe_low"], r["sharpe_high"]), pct(r["psr"], 1), pct(r["max_drawdown"]),
                     pct(r["var95"]), pct(r["cvar95"])] for n, r in wf.iterrows()]
    if wb:
        wf_risk_rows.append([bench_label, pct(wb["return"]), pct(wb["volatility"]), num(wb["sharpe"]),
                             ci(wb["sharpe_low"], wb["sharpe_high"]), pct(wb["psr"], 1), pct(wb["max_drawdown"]),
                             "—", "—"])
    wf_risk_table = table(
        ["Portefeuille", "Rendement ann.", "Volatilité ann.", "Sharpe", "IC 95 % du Sharpe", "PSR",
         "Max drawdown", "VaR 95 %", "CVaR 95 %"],
        wf_risk_rows,
    )

    points = {n: (summary.loc[n, "return"], summary.loc[n, "volatility"]) for n in names}

    def rc_chart(w):
        rc = risk_contributions(w, R["cov"])
        return charts.bars(tickers, rc / rc.sum(), y_title="Part du risque total")

    equity_series = dict(R["equity_curves"])
    drawdown_series_ = dict(R["drawdowns"])
    if bs:
        equity_series[bench_label] = bs["equity"]
        drawdown_series_[bench_label] = bs["drawdown"]

    chart_data = {
        "ann_returns": charts.annual_returns_bar(am),
        "risk_return": charts.risk_return_scatter(am),
        "corr": charts.correlation_heatmap(R["returns"]),
        "equity": charts.lines(equity_series),
        "drawdowns": charts.lines(drawdown_series_, y_title="Drawdown", fmt=".2%"),
        "mdd_bar": charts.bars(names, [summary.loc[n, "max_drawdown"] for n in names], y_title="Max drawdown",
                               fmt=".2%"),
        "frontier": charts.efficient_frontier(R["frontier"], points),
        "mc_violin": charts.monte_carlo_violin(mc, params["horizon"]),
    }
    if R["wf_equity_curves"]:
        wf_curves = dict(R["wf_equity_curves"])
        if wb:
            wf_curves[bench_label] = wb["equity"]
        chart_data["wf_equity"] = charts.lines(wf_curves)
        chart_data["wf_drawdowns"] = charts.lines(
            {n: eq / eq.cummax() - 1 for n, eq in wf_curves.items()},
            y_title="Drawdown", fmt=".2%", height=320,
        )

    groups = {
        "mc_fan": [(n, charts.monte_carlo_fan(mc, n)) for n in names],
        "var_backtest": [
            (n, charts.var_backtest_chart(R["port_returns"][n], tests, charts.COLORS[n]))
            for n, tests in R.get("var_tests", {}).items()
        ],
        "pies": [(n, charts.allocation_pie(tickers, w)) for n, w in R["portfolios"].items()],
        "risk_contrib": [(n, rc_chart(w)) for n, w in R["portfolios"].items()],
        "histograms": [
            (n, charts.returns_histogram(R["port_returns"][n], historical_var(R["port_returns"][n]),
                                         charts.COLORS[n]))
            for n in names
        ],
    }

    start, end = R["period"]
    return {
        "run_key": run_key,
        "tickers": tickers,
        "portfolio_names": names,
        "params": params,
        "currency_label": "devises locales" if params["base_ccy"] == LOCAL_CCY else params["base_ccy"],
        "period_start": date_fmt(start),
        "period_end": date_fmt(end),
        "n_days": len(R["returns"]),
        "rf": pct(R["rf"]),
        "kpis": kpis,
        "charts": chart_data,
        "groups": groups,
        "tables": {
            "assets": assets_table,
            "perf": perf_table,
            "risk": risk_table,
            "weights": weights_table,
            "mc": mc_table,
            "prices": frame_table(prices_tail, lambda v: num(v), "Date", date_fmt),
            "returns": frame_table(returns_tail, lambda v: pct(v, 3), "Date", date_fmt),
            "cov": frame_table(cov_df, lambda v: f"{v:.4f}".replace("-", "−"), ""),
            "wf_comp": wf_comp_table,
            "relative": relative_table(R["relative"]) if bs else None,
            "wf_relative": relative_table(R["wf_relative"]) if wb and not R["wf_relative"].empty else None,
            "var": var_table,
            "wf_risk": wf_risk_table,
        },
        "wf_errors": R["wf_errors"],
        "benchmark": bench,
        "benchmark_label": bench_label,
        "benchmark_error": R.get("benchmark_error"),
        "has_var_tests": bool(R.get("var_tests")),
        "equal_weight": round(100 / len(tickers), 2),
    }


# ── Vues ─────────────────────────────────────────────────────────────────────

@require_GET
def index(request):
    form = AnalysisForm(request.GET) if request.GET else AnalysisForm()
    context = {"form": form}

    if form.is_bound and form.is_valid():
        params = dict(form.cleaned_data)
        params["start"] = params["start"].isoformat()
        run_key = _run_key(params)
        R = _get_results(run_key)
        if R is None:
            try:
                R = run_analysis(**params)
            except Exception as exc:  # noqa: BLE001 — erreur affichée à l'utilisateur
                context["error"] = str(exc)
            else:
                cache.set(CACHE_PREFIX + run_key, R)
        if R is not None:
            context.update(build_results_context(R, run_key))

    return render(request, "optimizer/index.html", context)


@require_POST
def black_litterman(request):
    R = _get_results(request.POST.get("run", ""))
    if R is None:
        return render(request, "optimizer/partials/bl_result.html",
                      {"error": "Les résultats ont expiré : relancez l'analyse."})

    tickers = R["tickers"]
    n = len(tickers)
    errors = []

    ref = request.POST.get("ref", "Equal Weight")
    if ref in R["portfolios"]:
        w_ref = np.asarray(R["portfolios"][ref], dtype=float)
    else:
        try:
            raw = np.array([float(request.POST.get(f"w_{i}", "0").replace(",", ".")) for i in range(n)])
        except ValueError:
            raw = np.zeros(n)
        if (raw < 0).any() or raw.sum() <= 0:
            errors.append("Poids personnalisés invalides : ils doivent être positifs et de somme non nulle.")
            raw = np.ones(n)
        w_ref = raw / raw.sum()

    try:
        tau = float(request.POST.get("tau", "0.05"))
        confidence = float(request.POST.get("confidence", "0.5"))
    except ValueError:
        tau, confidence = 0.05, 0.5
    tau = min(max(tau, 0.001), 1.0)
    confidence = min(max(confidence, 0.05), 0.95)

    P_rows, Q_vals = [], []
    views = zip(request.POST.getlist("view_a"), request.POST.getlist("view_b"), request.POST.getlist("view_q"))
    for i, (a, b, q) in enumerate(views, start=1):
        if a not in tickers or b not in tickers:
            continue
        if a == b:
            errors.append(f"Vue {i} ignorée : les deux actifs sont identiques ({a}).")
            continue
        try:
            q_val = float(q.replace(",", ".")) / 100
        except ValueError:
            errors.append(f"Vue {i} ignorée : écart attendu invalide.")
            continue
        row = np.zeros(n)
        row[tickers.index(a)] = 1.0
        row[tickers.index(b)] = -1.0
        P_rows.append(row)
        Q_vals.append(q_val)

    if not P_rows:
        return render(request, "optimizer/partials/bl_result.html",
                      {"error": "Aucune vue valide.", "warnings": errors})

    try:
        w_bl, mu_bl, _ = black_litterman_weights(
            R["cov"], w_ref, np.array(P_rows), np.array(Q_vals), tau=tau, view_confidence=confidence,
        )
    except Exception as exc:  # noqa: BLE001
        return render(request, "optimizer/partials/bl_result.html", {"error": f"Calcul impossible : {exc}"})

    ret_bl, vol_bl = portfolio_performance(w_bl, R["mean_ret"], R["cov"])
    ret_ref, vol_ref = portfolio_performance(w_ref, R["mean_ret"], R["cov"])

    context = {
        "warnings": errors,
        "chart_weights": charts.grouped_weights(tickers, w_ref, w_bl),
        "chart_mu": charts.bars(tickers, mu_bl, y_title="Rendement a posteriori (annuel)", color="#7B4FA8"),
        "table": table(
            ["Actif", "Poids référence", "Poids Black-Litterman", "Variation", "Rendement a posteriori"],
            [[t, pct(w_ref[i]), pct(w_bl[i]), pct(w_bl[i] - w_ref[i]), pct(mu_bl[i])] for i, t in enumerate(tickers)],
        ),
        "stats": [
            ("Référence — rendement historique", pct(ret_ref)), ("Référence — volatilité", pct(vol_ref)),
            ("BL — rendement historique", pct(ret_bl)), ("BL — volatilité", pct(vol_bl)),
        ],
    }
    return render(request, "optimizer/partials/bl_result.html", context)


@require_GET
def export_csv(request, run, kind):
    R = _get_results(run)
    if R is None:
        raise Http404("Résultats expirés : relancez l'analyse.")
    if kind == "prices":
        df = R["prices"]
    elif kind == "returns":
        df = R["returns"]
    elif kind == "weights":
        df = pd.DataFrame(R["portfolios"], index=R["tickers"])
    else:
        raise Http404()
    response = HttpResponse(df.to_csv(), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{kind}.csv"'
    return response


@lru_cache(maxsize=1)
def _plotly_bundle() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


@require_GET
def plotly_js(request):
    """Sert Plotly.js depuis le package Python : version toujours identique, fonctionne hors ligne."""
    response = HttpResponse(_plotly_bundle(), content_type="application/javascript")
    response["Cache-Control"] = "public, max-age=86400"
    return response
