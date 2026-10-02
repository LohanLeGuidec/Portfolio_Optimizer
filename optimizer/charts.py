"""Construction des graphiques Plotly (sérialisés en JSON pour Plotly.js)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "Equal Weight": "#2F6FB3",
    "Min Variance": "#1D8A66",
    "Risk Parity": "#D9622B",
    "Max Sharpe": "#7B4FA8",
    "Black-Litterman": "#2F6FB3",
    "Référence": "#9A968E",
}
BENCHMARK_COLOR = "#55524C"
COLORWAY = ["#2F6FB3", "#1D8A66", "#D9622B", "#C4413D", "#7B4FA8", "#C9901A", "#178F86", "#8A5A3C",
            "#5C7CA8", "#6B8E3A"]
SYMBOLS = {"Equal Weight": "circle", "Min Variance": "diamond", "Risk Parity": "square", "Max Sharpe": "star"}
DIVERGING = [[0, "#C4413D"], [0.5, "#F2F0EB"], [1, "#2F6FB3"]]
RED_GREEN = [[0, "#C4413D"], [0.5, "#C9901A"], [1, "#1D8A66"]]


def _layout(height=360, **kwargs):
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        colorway=COLORWAY,
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color="#2B2A27"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_size=12),
        xaxis=dict(gridcolor="#E7E4DD", zerolinecolor="#CFCBC2"),
        yaxis=dict(gridcolor="#E7E4DD", zerolinecolor="#CFCBC2"),
    )
    for key in ("xaxis", "yaxis"):
        if key in kwargs:
            layout[key] = {**layout[key], **kwargs.pop(key)}
    layout.update(kwargs)
    return layout


def to_json(fig: go.Figure) -> str:
    # « </ » échappé pour pouvoir insérer le JSON dans une balise <script>
    return fig.to_json().replace("</", "<\\/")


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# ── Vue d'ensemble ───────────────────────────────────────────────────────────

def annual_returns_bar(asset_metrics: pd.DataFrame) -> str:
    vals = asset_metrics["return"]
    fig = go.Figure(go.Bar(
        x=list(vals.index), y=vals.values,
        marker=dict(color=vals.values, colorscale=RED_GREEN, line=dict(width=0)),
        text=[f"{v:.1%}" for v in vals.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**_layout(320, yaxis=dict(tickformat=".0%", title="Rendement annualisé")))
    return to_json(fig)


def risk_return_scatter(asset_metrics: pd.DataFrame) -> str:
    fig = go.Figure()
    for i, (ticker, row) in enumerate(asset_metrics.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row["volatility"]], y=[row["return"]], mode="markers+text", text=[ticker],
            textposition="top center", name=ticker, showlegend=False,
            marker=dict(size=11, color=COLORWAY[i % len(COLORWAY)], line=dict(width=1, color="white")),
            hovertemplate=f"{ticker}<br>Vol : %{{x:.2%}}<br>Rdt : %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(320, xaxis=dict(tickformat=".0%", title="Volatilité annualisée"),
                                yaxis=dict(tickformat=".0%", title="Rendement annualisé")))
    return to_json(fig)


def correlation_heatmap(returns: pd.DataFrame) -> str:
    corr = returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}<br>%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(420, yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                                xaxis=dict(gridcolor="rgba(0,0,0,0)")))
    return to_json(fig)


# ── Portefeuilles ────────────────────────────────────────────────────────────

def lines(series: dict, height=380, y_title="Valeur (base 1)", fmt=".3f", fill=False) -> str:
    fig = go.Figure()
    for name, s in series.items():
        is_bench = name.startswith("Indice")
        color = BENCHMARK_COLOR if is_bench else COLORS.get(name, COLORWAY[0])
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=color, width=1.6 if fill else 2, dash="dash" if is_bench else None),
            fill="tozeroy" if fill else None,
            fillcolor=_rgba(color, 0.12) if fill else None,
            hovertemplate=f"%{{x|%d %b %Y}}  %{{y:{fmt}}}<extra>{name}</extra>",
        ))
    yaxis = dict(title=y_title)
    if fmt.endswith("%"):
        yaxis["tickformat"] = ".0%"
    fig.update_layout(**_layout(height, yaxis=yaxis, hovermode="x unified"))
    return to_json(fig)


def allocation_pie(tickers: list[str], weights) -> str:
    fig = go.Figure(go.Pie(
        labels=tickers, values=weights, hole=0.5, sort=False,
        textinfo="label+percent", textposition="outside",
        marker=dict(colors=COLORWAY[:len(tickers)], line=dict(color="white", width=1.5)),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_layout(360), showlegend=False)
    return to_json(fig)


def bars(labels, values, height=300, y_title="", color=None, fmt=".1%") -> str:
    colors = color or [COLORS.get(lbl, COLORWAY[i % len(COLORWAY)]) for i, lbl in enumerate(labels)]
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values), marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:{fmt}}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:" + fmt + "}<extra></extra>",
    ))
    fig.update_layout(**_layout(height, yaxis=dict(tickformat=".0%", title=y_title)))
    return to_json(fig)


# ── Frontière efficiente ─────────────────────────────────────────────────────

def efficient_frontier(frontier: dict, points: dict) -> str:
    fig = go.Figure(go.Scattergl(
        x=frontier["vols"], y=frontier["returns"], mode="markers", name="Portefeuilles aléatoires",
        marker=dict(size=4, color=frontier["sharpes"], colorscale=RED_GREEN, opacity=0.55,
                    colorbar=dict(title="Sharpe", thickness=12)),
        hovertemplate="Vol : %{x:.2%}<br>Rdt : %{y:.2%}<extra></extra>",
    ))
    for name, (ret, vol) in points.items():
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", text=[name], textposition="top right", name=name,
            marker=dict(size=14, color=COLORS.get(name), symbol=SYMBOLS.get(name, "circle"),
                        line=dict(width=2, color="white")),
            hovertemplate=f"{name}<br>Vol : %{{x:.2%}}<br>Rdt : %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(520, xaxis=dict(tickformat=".0%", title="Volatilité annualisée"),
                                yaxis=dict(tickformat=".0%", title="Rendement annualisé")))
    return to_json(fig)


# ── Risque ───────────────────────────────────────────────────────────────────

def returns_histogram(port_returns: pd.Series, var95: float, color: str) -> str:
    fig = go.Figure(go.Histogram(x=port_returns.values, nbinsx=70, marker_color=color, opacity=0.8,
                                 hovertemplate="%{x:.2%}<br>%{y} jours<extra></extra>"))
    fig.add_vline(x=var95, line_dash="dash", line_color="#C4413D",
                  annotation_text=f"VaR 95 % : {var95:.2%}", annotation_position="top left")
    fig.update_layout(**_layout(320, xaxis=dict(tickformat=".1%", title="Rendement journalier"),
                                yaxis=dict(title="Nombre de jours")), showlegend=False)
    return to_json(fig)


def var_backtest_chart(returns: pd.Series, tests: dict, color: str) -> str:
    """Rendements journaliers, VaR prévues à 95 % et 99 %, et exceptions."""
    first = min(t["var_series"].index[0] for t in tests.values())
    r = returns.loc[first:]
    fig = go.Figure(go.Scatter(
        x=r.index, y=r.values, mode="lines", name="Rendement journalier",
        line=dict(color=_rgba(color, 0.45), width=1),
        hovertemplate="%{x|%d %b %Y}  %{y:.2%}<extra></extra>",
    ))
    styles = {0.95: ("#C9901A", "dot"), 0.99: ("#C4413D", "solid")}
    for level, t in sorted(tests.items()):
        lc, dash = styles.get(level, ("#C4413D", "solid"))
        label = f"VaR {level:.0%}"
        fig.add_trace(go.Scatter(
            x=t["var_series"].index, y=t["var_series"].values, mode="lines", name=label,
            line=dict(color=lc, width=1.6, dash=dash),
            hovertemplate="%{x|%d %b %Y}  %{y:.2%}<extra>" + label + "</extra>",
        ))
        hits = t["hits"][t["hits"]]
        fig.add_trace(go.Scatter(
            x=hits.index, y=returns.loc[hits.index].values, mode="markers",
            name=f"Exceptions {level:.0%} ({len(hits)})",
            marker=dict(color=lc, size=6 if level == 0.99 else 5, symbol="x" if level == 0.99 else "circle"),
            hovertemplate="%{x|%d %b %Y}  %{y:.2%}<extra>Exception " + label + "</extra>",
        ))
    fig.update_layout(**_layout(380, yaxis=dict(tickformat=".1%", title="Rendement journalier")))
    return to_json(fig)


# ── Monte Carlo ──────────────────────────────────────────────────────────────

def monte_carlo_fan(monte_carlo: dict, focus: str) -> str:
    """Bande P10–P90 du portefeuille `focus`, trajectoires moyennes des autres en comparaison."""
    fig = go.Figure()
    mc = monte_carlo[focus]
    days = np.arange(1, len(mc["mean"]) + 1)
    color = COLORS.get(focus, COLORWAY[0])
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]), y=np.concatenate([mc["p90"], mc["p10"][::-1]]),
        fill="toself", fillcolor=_rgba(color, 0.16), line=dict(width=0),
        name=f"{focus} — P10–P90", hoverinfo="skip",
    ))
    for name, other in monte_carlo.items():
        if name == focus:
            continue
        fig.add_trace(go.Scatter(
            x=days, y=other["mean"], name=name, line=dict(color=COLORS.get(name), width=1.2, dash="dot"),
            hovertemplate="Jour %{x}<br>%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=days, y=mc["mean"], name=f"{focus} — moyenne", line=dict(color=color, width=2.6),
        hovertemplate="Jour %{x}<br>%{y:.3f}<extra>" + focus + "</extra>",
    ))
    fig.update_layout(**_layout(440, xaxis=dict(title="Jours"), yaxis=dict(title="Valeur (base 1)")))
    return to_json(fig)


def monte_carlo_violin(monte_carlo: dict, horizon: int) -> str:
    fig = go.Figure()
    for name, mc in monte_carlo.items():
        color = COLORS.get(name, COLORWAY[0])
        fig.add_trace(go.Violin(
            y=mc["final"], name=name, fillcolor=_rgba(color, 0.45), line=dict(color=color, width=1),
            box_visible=True, meanline_visible=True, points=False, showlegend=False,
        ))
    fig.update_layout(**_layout(380, yaxis=dict(title=f"Valeur finale à {horizon} jours")))
    return to_json(fig)


# ── Black-Litterman ──────────────────────────────────────────────────────────

def grouped_weights(tickers, reference, bl) -> str:
    fig = go.Figure()
    for name, values in (("Référence", reference), ("Black-Litterman", bl)):
        fig.add_trace(go.Bar(name=name, x=tickers, y=values, marker_color=COLORS[name],
                             hovertemplate="%{x}<br>%{y:.2%}<extra>" + name + "</extra>"))
    fig.update_layout(**_layout(320, barmode="group", yaxis=dict(tickformat=".0%", title="Poids")))
    return to_json(fig)
