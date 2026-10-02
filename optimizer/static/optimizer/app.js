// Portfolio Optimizer — interactions côté navigateur (sans dépendance hormis Plotly.js)
(function () {
  "use strict";

  const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  // ── Graphiques : rendu à la demande (seulement quand ils sont visibles) ──
  function renderVisibleCharts(root) {
    (root || document).querySelectorAll(".chart:not([data-rendered])").forEach((el) => {
      if (el.offsetParent === null) return; // masqué : rendu quand il deviendra visible
      const script = el.querySelector('script[type="application/json"]');
      if (!script || !window.Plotly) return;
      const fig = JSON.parse(script.textContent);
      el.dataset.rendered = "1";
      // Hauteur explicite : sinon, au redimensionnement de la fenêtre, Plotly (responsive)
      // relit la hauteur du conteneur, la trouve nulle et le graphique déborde de sa carte.
      el.style.height = `${(fig.layout && fig.layout.height) || 400}px`;
      Plotly.newPlot(el, fig.data, fig.layout, PLOT_CONFIG);
    });
  }

  function resizeCharts(root) {
    (root || document).querySelectorAll(".chart[data-rendered]").forEach((el) => {
      if (el.offsetParent !== null) Plotly.Plots.resize(el);
    });
  }

  // ── Onglets ──
  function initTabs() {
    const tabs = document.querySelectorAll(".tabs [data-tab]");
    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.hidden = panel.dataset.panel !== btn.dataset.tab;
        });
        const panel = document.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`);
        renderVisibleCharts(panel);
        resizeCharts(panel);
        try { sessionStorage.setItem("po-tab", btn.dataset.tab); } catch (e) { /* stockage indisponible */ }
      });
    });
    let saved = null;
    try { saved = sessionStorage.getItem("po-tab"); } catch (e) { /* stockage indisponible */ }
    const target = saved && document.querySelector(`.tabs [data-tab="${saved}"]`);
    if (target) target.click();
  }

  // ── Sélecteur de portefeuille dans un groupe de graphiques ──
  function initGroups() {
    document.querySelectorAll("[data-group-select]").forEach((select) => {
      select.addEventListener("change", () => {
        const group = select.closest(".chart-group");
        group.querySelectorAll(".group-item").forEach((item) => {
          item.hidden = item.dataset.index !== select.value;
        });
        renderVisibleCharts(group);
        resizeCharts(group);
      });
    });
  }

  // ── Écran de chargement pendant l'analyse ──
  function initLoading() {
    const form = document.getElementById("analysis-form");
    const loading = document.getElementById("loading");
    if (!form || !loading) return;
    form.addEventListener("submit", () => {
      try { sessionStorage.removeItem("po-tab"); } catch (e) { /* stockage indisponible */ }
      loading.hidden = false;
    });
    // Retour arrière du navigateur : ne pas rester bloqué sur l'écran de chargement
    window.addEventListener("pageshow", () => { loading.hidden = true; });
  }

  // ── Black-Litterman ──
  function initBlackLitterman() {
    const form = document.getElementById("bl-form");
    if (!form) return;

    const custom = document.getElementById("custom-weights");
    form.querySelectorAll('input[name="ref"]').forEach((radio) => {
      radio.addEventListener("change", () => { custom.hidden = form.ref.value !== "custom"; });
    });

    const conf = document.getElementById("confidence");
    const confOut = document.getElementById("conf-out");
    const showConf = () => { confOut.textContent = Number(conf.value).toFixed(2).replace(".", ","); };
    conf.addEventListener("input", showConf);
    showConf();

    const views = document.getElementById("views");
    document.getElementById("add-view").addEventListener("click", () => {
      const rows = views.querySelectorAll(".view-row");
      if (rows.length >= 10) return;
      const clone = rows[rows.length - 1].cloneNode(true);
      views.appendChild(clone);
    });
    views.addEventListener("click", (event) => {
      const btn = event.target.closest("[data-remove-view]");
      if (!btn) return;
      if (views.querySelectorAll(".view-row").length > 1) btn.closest(".view-row").remove();
    });

    const result = document.getElementById("bl-result");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const submit = form.querySelector('button[type="submit"]');
      submit.disabled = true;
      submit.textContent = "Calcul en cours…";
      try {
        const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
        result.innerHTML = await response.text();
        renderVisibleCharts(result);
        result.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (err) {
        result.innerHTML = '<div class="alert alert-error">Erreur réseau : réessayez.</div>';
      } finally {
        submit.disabled = false;
        submit.textContent = "Calculer Black-Litterman";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initGroups();
    initLoading();
    initBlackLitterman();
    renderVisibleCharts();
  });
})();
