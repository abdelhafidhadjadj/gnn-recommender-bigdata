"""
Rapport de comparaison standard vs bigdata distribué.

Lit les fichiers outputs/<model>_w<N>_<size>/metrics/<model>_metrics.json
et calcule :
  - Speedup    : t_train(w1) / t_train(wN)
  - Efficiency : speedup / N  (utilisation réelle des workers)
  - Toutes les métriques de qualité (RMSE, MAE, NDCG...)

Usage :
    python3.13 scripts/compare_distributed.py --model sage --size 1k
    python3.13 scripts/compare_distributed.py --model sage --size 50k
    python3.13 scripts/compare_distributed.py --all
    python3.13 scripts/compare_distributed.py --all --html          # rapport HTML
    python3.13 scripts/compare_distributed.py --all --html --no-browser  # sans ouvrir le navigateur
"""
from __future__ import annotations
import argparse
import json
import os
import webbrowser
import tempfile
from datetime import datetime
from pathlib import Path


WORKERS  = [1, 2, 3, 4]
MODELS   = ["sage", "gat", "lightgcn"]
SIZES    = ["1k", "5k", "10k", "50k", "100k", "full"]


def load_metrics(model: str, workers: int, size: str,
                 output_base: str = "outputs") -> dict | None:
    path = Path(output_base) / f"{model}_w{workers}_{size}" / "metrics" / f"{model}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_speedup(base_time: float, other_time: float) -> float:
    if other_time <= 0:
        return 0.0
    return round(base_time / other_time, 2)


def compute_efficiency(speedup: float, n_workers: int) -> float:
    if n_workers <= 0:
        return 0.0
    return round(speedup / n_workers * 100, 1)


def print_report(model: str, size: str, output_base: str = "outputs") -> bool:
    # Charger toutes les configs disponibles
    data = {}
    for w in WORKERS:
        m = load_metrics(model, w, size, output_base)
        if m:
            data[w] = m

    if not data:
        print(f"  Aucune donnée trouvée pour {model} @ {size}")
        return False

    base_time = data.get(1, {}).get("timings", {}).get("t_train", 0)

    print(f"\n{'='*65}")
    print(f"  {model.upper()} @ {size}")
    print(f"{'='*65}")

    # ── Loader backend (pandas vs spark) ────────────────────────────────────
    print(f"\n  {'Backend':<20} " + "  ".join(f"{'w'+str(w):>10}" for w in data))
    print(f"  {'-'*60}")
    loader_row = f"  {'t_load backend':<20} "
    for w, m in data.items():
        loader = m.get("timings", {}).get("loader", "pandas" if w == 1 else "spark")
        loader_row += f"  {loader:>10}"
    print(loader_row)

    # ── Timings système ───────────────────────────────────────────────────────
    print(f"\n  {'Métrique':<20} " + "  ".join(f"{'w'+str(w):>10}" for w in data))
    print(f"  {'-'*60}")

    timing_keys = ["t_load", "t_sbert", "t_graph", "t_train", "t_eval"]
    for key in timing_keys:
        row = f"  {key:<20} "
        for w, m in data.items():
            val = m.get("timings", {}).get(key, "N/A")
            row += f"  {str(val)+'s':>10}" if val != "N/A" else f"  {'N/A':>10}"
        print(row)

    # ── Speedup et efficacité ─────────────────────────────────────────────────
    print(f"\n  {'Speedup':<20} " + "  ".join(f"{'w'+str(w):>10}" for w in data))
    print(f"  {'-'*60}")

    speedups = {}
    for w, m in data.items():
        t = m.get("timings", {}).get("t_train", 0)
        speedups[w] = compute_speedup(base_time, t) if base_time > 0 else 1.0

    row_sp  = f"  {'Speedup (×)':<20} "
    row_eff = f"  {'Efficiency (%)':<20} "
    for w in data:
        sp  = speedups[w]
        eff = compute_efficiency(sp, w)
        row_sp  += f"  {str(sp)+'×':>10}"
        row_eff += f"  {str(eff)+'%':>10}"
    print(row_sp)
    print(row_eff)

    # ── Métriques de qualité ──────────────────────────────────────────────────
    print(f"\n  {'Qualité':<20} " + "  ".join(f"{'w'+str(w):>10}" for w in data))
    print(f"  {'-'*60}")

    quality_keys = [
        ("RMSE",       lambda m: m.get("rmse")),
        ("MAE",        lambda m: m.get("mae")),
        ("Accuracy",   lambda m: m.get("accuracy")),
        ("Precision",  lambda m: m.get("global_precision")),
        ("NDCG@10",    lambda m: m.get("ranking", {}).get("10", {}).get("NDCG")),
        ("Recall@10",  lambda m: m.get("ranking", {}).get("10", {}).get("R")),
    ]

    for label, getter in quality_keys:
        row = f"  {label:<20} "
        for w, m in data.items():
            val = getter(m)
            row += f"  {f'{val:.4f}':>10}" if val is not None else f"  {'N/A':>10}"
        print(row)

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n  Résumé speedup t_train :")
    for w in data:
        loader = data[w].get("timings", {}).get("loader", "pandas" if w == 1 else "spark")
        t_load = data[w].get("timings", {}).get("t_load", "?")
        if w == 1:
            t_tr = data[w].get("timings", {}).get("t_train", "?")
            print(f"    w{w} (standard)  : référence  t_train={t_tr}s  t_load={t_load}s [{loader}]")
        else:
            t = data[w].get("timings", {}).get("t_train", "?")
            sp = speedups[w]
            eff = compute_efficiency(sp, w)
            print(f"    w{w} (bigdata)   : {sp}× plus rapide  t_train={t}s  t_load={t_load}s [{loader}]  efficacité={eff}%")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

# ── Palette ───────────────────────────────────────────────────────────────────
_W_COLORS   = ["#60a5fa","#34d399","#fbbf24","#a78bfa"]   # w1 w2 w3 w4
_W_BG       = ["rgba(96,165,250,.75)","rgba(52,211,153,.75)",
               "rgba(251,191,36,.75)","rgba(167,139,250,.75)"]
_MODEL_COL  = {"sage":"#60a5fa","gat":"#f472b6","lightgcn":"#34d399"}
_SIZE_ORDER = ["1k","5k","10k","50k","100k","full"]

def _sp_class(sp: float) -> str:
    if sp >= 1.8: return "sp-good"
    if sp >= 1.2: return "sp-ok"
    return "sp-bad"

def _js(v):
    """Python value → safe JS literal (None → null, float kept)."""
    if v is None:  return "null"
    if isinstance(v, float): return f"{v:.4f}"
    return _json.dumps(v)

def _js_arr(lst):
    return "[" + ",".join(_js(v) for v in lst) + "]"


def build_html_report(all_results: list[dict], output_base: str = "outputs") -> str:
    """Génère le dashboard HTML complet avec graphiques Chart.js."""

    if not all_results:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            "<!DOCTYPE html><html><body style='background:#0f1117;color:#f87171;"
            "font-family:sans-serif;padding:40px'><h2>Aucune donnée trouvée dans "
            f"outputs/</h2><p>Généré le {ts}</p></body></html>"
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sizes_found  = [s for s in _SIZE_ORDER
                    if any(r["size"] == s for r in all_results)]
    models_found = [m for m in MODELS
                    if any(r["model"] == m for r in all_results)]
    workers_found = sorted({w for r in all_results for w in r["data"]})

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lookup(model, size, w):
        for r in all_results:
            if r["model"] == model and r["size"] == size:
                return r["data"].get(w, {})
        return {}

    def _timing(model, size, w, key):
        return _lookup(model, size, w).get("timings", {}).get(key)

    def _quality(model, size, w, key, sub=None):
        m = _lookup(model, size, w)
        if sub:
            return (m.get("ranking") or {}).get("10", {}).get(sub)
        return m.get(key)

    def _speedup(model, size, w):
        base = _timing(model, size, 1, "t_train")
        t    = _timing(model, size, w, "t_train")
        return round(base / t, 3) if base and t and t > 0 else None

    # ── KPI ───────────────────────────────────────────────────────────────────
    total_runs = sum(len(r["data"]) for r in all_results)
    best_sp    = max(
        (_speedup(r["model"], r["size"], w) or 1.0
         for r in all_results for w in r["data"] if w > 1),
        default=1.0,
    )
    best_ndcg = max(
        (_quality(r["model"], r["size"], 1, None, "NDCG") or 0.0
         for r in all_results),
        default=0.0,
    )

    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-val">{total_runs}</div>
        <div class="kpi-label">Runs totaux</div></div>
      <div class="kpi"><div class="kpi-val">{len(models_found)}</div>
        <div class="kpi-label">Modèles</div></div>
      <div class="kpi"><div class="kpi-val">{len(sizes_found)}</div>
        <div class="kpi-label">Tailles dataset</div></div>
      <div class="kpi"><div class="kpi-val kpi-green">{best_sp}×</div>
        <div class="kpi-label">Meilleur speedup</div></div>
      <div class="kpi"><div class="kpi-val kpi-blue">{best_ndcg:.4f}</div>
        <div class="kpi-label">Meilleur NDCG@10</div></div>
    </div>"""

    # ── Vue globale — graphiques croisés ──────────────────────────────────────
    cross_html    = ""
    cross_scripts = ""

    if models_found and sizes_found:
        m_labels = _json.dumps([m.upper() for m in models_found])
        s_labels = _json.dumps(sizes_found)
        w_labels = _json.dumps([f"w{w}" for w in workers_found])

        # 1) Speedup max par modèle × taille (grouped bar)
        sp_datasets = []
        for i, size in enumerate(sizes_found):
            vals = []
            for model in models_found:
                best = max(
                    (_speedup(model, size, w) or 1.0
                     for w in workers_found if w > 1),
                    default=1.0,
                )
                vals.append(round(best, 3))
            sp_datasets.append({
                "label": size,
                "data": vals,
                "backgroundColor": f"hsl({i*55},70%,55%)",
                "borderRadius": 5,
                "borderWidth": 1,
            })

        # 2) NDCG@10 par modèle × taille (line)
        ndcg_datasets = []
        for i, size in enumerate(sizes_found):
            vals = [_quality(m, size, 1, None, "NDCG") for m in models_found]
            ndcg_datasets.append({
                "label": size,
                "data": vals,
                "borderColor": f"hsl({i*55},70%,60%)",
                "backgroundColor": f"hsla({i*55},70%,60%,.15)",
                "tension": 0.35,
                "pointRadius": 5,
                "fill": False,
                "borderWidth": 2,
            })

        # 3) t_train par modèle × taille × worker (grouped — un dataset par worker)
        ttime_datasets = []
        for wi, w in enumerate(workers_found):
            vals = []
            for model in models_found:
                # moyenne sur toutes les tailles disponibles
                ts_vals = [_timing(model, sz, w, "t_train")
                           for sz in sizes_found
                           if _timing(model, sz, w, "t_train") is not None]
                vals.append(round(sum(ts_vals)/len(ts_vals), 2) if ts_vals else None)
            ttime_datasets.append({
                "label": f"w{w}",
                "data": vals,
                "backgroundColor": _W_BG[wi % 4],
                "borderColor":     _W_COLORS[wi % 4],
                "borderRadius": 5,
                "borderWidth": 2,
            })

        # 4) Heatmap speedup : tableau visuel (toutes tailles × tous workers × tous modèles)
        # → on le fait en graphique "bar" par taille, une valeur moyenne sur modèles
        sp_by_size = []
        for sz in sizes_found:
            row = []
            for w in workers_found:
                sps = [_speedup(m, sz, w) for m in models_found
                       if _speedup(m, sz, w) is not None]
                row.append(round(sum(sps)/len(sps), 3) if sps else None)
            sp_by_size.append(row)

        sp_by_w_datasets = []
        for wi, w in enumerate(workers_found):
            vals = [sp_by_size[si][wi] for si in range(len(sizes_found))]
            sp_by_w_datasets.append({
                "label": f"w{w}",
                "data": vals,
                "backgroundColor": _W_BG[wi % 4],
                "borderColor":     _W_COLORS[wi % 4],
                "borderRadius": 4,
                "borderWidth": 2,
            })

        def _ds(lst):
            lines = []
            for d in lst:
                parts = []
                for k, v in d.items():
                    if isinstance(v, str):
                        parts.append(f'  {k}: {_json.dumps(v)}')
                    elif isinstance(v, list):
                        parts.append(f'  {k}: {_js_arr(v)}')
                    elif isinstance(v, bool):
                        parts.append(f'  {k}: {"true" if v else "false"}')
                    elif v is None:
                        parts.append(f'  {k}: null')
                    else:
                        parts.append(f'  {k}: {v}')
                lines.append("{" + ", ".join(parts) + "}")
            return "[" + ",\n".join(lines) + "]"

        cross_html = f"""
        <div class="section-title" style="margin:0 0 18px">Vue globale — tous modèles &amp; tailles</div>
        <div class="cross-grid">
          <div class="chart-box-lg">
            <div class="chart-label">Meilleur speedup par modèle (max sur w2/3/4)</div>
            <canvas id="cg_sp_model"></canvas></div>
          <div class="chart-box-lg">
            <div class="chart-label">NDCG@10 baseline w1 par modèle</div>
            <canvas id="cg_ndcg"></canvas></div>
          <div class="chart-box-lg">
            <div class="chart-label">t_train moyen par modèle (tous workers)</div>
            <canvas id="cg_ttime"></canvas></div>
          <div class="chart-box-lg">
            <div class="chart-label">Speedup moyen (tous modèles) par taille × workers</div>
            <canvas id="cg_sp_size"></canvas></div>
        </div>"""

        cross_scripts = f"""
        mkBar('cg_sp_model',  {m_labels}, {_ds(sp_datasets)},  'Speedup ×', v => v+'×');
        mkLine('cg_ndcg',     {m_labels}, {_ds(ndcg_datasets)}, 'NDCG@10');
        mkBar('cg_ttime',     {m_labels}, {_ds(ttime_datasets)}, 'Temps moyen (s)', v => v+'s');
        mkBar('cg_sp_size',   {s_labels}, {_ds(sp_by_w_datasets)}, 'Speedup ×', v => v+'×');
        """

    # ── Sections par modèle × taille ─────────────────────────────────────────
    sections_html = ""
    detail_scripts = ""

    TIMING_KEYS  = ["t_load","t_sbert","t_graph","t_train","t_eval"]
    QUALITY_DEFS = [
        ("RMSE",      lambda m: m.get("rmse")),
        ("MAE",       lambda m: m.get("mae")),
        ("Accuracy",  lambda m: m.get("accuracy")),
        ("Precision", lambda m: m.get("global_precision")),
        ("NDCG@10",   lambda m: (m.get("ranking") or {}).get("10", {}).get("NDCG")),
        ("Recall@10", lambda m: (m.get("ranking") or {}).get("10", {}).get("R")),
        ("MRR@10",    lambda m: (m.get("ranking") or {}).get("10", {}).get("MRR")),
        ("HR@10",     lambda m: (m.get("ranking") or {}).get("10", {}).get("HR")),
    ]

    for idx, r in enumerate(all_results):
        model, size = r["model"], r["size"]
        data = r["data"]
        ws   = sorted(data.keys())
        if not ws:
            continue

        base_t  = data.get(1, {}).get("timings", {}).get("t_train", 0)
        speedups = {
            w: round(base_t / t, 3)
            if (t := data[w].get("timings", {}).get("t_train", 0)) and base_t > 0
            else 1.0
            for w in ws
        }

        mc = _MODEL_COL.get(model, "#888")
        w_labels_local = _json.dumps([f"w{w}" for w in ws])

        # ── sparkline speedup ─────────────────────────────────────────────────
        sp_vals_js  = _js_arr([speedups[w] for w in ws])
        sp_bg_js    = _json.dumps([_W_BG[w-1] for w in ws])
        sp_col_js   = _json.dumps([_W_COLORS[w-1] for w in ws])

        # ── timings stacked ───────────────────────────────────────────────────
        t_ds_parts = []
        for key, color in [("t_load","#60a5fa"),("t_train","#34d399"),
                            ("t_sbert","#f472b6"),("t_eval","#fbbf24")]:
            vals = [data[w].get("timings", {}).get(key) for w in ws]
            t_ds_parts.append(
                f'{{label:{_json.dumps(key)},data:{_js_arr(vals)},'
                f'backgroundColor:"{color}aa",borderColor:"{color}",'
                f'borderWidth:1}}'
            )
        t_ds_js = "[" + ",".join(t_ds_parts) + "]"

        # ── radar qualité ─────────────────────────────────────────────────────
        q_labels_js = _json.dumps(["RMSE","MAE","Acc","Prec","NDCG@10","Recall","MRR","HR"])
        q_getters   = [fn for _, fn in QUALITY_DEFS]
        q_ds_parts  = []
        for wi, w in enumerate(ws):
            vals = [g(data[w]) for g in q_getters]
            q_ds_parts.append(
                f'{{label:"w{w}",data:{_js_arr(vals)},'
                f'backgroundColor:"{_W_BG[wi%4]}",'
                f'borderColor:"{_W_COLORS[wi%4]}",borderWidth:2}}'
            )
        q_ds_js = "[" + ",".join(q_ds_parts) + "]"

        # ── HTML table ────────────────────────────────────────────────────────
        th = "".join(
            f'<th>w{w} <span class="{"badge-std" if w==1 else "badge-big"}">'
            f'{"std" if w==1 else "big"}</span></th>'
            for w in ws
        )
        def td_sp(w):
            sp = speedups[w]
            return f'<td class="{_sp_class(sp)}">{sp}×</td>'
        def td_eff(w):
            return f'<td>{round(speedups[w]/w*100,1)}%</td>'
        def td_t(w, k):
            v = data[w].get("timings", {}).get(k, "N/A")
            return f'<td>{"N/A" if v == "N/A" else str(v)+"s"}</td>'
        def td_q(w, fn):
            v = fn(data[w])
            return f'<td>{"N/A" if v is None else f"{v:.4f}"}</td>'

        sp_row  = "<tr><td>Speedup (×)</td>"   + "".join(td_sp(w)  for w in ws) + "</tr>"
        eff_row = "<tr><td>Efficiency (%)</td>" + "".join(td_eff(w) for w in ws) + "</tr>"
        t_rows  = "".join(
            f"<tr><td>{k}</td>" + "".join(td_t(w, k) for w in ws) + "</tr>"
            for k in TIMING_KEYS
        )
        q_rows  = "".join(
            f"<tr><td>{label}</td>" + "".join(td_q(w, fn) for w in ws) + "</tr>"
            for label, fn in QUALITY_DEFS
        )

        cid_sp = f"d_sp_{idx}"
        cid_t  = f"d_t_{idx}"
        cid_q  = f"d_q_{idx}"

        detail_scripts += f"""
        mkBarMulti('{cid_sp}', {w_labels_local},
          [{{label:'Speedup',data:{sp_vals_js},backgroundColor:{sp_bg_js},
             borderColor:{sp_col_js},borderWidth:2,borderRadius:6}}],
          'Speedup t_train', v => v+'×');
        mkBar('{cid_t}', {w_labels_local}, {t_ds_js}, 'Timings (s)', v => v+'s', true);
        mkRadar('{cid_q}', {q_labels_js}, {q_ds_js});
        """

        sections_html += f"""
        <div class="card">
          <div class="card-header">
            <span class="model-badge"
              style="background:{mc}22;color:{mc};border:1px solid {mc}55">{model.upper()}</span>
            <span class="size-badge">{size}</span>
            <span class="runs-info">{len(ws)} worker(s) · {len(ws)*1} run(s)</span>
          </div>
          <div class="charts-row">
            <div class="chart-box"><canvas id="{cid_sp}"></canvas></div>
            <div class="chart-box"><canvas id="{cid_t}"></canvas></div>
            <div class="chart-box"><canvas id="{cid_q}"></canvas></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Métrique</th>{th}</tr></thead>
              <tbody>{sp_row}{eff_row}{t_rows}{q_rows}</tbody>
            </table>
          </div>
        </div>"""

    # ── Assemblage final ──────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GNN Dashboard — Distribué</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e2e8f0;padding:28px 32px}}
    .header{{margin-bottom:28px}}
    .header h1{{font-size:1.7rem;color:#7dd3fc;letter-spacing:-.5px}}
    .header h1 span{{color:#a78bfa}}
    .subtitle{{color:#64748b;font-size:.82rem;margin-top:4px}}
    .kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:32px}}
    .kpi{{background:#161b27;border:1px solid #1e2740;border-radius:12px;padding:16px 22px;
          flex:1;min-width:120px;text-align:center}}
    .kpi-val{{font-size:1.6rem;font-weight:800;color:#f1f5f9}}
    .kpi-green{{color:#34d399}}.kpi-blue{{color:#60a5fa}}
    .kpi-label{{font-size:.73rem;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}}
    .section-title{{font-size:1.1rem;font-weight:700;color:#a78bfa;
                    border-left:3px solid #a78bfa;padding-left:10px}}
    .cross-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-bottom:36px}}
    .chart-box-lg{{background:#161b27;border:1px solid #1e2740;border-radius:10px;
                   padding:14px;height:270px;position:relative}}
    .card{{background:#161b27;border:1px solid #1e2740;border-radius:14px;
           padding:20px 22px;margin-bottom:24px}}
    .card-header{{display:flex;align-items:center;gap:10px;margin-bottom:18px}}
    .model-badge{{font-size:.82rem;font-weight:700;padding:3px 12px;border-radius:99px;letter-spacing:.5px}}
    .size-badge{{background:#1e2740;color:#93c5fd;padding:3px 10px;border-radius:99px;font-size:.78rem}}
    .runs-info{{color:#475569;font-size:.76rem;margin-left:auto}}
    .charts-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}}
    .chart-box{{background:#0f1520;border-radius:10px;padding:14px;height:220px;position:relative}}
    .chart-label{{font-size:.78rem;color:#64748b;margin-bottom:8px;text-align:center}}
    .table-wrap{{overflow-x:auto}}
    table{{width:100%;border-collapse:collapse;font-size:.83rem}}
    thead th{{background:#1e2740;color:#7dd3fc;padding:8px 14px;text-align:center;
              border-bottom:2px solid #2d3a55}}
    thead th:first-child{{text-align:left}}
    tbody tr:nth-child(even){{background:#111722}}
    tbody tr:hover{{background:#1a2236}}
    td{{padding:6px 14px;text-align:center;border-bottom:1px solid #1a2236}}
    td:first-child{{text-align:left;color:#94a3b8;font-weight:500}}
    .badge-std{{background:#1d4ed833;color:#93c5fd;padding:1px 7px;border-radius:99px;font-size:.72rem}}
    .badge-big{{background:#06564633;color:#6ee7b7;padding:1px 7px;border-radius:99px;font-size:.72rem}}
    .sp-good{{color:#34d399;font-weight:700}}.sp-ok{{color:#fbbf24;font-weight:700}}
    .sp-bad{{color:#f87171;font-weight:700}}
    @media(max-width:900px){{.charts-row,.cross-grid{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
  <div class="header">
    <h1>GNN Recommender <span>— Dashboard Distribué</span></h1>
    <p class="subtitle">Généré le {ts} &nbsp;·&nbsp; Standard (pandas) vs Bigdata (Spark+HDFS)
      &nbsp;·&nbsp; {os.path.abspath(output_base)}</p>
  </div>

  {kpi_html}
  {cross_html}

  <div class="section-title" style="margin-bottom:20px">Résultats détaillés par modèle &amp; taille</div>
  {sections_html}

  <script>
    Chart.defaults.color='#94a3b8';
    Chart.defaults.borderColor='#1e2740';

    const OPT_AXES = (yFmt, stacked) => ({{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ labels:{{ color:'#94a3b8', boxWidth:12 }} }} }},
      scales:{{
        x:{{ stacked:!!stacked, ticks:{{ color:'#94a3b8' }}, grid:{{ color:'#1e2740' }} }},
        y:{{ stacked:!!stacked, beginAtZero:true,
             ticks:{{ color:'#94a3b8', callback: yFmt || (v=>v) }},
             grid:{{ color:'#1e2740' }} }}
      }}
    }});

    function mkBar(id, labels, datasets, title, yFmt, stacked) {{
      const el = document.getElementById(id); if(!el) return;
      new Chart(el, {{ type:'bar', data:{{labels, datasets}},
        options:{{ ...OPT_AXES(yFmt, stacked),
          plugins:{{ ...OPT_AXES(yFmt,stacked).plugins,
            title:{{ display:!!title, text:title||'', color:'#64748b', font:{{size:11}} }} }} }}
      }});
    }}

    function mkBarMulti(id, labels, datasets, title, yFmt) {{
      mkBar(id, labels, datasets, title, yFmt, false);
    }}

    function mkLine(id, labels, datasets, title) {{
      const el = document.getElementById(id); if(!el) return;
      new Chart(el, {{ type:'line', data:{{labels, datasets}},
        options:{{ responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{ labels:{{ color:'#94a3b8', boxWidth:12 }} }},
            title:{{ display:!!title, text:title||'', color:'#64748b', font:{{size:11}} }} }},
          scales:{{
            x:{{ ticks:{{ color:'#94a3b8' }}, grid:{{ color:'#1e2740' }} }},
            y:{{ beginAtZero:true, ticks:{{ color:'#94a3b8' }}, grid:{{ color:'#1e2740' }} }}
          }}
        }}
      }});
    }}

    function mkRadar(id, labels, datasets) {{
      const el = document.getElementById(id); if(!el) return;
      new Chart(el, {{ type:'radar', data:{{labels, datasets}},
        options:{{ responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{ labels:{{ color:'#94a3b8', boxWidth:12 }} }} }},
          scales:{{ r:{{ beginAtZero:true,
            ticks:{{ color:'#94a3b8', backdropColor:'transparent', font:{{size:9}} }},
            grid:{{ color:'#1e2740' }},
            pointLabels:{{ color:'#cbd5e1', font:{{size:10}} }} }} }}
        }}
      }});
    }}

    {cross_scripts}
    {detail_scripts}
  </script>
</body>
</html>"""


def collect_all_data(models: list[str], sizes: list[str],
                     output_base: str = "outputs") -> list[dict]:
    results = []
    for model in models:
        for size in sizes:
            data = {}
            for w in WORKERS:
                m = load_metrics(model, w, size, output_base)
                if m:
                    data[w] = m
            if data:
                results.append({"model": model, "size": size, "data": data})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Comparaison standard vs bigdata distribué")
    p.add_argument("--model",  default="sage",  choices=MODELS + ["all"])
    p.add_argument("--size",   default="1k",    choices=SIZES  + ["all"])
    p.add_argument("--output-base", default="outputs")
    p.add_argument("--all",    action="store_true", help="Comparer tous les modèles × toutes les tailles")
    p.add_argument("--html",   action="store_true", help="Générer un rapport HTML et l'ouvrir dans le navigateur")
    p.add_argument("--no-browser", action="store_true", help="Générer le HTML sans ouvrir le navigateur")
    p.add_argument("--out",    default=None, help="Chemin du fichier HTML (défaut: outputs/report.html)")
    args = p.parse_args()

    models = MODELS if (args.all or args.model == "all") else [args.model]
    sizes  = SIZES  if (args.all or args.size  == "all") else [args.size]

    # ── Mode HTML ─────────────────────────────────────────────────────────────
    if args.html:
        all_results = collect_all_data(models, sizes, args.output_base)
        html = build_html_report(all_results, args.output_base)

        out_path = Path(args.out) if args.out else Path(args.output_base) / "report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"[HTML] Rapport genere -> {out_path.resolve()}")

        if not args.no_browser:
            webbrowser.open(out_path.resolve().as_uri())
            print("[HTML] Ouverture dans le navigateur...")
        return

    # ── Mode terminal (comportement original) ─────────────────────────────────
    found = False
    for model in models:
        for size in sizes:
            if print_report(model, size, args.output_base):
                found = True

    if not found:
        print("\nAucune donnée trouvée.")
        print("Lancez d'abord :")
        print("  WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up")
        print("  WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up")
        print("  WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up")


if __name__ == "__main__":
    main()
