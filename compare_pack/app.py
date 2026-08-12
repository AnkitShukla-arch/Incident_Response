"""
Incident Memory — the demo page that wraps the whole pipeline.

The page puts the complete "memory compounds" loop behind one UI:

  ┌──────────────┐     1. embed the new incident      (new_ingest.embed_text)
  │ New incident │     2. retrieve similar past hits  (query_incidents)
  └──────────────┘     3. LLM suggests a fix          (suggest_fix)
        │              4. edit / approve the fix
        ▼              5. write it back to memory     (resolve_incident)
  ┌──────────────┐     6. postmortem doc on the row   (postmortem)
  │ Mark resolved │──▶ Postgres row + Qdrant vector → the memory grew by one
  └──────────────┘        + postmortem_doc column filled

Incidents can enter the pipeline two ways: the page's search button
(POST /api/search) or any HTTP POST to /webhook/incident (webhook.py) — both
call the exact same retrieve_and_suggest() function.

The incidents table at the bottom shows every row in Postgres, newest first,
so the growth is visible live (e.g. 12 → 13 → 14 …).

Run (from this folder, with the hackenv venv active):
    python app.py                  # http://localhost:5000
    python app.py --port 5001      # custom port

Endpoints:
    GET  /                            the page
    GET  /perf                        performance-matrix dashboard (reads the
                                      benchmark snapshot from perf_charts/)
    GET  /api/incidents               all incidents (newest first) + active/deleted counts
    POST /api/search                  {description, service?, severity?, top?, threshold?}
                                      -> {matches, suggestion, llm_error?}
    POST /api/resolve                 {description, suggestion, service?, severity?}
                                      -> stores the fix, then generates + saves the
                                         postmortem doc on the same row (write-back + doc)
    POST /webhook/incident            {title?, description, service?, severity?}
                                      validated by the IncidentIn schema (Pydantic, 422 on
                                      malformed bodies) -> same retrieve-and-suggest
                                      pipeline as /api/search, as a live HTTP service
    GET  /api/export                  download the full memory snapshot (rows + vectors)
    POST /api/import                  restore a snapshot JSON body -> {rows_created, ...}
    POST /api/incidents/<id>/delete   soft delete (undo-able): drop the Qdrant point, mark row
    POST /api/incidents/<id>/restore  undo a delete: re-embed + re-upsert the point
    POST /api/incidents/<id>/purge    permanent delete from both stores
    POST /api/incidents/<id>/postmortem  generate (or regenerate) + save the postmortem
                                      doc for an existing row -> {postmortem (markdown)}
"""

import argparse
import json
import math
import os
import sys

from flask import Flask, Response, jsonify, render_template, request
from pydantic import ValidationError
from qdrant_client.models import PointStruct

from memory_backup import export_memory, import_memory
from new_ingest import COLLECTION_NAME, connect_db, embed_text, qdrant_client
from postmortem import generate_postmortem, save_postmortem
from resolve_incident import ensure_schema, store_resolved_incident
from suggest_fix import retrieve_and_suggest
from webhook import IncidentIn, receive_incident

app = Flask(__name__)

PERF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_charts")
PERF_RESULTS_FILE = os.path.join(PERF_DIR, "performance_results.json")
PERF_HISTORY_FILE = os.path.join(PERF_DIR, "performance_history.json")

# Order + display names of the latency stages measured by benchmark_perf.py.
LATENCY_STAGES = [
    ("embedding", "Gemini embedding"),
    ("qdrant", "Qdrant search"),
    ("postgres", "Postgres enrichment"),
    ("retrieval_full", "Full retrieval (1+2+3)"),
    ("llm_suggestion", "Groq LLM suggestion"),
    ("postmortem", "Postmortem gen"),
    ("http_e2e", "HTTP /api/search (e2e)"),
    ("pipeline_full", "Full pipeline"),
]

# Apply the schema migrations (nullable root_cause + postmortem_doc column)
# no matter how the app is launched — `python app.py`, `flask run`, a WSGI
# server, or the test client all import this module. The /api routes SELECT
# postmortem_doc, so an unmigrated database would 500 on them. Non-fatal on
# failure: the app still starts and the routes report the DB error as JSON.
try:
    with connect_db() as conn:
        ensure_schema(conn)
except (SystemExit, Exception):  # noqa: BLE001 - schema setup must never block startup
    pass

SERVICES = [
    "payments", "checkout", "notifications", "reporting", "web-frontend",
    "auth", "api-gateway", "search", "recommendations", "database", "other",
]
SEVERITIES = ["critical", "high", "medium", "low"]



# --- Routes ----------------------------------------------------------------


@app.get("/")
def index():
    return render_template("index.html", services=SERVICES, severities=SEVERITIES)


def _build_perf_view(data, history, error=None) -> dict:
    """Flatten the benchmark snapshot JSON into chart-ready structures so the
    /perf template stays math-free. Returns the kwargs for render_template.
    """
    view = {"data": data, "error": error,
            "latency": [], "recall": [], "donut": None, "histogram": None,
            "trend": None, "kpis": [], "system": {}, "generated_at": "—"}

    if not isinstance(data, dict) or "retrieval_quality" not in data:
        return view

    rq = data.get("retrieval_quality") or {}
    rl = data.get("related_queries") or {}
    lat = data.get("latency") or {}
    tp = data.get("throughput") or {}
    th = data.get("thresholds") or {}
    meta = data.get("system") or {}
    view["system"] = meta
    view["generated_at"] = data.get("generated_at", "—")

    tot = max(rq.get("total", 1), 1)
    rl_tot = max(rl.get("total", 1), 1)

    # ---- KPI cards -------------------------------------------------------
    def pct_str(n, d):
        return f"{n}/{d} ({100.0 * n / max(d, 1):.0f}%)"

    synced = bool(meta.get("synced"))
    kpis = [
        ("Recall@5", pct_str(rq.get("recall5", 0), tot),
         "green" if 100.0 * rq.get("recall5", 0) / tot >= 90 else "red",
         "self-retrieval over all active incidents"),
        ("MRR@5", f"{rq.get('mrr', 0):.4f}", "cyan", "mean reciprocal rank"),
        ("Full pipeline", f"{((lat.get('pipeline_full') or {}).get('mean_ms', 0)):.0f} ms",
         "accent", "retrieve + LLM, mean per call"),
        ("Throughput", f"{tp.get('pipeline_per_min', 0):.1f} inc/min",
         "amber", "sequential full-pipeline calls"),
        ("Related Recall@5", pct_str(rl.get("recall5", 0), rl_tot),
         "green", "8 realistic new-incident queries"),
        ("Stores in sync", f"{meta.get('pg_active', 0)} PG = {meta.get('qdrant', 0)} Qdrant",
         "cyan" if synced else "red", "Postgres rows vs Qdrant points"),
    ]
    view["kpis"] = [{"k": k, "v": v, "c": c, "s": s} for k, v, c, s in kpis]

    # ---- latency matrix rows (widths relative to the slowest p95) --------
    rows = []
    for key, name in LATENCY_STAGES:
        s = lat.get(key) or {}
        rows.append({"name": name, "n": s.get("n", 0),
                     "mean": s.get("mean_ms", 0.0), "p50": s.get("p50_ms", 0.0),
                     "p95": s.get("p95_ms", 0.0)})
    max_p95 = max([r["p95"] for r in rows] or [1.0]) or 1.0
    for r in rows:
        r["w_mean"] = min(100.0, 100.0 * r["mean"] / max_p95)
        r["w_p50"] = min(100.0, 100.0 * r["p50"] / max_p95)
        r["w_p95"] = min(100.0, 100.0 * r["p95"] / max_p95)
    view["latency"] = {"rows": rows, "max_p95": max_p95}

    # ---- recall bars -----------------------------------------------------
    view["recall"] = [
        {"lab": "Self<br>Recall@1", "v": 100.0 * rq.get("recall1", 0) / tot, "c": "#5e6ad2"},
        {"lab": "Self<br>Recall@3", "v": 100.0 * rq.get("recall3", 0) / tot, "c": "#22d3ee"},
        {"lab": "Self<br>Recall@5", "v": 100.0 * rq.get("recall5", 0) / tot, "c": "#4ade80"},
        {"lab": "Related<br>Recall@1", "v": 100.0 * rl.get("recall1", 0) / rl_tot, "c": "#fbbf24"},
        {"lab": "Related<br>Recall@5", "v": 100.0 * rl.get("recall5", 0) / rl_tot, "c": "#4ade80"},
    ]

    # ---- latency-budget donut (the stages that make up /api/search) ------
    def lat_ms(key):
        return (lat.get(key) or {}).get("mean_ms", 0.0)

    emb, qdr = lat_ms("embedding"), lat_ms("qdrant")
    pg, llm = lat_ms("postgres"), lat_ms("llm_suggestion")
    total_ms = (emb + qdr + pg + llm) or 1.0
    circumference = 2 * math.pi * 60
    segments, acc = [], 0.0
    for label, ms, color in (
        ("Postgres enrichment", pg, "#f87171"),
        ("Gemini embedding", emb, "#fbbf24"),
        ("Groq LLM", llm, "#a78bfa"),
        ("Qdrant search", qdr, "#22d3ee"),
    ):
        dash = circumference * ms / total_ms
        segments.append({"label": label, "ms": round(ms, 1), "color": color,
                         "dash": dash, "offset": acc,
                         "pct": round(100.0 * ms / total_ms, 1)})
        acc += dash
    view["donut"] = {"segments": segments, "total": round(total_ms, 1),
                      "circumference": circumference}

    # ---- top-1 score histogram (0.60 -> 1.00, 0.02 bins) -----------------
    scores = rq.get("scores_first") or []
    bins, lo, step = [], 0.60, 0.02
    for i in range(20):
        bl = lo + i * step
        bins.append({"label": f"{bl:.2f}",
                     "count": sum(1 for s in scores if bl <= s < bl + step)})
    max_count = max([b["count"] for b in bins] or [1]) or 1
    for b in bins:
        b["h"] = 100.0 * b["count"] / max_count
        b["cls"] = ("ok" if float(b["label"]) >= th.get("cli", 0.3)
                     else "mid" if float(b["label"]) >= th.get("app", 0.22)
                     else "low")
    view["histogram"] = {"bins": bins, "app": th.get("app", 0.22),
                          "cli": th.get("cli", 0.3),
                          "scores": {"mean": th.get("scores_mean"),
                                      "min": th.get("scores_min"),
                                      "max": th.get("scores_max")},
                          "over_app": th.get("over_app"), "over_cli": th.get("over_cli")}

    # ---- trend lines from the append-only history (needs 2+ runs) --------
    runs = []
    for h in history or []:
        q = h.get("quality") or {}
        lm = h.get("latency_ms") or {}
        runs.append({"run": h.get("run", 0),
                     "label": (h.get("generated_at") or "")[11:16],
                     "recall5": 100.0 * q.get("recall5", 0) / max(q.get("total", 1), 1),
                     "pipe_ms": lm.get("pipeline_full", 0.0)})
    if len(runs) >= 2:
        W, H, pl, pr, pt, pb = 660, 180, 46, 64, 16, 34
        xs = [pl + i * (W - pl - pr) / (len(runs) - 1) for i in range(len(runs))]

        def series(vals):
            vmin, vmax = min(vals), max(vals)
            span = (vmax - vmin) or 1.0
            pts = []
            for x, v in zip(xs, vals):
                y = pt + (H - pt - pb) * (1 - (v - vmin) / span)
                pts.append((x, y))
            return pts

        pts_r5 = series([r["recall5"] for r in runs])
        pts_pm = series([r["pipe_ms"] for r in runs])
        view["trend"] = {
            "W": W, "H": H,
            "r5_poly": " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_r5),
            "pm_poly": " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_pm),
            "r5_dots": " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_r5),
            "pm_dots": " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_pm),
            "labels": [f"#{r['run']}" for r in runs],
            "xpos": [f"{x:.1f}" for x in xs],
            "end_r5": {"x": pts_r5[-1][0], "y": pts_r5[-1][1],
                        "v": round(runs[-1]["recall5"], 1)},
            "end_pm": {"x": pts_pm[-1][0], "y": pts_pm[-1][1],
                        "v": round(runs[-1]["pipe_ms"], 0)},
        }
    return view


@app.get("/perf")
def api_perf():
    """Performance-matrix dashboard: renders the latest benchmark snapshot
    (perf_charts/performance_results.json, written by benchmark_perf.py) as a
    page at /perf. Pure file read — no Postgres/Qdrant access — so it works
    even when the stores are down.
    """
    data, error = None, None
    try:
        with open(PERF_RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        error = f"Could not read {PERF_RESULTS_FILE}: {exc}"

    history = []
    try:
        with open(PERF_HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        history = []

    return render_template("perf.html", **_build_perf_view(data, history, error))


@app.get("/api/incidents")
def api_incidents():
    try:
        with connect_db() as conn:
            rows = conn.execute(
                "SELECT id, title, description, root_cause, resolution, service, "
                "severity, status, created_at, postmortem_doc "
                "FROM incidents ORDER BY id DESC"
            ).fetchall()
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 503
    incidents = [
        {
            "id": r[0], "title": r[1], "description": r[2], "root_cause": r[3],
            "resolution": r[4], "service": r[5], "severity": r[6],
            "status": r[7], "created_at": str(r[8]), "postmortem_doc": r[9],
        }
        for r in rows
    ]
    total = sum(1 for r in rows if r[7] != "deleted")
    deleted = len(rows) - total
    return jsonify(ok=True, total=total, deleted=deleted, incidents=incidents)


@app.post("/api/search")
def api_search():
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return jsonify(ok=False, error="Missing incident description."), 400

    try:
        top = max(1, min(int(data.get("top", 5)), 10))
    except (TypeError, ValueError):
        top = 5
    try:
        threshold = float(data.get("threshold", 0.22))
    except (TypeError, ValueError):
        threshold = 0.22

    try:
        # 1 + 2 + 3: embed the query, retrieve similar past incidents from
        # Qdrant/Postgres, and ask the LLM for a suggested fix — the same
        # pipeline the /webhook/incident route runs.
        out = retrieve_and_suggest(description, top, threshold)
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 400

    return jsonify(ok=True, **out)


@app.post("/api/resolve")
def api_resolve():
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    suggestion = (data.get("suggestion") or "").strip()
    service = (data.get("service") or "").strip() or None
    severity = (data.get("severity") or "").strip() or None

    if not description:
        return jsonify(ok=False, error="Missing incident description."), 400
    if not suggestion:
        return jsonify(ok=False, error="Missing the fix text to save."), 400

    # 4 + 5: write the approved fix back into Postgres + Qdrant (write-back).
    try:
        incident = store_resolved_incident(
            description=description,
            resolution=suggestion,
            service=service,
            severity=severity,
        )
        with connect_db() as conn:
            total = conn.execute(
                "SELECT count(*) FROM incidents WHERE status <> 'deleted'"
            ).fetchone()[0]
    except (ValueError, SystemExit) as exc:
        return jsonify(ok=False, error=str(exc)), 400 if isinstance(exc, ValueError) else 500

    # Postmortem (runs immediately after the write-back, not instead of it):
    # expand the short resolution text into a full document and save it on the
    # same row. If the LLM call fails the write-back already succeeded, so the
    # resolve is reported as ok and the failure is surfaced in postmortem_error.
    postmortem, postmortem_error = None, None
    try:
        doc = generate_postmortem(incident)
        with connect_db() as conn:
            save_postmortem(conn, incident["id"], doc)
        postmortem = {"incident_id": incident["id"], "characters": len(doc)}
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - never fail the resolve
        postmortem_error = str(exc)

    return jsonify(ok=True, incident=incident, total=total,
                   postmortem=postmortem, postmortem_error=postmortem_error)


@app.get("/api/export")
def api_export():
    """Download a full snapshot (rows + vectors + embedding metadata)."""
    try:
        snapshot = export_memory()
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 500
    resp = Response(json.dumps(snapshot, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = "attachment; filename=incidents_backup.json"
    return resp


@app.post("/api/import")
def api_import():
    """Restore a snapshot sent as the raw JSON body (the exported file)."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Body must be a JSON snapshot (an exported file)."), 400
    try:
        counts = import_memory(data)
        with connect_db() as conn:
            total = conn.execute(
                "SELECT count(*) FROM incidents WHERE status <> 'deleted'"
            ).fetchone()[0]
    except (ValueError, SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        code = 400 if isinstance(exc, ValueError) else 500
        return jsonify(ok=False, error=str(exc)), code
    return jsonify(ok=True, total=total, **counts)


def _active_total(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM incidents WHERE status <> 'deleted'"
    ).fetchone()[0]


@app.post("/api/incidents/<int:incident_id>/delete")
def api_delete(incident_id):
    """Soft delete (undo-able): drop the Qdrant point, mark the row deleted."""
    try:
        # 1. Remove from Qdrant first so it stops matching searches immediately.
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=[incident_id])
        with connect_db() as conn:
            row = conn.execute(
                "UPDATE incidents SET status = 'deleted' WHERE id = %s RETURNING title",
                (incident_id,),
            ).fetchone()
            if row is None:
                return jsonify(ok=False, error=f"No incident #{incident_id}."), 404
            conn.commit()
            total = _active_total(conn)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, id=incident_id, title=row[0], total=total, action="deleted")


@app.post("/webhook/incident")
def api_webhook_incident():
    """Second entry point into retrieve_and_suggest(): receive an incident as
    a plain HTTP POST (a script, curl, a monitoring tool, the demo simulator),
    validate it with the IncidentIn schema, and return the matched past
    incidents + suggested fix.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(
            ok=False,
            error="Body must be a JSON object with at least 'description'.",
        ), 400

    try:
        incident = IncidentIn(**data)
    except ValidationError as exc:
        # Malformed payloads are rejected before any pipeline code runs.
        return jsonify(
            ok=False,
            error="Invalid incident payload.",
            details=exc.errors(),
        ), 422

    try:
        return jsonify(receive_incident(incident))
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 503


@app.post("/api/incidents/<int:incident_id>/restore")
def api_restore(incident_id):
    """Undo a delete: re-embed the row's text and re-upsert the Qdrant point."""
    try:
        with connect_db() as conn:
            row = conn.execute(
                "SELECT title, description, resolution, service, severity "
                "FROM incidents WHERE id = %s",
                (incident_id,),
            ).fetchone()
        if row is None:
            return jsonify(ok=False, error=f"No incident #{incident_id}."), 404
        title, description, resolution, service, severity = row
        vector = embed_text(f"{title}. {description} {resolution}")
        # Mark the row resolved FIRST, then re-create the point. If the upsert
        # fails the row simply has no point (it won't match searches) and a
        # re-run heals it — the opposite order could make a *deleted* row start
        # matching searches, which is the worse desync direction. (Postgres and
        # Qdrant can't share one transaction, so the ordering chooses the
        # fail-safe failure mode.)
        with connect_db() as conn:
            conn.execute(
                "UPDATE incidents SET status = 'resolved' WHERE id = %s", (incident_id,)
            )
            conn.commit()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=incident_id,
                    vector=vector,
                    payload={"service": service, "severity": severity, "title": title},
                )
            ],
        )
        with connect_db() as conn:
            total = _active_total(conn)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, id=incident_id, title=title, total=total, action="restored")


@app.post("/api/incidents/<int:incident_id>/purge")
def api_purge(incident_id):
    """Permanent delete from both stores (no undo)."""
    try:
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=[incident_id])
        with connect_db() as conn:
            row = conn.execute(
                "DELETE FROM incidents WHERE id = %s RETURNING title", (incident_id,)
            ).fetchone()
            if row is None:
                return jsonify(ok=False, error=f"No incident #{incident_id}."), 404
            conn.commit()
            total = _active_total(conn)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, id=incident_id, title=row[0], total=total, action="purged")


@app.post("/api/incidents/<int:incident_id>/postmortem")
def api_postmortem(incident_id):
    """Generate (or regenerate) the postmortem doc for an existing incident
    row and save it into postmortem_doc, returning the Markdown so the UI can
    show it. Works for any row — seeded or UI-resolved — and doubles as the
    UI's "Generate" action for incidents that don't have a doc yet.
    """
    try:
        with connect_db() as conn:
            row = conn.execute(
                "SELECT id, title, description, root_cause, resolution, service, "
                "severity FROM incidents WHERE id = %s",
                (incident_id,),
            ).fetchone()
        if row is None:
            return jsonify(ok=False, error=f"No incident #{incident_id}."), 404
        incident = {
            "id": row[0], "title": row[1], "description": row[2],
            "root_cause": row[3], "resolution": row[4],
            "service": row[5], "severity": row[6],
        }
        doc = generate_postmortem(incident)
        with connect_db() as conn:
            save_postmortem(conn, incident_id, doc)
    except (ValueError, SystemExit, Exception) as exc:  # noqa: BLE001 - always answer with JSON
        return jsonify(ok=False, error=str(exc)), 503
    return jsonify(ok=True, incident_id=incident_id, postmortem=doc, characters=len(doc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Incident Memory demo page.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # The schema migration ran at import time above (works for every launch
    # path); this is only a friendly startup confirmation.
    try:
        with connect_db() as conn:
            conn.execute("SELECT 1")
    except SystemExit as exc:
        print(f"WARNING: could not connect to Postgres at startup: {exc}")
        print("The page will still start; /api/incidents and /api/resolve will report errors.")
    else:
        print("Schema ready (root_cause nullable, postmortem_doc column present).")

    print(f"Incident Memory running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    sys.exit(main())
