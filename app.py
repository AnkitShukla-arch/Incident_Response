"""
Incident Memory — the demo page that wraps the whole pipeline.

The page puts the complete "memory compounds" loop behind one UI:

  ┌──────────────┐     1. embed the new incident      (new_ingest.embed_text)
  │ New incident │     2. retrieve similar past hits  (query_incidents)
  └──────────────┘     3. LLM suggests a fix          (suggest_fix)
        │              4. edit / approve the fix
        ▼              5. write it back to memory     (resolve_incident)
  ┌──────────────┐
  │ Mark resolved │──▶ Postgres row + Qdrant vector → the memory grew by one
  └──────────────┘

The incidents table at the bottom shows every row in Postgres, newest first,
so the growth is visible live (e.g. 12 → 13 → 14 …).

Run (from this folder, with the hackenv venv active):
    python app.py                  # http://localhost:5000
    python app.py --port 5001      # custom port

Endpoints:
    GET  /                            the page
    GET  /api/incidents               all incidents (newest first) + active/deleted counts
    POST /api/search                  {description, service?, severity?, top?, threshold?}
                                      -> {matches, suggestion, llm_error?}
    POST /api/resolve                 {description, suggestion, service?, severity?}
                                      -> stores the fix, {incident, total}
    GET  /api/export                  download the full memory snapshot (rows + vectors)
    POST /api/import                  restore a snapshot JSON body -> {rows_created, ...}
    POST /api/incidents/<id>/delete   soft delete (undo-able): drop the Qdrant point, mark row
    POST /api/incidents/<id>/restore  undo a delete: re-embed + re-upsert the point
    POST /api/incidents/<id>/purge    permanent delete from both stores
"""

import argparse
import json
import os
import sys

from flask import Flask, Response, jsonify, render_template, request
from qdrant_client.models import PointStruct

from memory_backup import export_memory, import_memory
from new_ingest import COLLECTION_NAME, connect_db, embed_text, qdrant_client
from query_incidents import retrieve_similar_incidents, validate_collection
from resolve_incident import ensure_schema, store_resolved_incident
from suggest_fix import (
    GEN_MODEL_DEFAULT,
    OPENAI_MODEL_DEFAULT,
    build_user_prompt,
    generate_fix,
    resolve_gen_provider,
)

app = Flask(__name__)

SERVICES = [
    "payments", "checkout", "notifications", "reporting", "web-frontend",
    "auth", "api-gateway", "search", "recommendations", "database", "other",
]
SEVERITIES = ["critical", "high", "medium", "low"]



# --- Routes ----------------------------------------------------------------


@app.get("/")
def index():
    return render_template("index.html", services=SERVICES, severities=SEVERITIES)


@app.get("/api/incidents")
def api_incidents():
    try:
        with connect_db() as conn:
            rows = conn.execute(
                "SELECT id, title, description, root_cause, resolution, service, "
                "severity, status, created_at FROM incidents ORDER BY id DESC"
            ).fetchall()
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 503
    incidents = [
        {
            "id": r[0], "title": r[1], "description": r[2], "root_cause": r[3],
            "resolution": r[4], "service": r[5], "severity": r[6],
            "status": r[7], "created_at": str(r[8]),
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
        validate_collection()
        # 1 + 2: embed the query and retrieve similar past incidents from Qdrant,
        # enriched with their full records from Postgres.
        results = retrieve_similar_incidents(description, top, threshold)
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 400
    past = [r for r in results if r["title"] is not None]

    if not past:
        return jsonify(ok=True, matches=[], suggestion=None,
                       note="No similar past incidents found above the threshold.")

    # 3: build the prompt and ask the LLM for a suggested fix.
    prompt = build_user_prompt(description, past)
    suggestion, llm_error = None, None
    try:
        provider = resolve_gen_provider()
        model = GEN_MODEL_DEFAULT if provider == "gemini" else OPENAI_MODEL_DEFAULT
        suggestion = generate_fix(prompt, provider, model)
    except SystemExit as exc:
        llm_error = str(exc)

    return jsonify(ok=True, matches=past, suggestion=suggestion, llm_error=llm_error)


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

    return jsonify(ok=True, incident=incident, total=total)


@app.get("/api/export")
def api_export():
    """Download a full snapshot (rows + vectors + embedding metadata)."""
    try:
        snapshot = export_memory()
    except SystemExit as exc:
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
    except (ValueError, SystemExit) as exc:
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
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, id=incident_id, title=row[0], total=total, action="deleted")


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
    except SystemExit as exc:
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
    except SystemExit as exc:
        return jsonify(ok=False, error=str(exc)), 500
    return jsonify(ok=True, id=incident_id, title=row[0], total=total, action="purged")


def main() -> int:
    parser = argparse.ArgumentParser(description="Incident Memory demo page.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # Keep the schema ready for UI-resolved rows (idempotent).
    try:
        with connect_db() as conn:
            ensure_schema(conn)
    except SystemExit as exc:
        print(f"WARNING: could not connect to Postgres at startup: {exc}")
        print("The page will still start; /api/incidents and /api/resolve will report errors.")
    else:
        print("Schema ready (root_cause nullable for UI-resolved incidents).")

    print(f"Incident Memory running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    sys.exit(main())
