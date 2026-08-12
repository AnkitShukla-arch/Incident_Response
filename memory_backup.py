"""
Snapshot and restore the incident memory (Postgres rows + Qdrant vectors).

Export  writes every Postgres row together with its Qdrant vector and the
        embedding metadata into one JSON file — a full backup of the memory,
        not just the records. Deleted (soft-deleted) rows are included so the
        snapshot is a faithful copy of the whole store.

Import  restores a snapshot: upserts rows by their original id, rebuilds the
        Qdrant points (using the stored vectors when they still match the
        current embedding model, otherwise re-embedding with the current
        model), and re-syncs the Postgres id sequence so the next SERIAL
        insert never collides with an imported id.

        Import is an UPSERT/MERGE, never a wipe: rows and points that already
        exist in the stores but are not in the snapshot are left untouched. To
        restore a snapshot onto a clean state, first remove the rows you want
        gone (e.g. the UI delete buttons, or reset to seeds in
        e2e_walkthrough.py --reset), then import.

CLI:
    python memory_backup.py export incidents_backup.json
    python memory_backup.py import incidents_backup.json
    python memory_backup.py check
"""

import argparse
import datetime
import json
import sys

from qdrant_client.models import PointStruct

from new_ingest import (
    COLLECTION_NAME,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_PROVIDER,
    connect_db,
    embed_text,
    qdrant_client,
    setup_qdrant,
)


def export_memory() -> dict:
    """Read the full memory state: every Postgres row + its Qdrant vector."""
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT id, title, description, root_cause, resolution, service, "
            "severity, status, created_at FROM incidents ORDER BY id"
        ).fetchall()

    # Pull the stored vectors straight out of Qdrant (exact, no re-embedding).
    vectors = {}
    ids = [r[0] for r in rows]
    if ids:
        try:
            records = qdrant_client.retrieve(
                collection_name=COLLECTION_NAME, ids=ids, with_vectors=True
            )
            for rec in records:
                vectors[rec.id] = list(rec.vector) if rec.vector is not None else None
        except Exception as exc:  # noqa: BLE001 - degrade to records-only
            print(f"Warning: could not read vectors from Qdrant ({exc}); "
                  "import will re-embed the records instead.")

    incidents = []
    for r in rows:
        inc = {
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "root_cause": r[3],
            "resolution": r[4],
            "service": r[5],
            "severity": r[6],
            "status": r[7],
            "created_at": str(r[8]),
        }
        if r[0] in vectors:  # deleted rows have no point/vector — leave it out
            inc["vector"] = vectors[r[0]]
        incidents.append(inc)

    return {
        "version": 1,
        "kind": "incident-memory",
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedding": {
            "provider": EMBED_PROVIDER,
            "model": EMBED_MODEL,
            "dims": EMBED_DIM,
        },
        "incidents": incidents,
    }


def import_memory(data: dict) -> dict:
    """Restore a snapshot. Returns counts:
    {rows_created, rows_updated, points_upserted, embedded}.
    """
    if not isinstance(data, dict) or data.get("kind") != "incident-memory":
        raise ValueError("Not a valid incident-memory snapshot (missing 'kind').")
    incidents = data.get("incidents")
    if not isinstance(incidents, list):
        raise ValueError("Snapshot field 'incidents' must be a list.")

    if data.get("embedding", {}).get("dims") != EMBED_DIM:
        print(
            f"Note: snapshot was exported with dims "
            f"{data.get('embedding', {}).get('dims')} but the current model "
            f"produces {EMBED_DIM}-dim vectors — stored vectors will be "
            "replaced by fresh embeddings where they don't match."
        )

    # Make sure the Qdrant collection exists (no-op when it already does).
    setup_qdrant()

    points = []
    rows_created = rows_updated = embedded = 0
    with connect_db() as conn:
        for inc in incidents:
            if not isinstance(inc, dict):
                continue
            try:
                inc_id = int(inc["id"])
            except (KeyError, TypeError, ValueError):
                continue
            title = (inc.get("title") or "").strip()
            description = (inc.get("description") or "").strip()
            resolution = (inc.get("resolution") or "").strip()
            if not title or not description or not resolution:
                continue
            status = (inc.get("status") or "resolved").strip() or "resolved"
            deleted = status == "deleted"
            root_cause = inc.get("root_cause")
            service = (inc.get("service") or "").strip() or None
            severity = (inc.get("severity") or "").strip() or None
            created_at = inc.get("created_at")

            # Vector: use the stored one when it matches the current model,
            # otherwise re-embed (the model may have changed since export).
            vector = inc.get("vector")
            if not deleted and (not vector or len(vector) != EMBED_DIM):
                vector = embed_text(f"{title}. {description} {resolution}")
                embedded += 1

            row = conn.execute(
                """
                INSERT INTO incidents
                    (id, title, description, root_cause, resolution, service,
                     severity, status, created_at)
                VALUES
                    (%(id)s, %(title)s, %(description)s, %(root_cause)s,
                     %(resolution)s, %(service)s, %(severity)s, %(status)s,
                     %(created_at)s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    root_cause = EXCLUDED.root_cause,
                    resolution = EXCLUDED.resolution,
                    service = EXCLUDED.service,
                    severity = EXCLUDED.severity,
                    status = EXCLUDED.status
                RETURNING (xmax = 0) AS inserted
                """,
                {
                    "id": inc_id,
                    "title": title,
                    "description": description,
                    "root_cause": root_cause,
                    "resolution": resolution,
                    "service": service,
                    "severity": severity,
                    "status": status,
                    "created_at": created_at,
                },
            ).fetchone()
            if row[0]:
                rows_created += 1
            else:
                rows_updated += 1

            # Deleted rows are restored as deleted: row only, no Qdrant point.
            if not deleted:
                points.append(
                    PointStruct(
                        id=inc_id,
                        vector=vector,
                        payload={
                            "service": service,
                            "severity": severity,
                            "title": title,
                        },
                    )
                )

        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        # Re-sync the id sequence: imported rows carry their original ids, so
        # the next SERIAL insert must start above the highest imported id.
        # is_called=false makes the next nextval return exactly MAX(id) + 1,
        # which is also correct when the table is empty (next id = 1).
        conn.execute(
            "SELECT setval(pg_get_serial_sequence('incidents', 'id'), "
            "COALESCE((SELECT MAX(id) FROM incidents), 0) + 1, false)"
        )
        conn.commit()

    return {
        "rows_created": rows_created,
        "rows_updated": rows_updated,
        "points_upserted": len(points),
        "embedded": embedded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup/restore the incident memory.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_exp = sub.add_parser("export", help="write a full snapshot JSON file")
    p_exp.add_argument("path", help="output file, e.g. incidents_backup.json")
    p_imp = sub.add_parser("import", help="restore a snapshot JSON file")
    p_imp.add_argument("path", help="the snapshot file to restore")
    sub.add_parser("check", help="show Postgres + Qdrant counts")
    args = parser.parse_args()

    if args.cmd == "check":
        with connect_db() as conn:
            total = conn.execute("SELECT count(*) FROM incidents").fetchone()[0]
            active = conn.execute(
                "SELECT count(*) FROM incidents WHERE status <> 'deleted'"
            ).fetchone()[0]
        points = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(f"Postgres: {total} rows ({active} active) | Qdrant: {points} vectors")
        return 0

    if args.cmd == "export":
        snapshot = export_memory()
        with open(args.path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        print(
            f"Exported {len(snapshot['incidents'])} incident(s) "
            f"({snapshot['embedding']['model']}) to {args.path}"
        )
        return 0

    # import
    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)
    counts = import_memory(data)
    print(
        f"Imported: {counts['rows_created']} created, {counts['rows_updated']} updated, "
        f"{counts['points_upserted']} vectors upserted "
        f"({counts['embedded']} freshly embedded)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
