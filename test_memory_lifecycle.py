"""
Lifecycle test for the new memory features: export/import + delete/undo.

Runs against the Flask app's real API (test client) and verifies:

  1. export  -> snapshot has every row + 768-dim vectors + embedding metadata
  2. delete  -> soft delete: active count drops, Qdrant point gone, search stops matching
  3. restore -> undo: active count back, point back, search matches again
  4. purge   -> permanent delete from BOTH stores
  5. reset to seeds, then import -> rows restored with their ORIGINAL ids and vectors
  6. import idempotency (running it twice adds nothing)
  7. SERIAL sequence safety (next resolve gets a fresh, non-colliding id)

Run:  hackenv/Scripts/python.exe test_memory_lifecycle.py
"""

import json
import sys

sys.path.insert(0, ".")

import app as appmod
from new_ingest import COLLECTION_NAME, connect_db, qdrant_client
from query_incidents import retrieve_similar_incidents

client = appmod.app.test_client()

PASS, FAIL, BOLD, END = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[1m", "\033[0m"

AUTH_QUERY = "Auth service TLS handshake failures at 3am, all logins failing with SSL errors"


def check(label, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {label}{('  -> ' + detail) if detail and not cond else ''}")
    return cond


def get(route):
    r = client.get(route)
    return r.status_code, r.get_json()


def post(route, payload=None):
    r = client.post(route, json=payload) if payload is not None else client.post(route)
    return r.status_code, r.get_json()


def pg_row(id_):
    with connect_db() as conn:
        return conn.execute("SELECT id, status FROM incidents WHERE id = %s", (id_,)).fetchone()


def qdrant_has(id_):
    return len(qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[id_])) > 0


def search_ids(desc):
    hits = retrieve_similar_incidents(desc, 5, 0.22)
    return [h["id"] for h in hits if h["title"] is not None]


def resolve_temp(desc, suggestion):
    sc, d = post("/api/resolve", {
        "description": desc, "suggestion": suggestion,
        "service": "other", "severity": "low",
    })
    assert sc == 200, d
    return d["incident"]["id"], d["total"]


def main():
    print(f"{BOLD}0. INITIAL STATE{END}")
    sc, d = get("/api/incidents")
    print(f"  active={d['total']} deleted={d.get('deleted')} | newest id {d['incidents'][0]['id']}")
    assert d["total"] == 17 and d["deleted"] == 0, f"expected 17 active, got {d}"
    test_id = d["incidents"][0]["id"]  # #26 (the auth incident)

    print(f"\n{BOLD}1. EXPORT — full snapshot with vectors{END}")
    sc, snap = get("/api/export")
    check("GET /api/export returns 200", sc == 200)
    assert isinstance(snap, dict) and snap.get("kind") == "incident-memory", "not a snapshot"
    check("snapshot has embedding metadata (provider/model/dims)",
          snap.get("embedding", {}).get("dims") == 768 and "model" in snap["embedding"])
    incs = snap["incidents"]
    check(f"snapshot has all {len(incs)} incidents", len(incs) == 17)
    with_vec = [i for i in incs if i.get("vector")]
    check(f"all {len(with_vec)} active rows carry stored vectors", len(with_vec) == 17)
    check("vectors are 768-dim (match the embedding model)",
          len(with_vec) > 0 and len(with_vec[0]["vector"]) == 768)
    with open("incidents_backup.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    print("  -> saved to incidents_backup.json")
    ids_exported = {i["id"] for i in incs}

    print(f"\n{BOLD}2. SOFT DELETE #{test_id} (undo-able){END}")
    sc, d = post(f"/api/incidents/{test_id}/delete")
    check("delete returns ok + active total 16", d.get("ok") and d["total"] == 16)
    check("Postgres row marked 'deleted'", pg_row(test_id) and pg_row(test_id)[1] == "deleted")
    check("Qdrant point removed", not qdrant_has(test_id))
    check("search no longer matches the deleted incident", test_id not in search_ids(AUTH_QUERY))
    sc, d2 = get("/api/incidents")
    check("API reports 1 deleted", d2.get("deleted") == 1)

    print(f"\n{BOLD}3. RESTORE #{test_id} (undo){END}")
    sc, d = post(f"/api/incidents/{test_id}/restore")
    check("restore returns ok + active total 17", d.get("ok") and d["total"] == 17)
    check("Postgres row back to 'resolved'", pg_row(test_id) and pg_row(test_id)[1] == "resolved")
    check("Qdrant point re-created", qdrant_has(test_id))
    check("search matches the restored incident again", test_id in search_ids(AUTH_QUERY))

    print(f"\n{BOLD}4. PURGE — permanent delete{END}")
    tmp_id, total = resolve_temp(
        "Temp incident for purge test, not a real one",
        "1. root cause: test. 2. fix: delete me after the test.",
    )
    print(f"  -> resolved temp incident #{tmp_id} (total {total})")
    sc, d = post(f"/api/incidents/{tmp_id}/purge")
    check("purge returns ok + total back to 17", d.get("ok") and d["total"] == 17)
    check("purged row gone from Postgres", pg_row(tmp_id) is None)
    check("purged point gone from Qdrant", not qdrant_has(tmp_id))

    print(f"\n{BOLD}5. RESET TO SEEDS, THEN IMPORT SNAPSHOT{END}")
    with connect_db() as conn:
        rows = conn.execute("SELECT id FROM incidents WHERE title LIKE 'Resolved:%'").fetchall()
        ids = [r[0] for r in rows]
        conn.execute("DELETE FROM incidents WHERE id = ANY(%s)", (ids,))
        conn.commit()
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=ids)
        print(f"  -> removed {len(ids)} UI-resolved row(s), back to seeds")
    with open("incidents_backup.json", encoding="utf-8") as f:
        payload = f.read()
    sc, d = post("/api/import", json.loads(payload))
    check("import returns ok, 5 created, total 17",
          d.get("ok") and d.get("rows_created") == 5 and d["total"] == 17)
    sc, d2 = get("/api/incidents")
    restored_ids = {i["id"] for i in d2["incidents"]}
    check("original ids preserved (snapshot ids == current ids)", restored_ids == ids_exported)
    check("deleted count 0 after import", d2.get("deleted") == 0)
    check("imported vector works: auth search matches #" + str(test_id),
          test_id in search_ids(AUTH_QUERY))

    print(f"\n{BOLD}6. IMPORT IDEMPOTENCY — running it again adds nothing{END}")
    sc, d = post("/api/import", json.loads(payload))
    check("second import creates 0 rows (updates 17)",
          d.get("ok") and d.get("rows_created") == 0 and d.get("rows_updated") == 17)

    print(f"\n{BOLD}7. SERIAL SEQUENCE SAFETY{END}")
    tmp_id, total = resolve_temp(
        "Temp incident for sequence test",
        "1. root cause: test. 2. fix: delete me.",
    )
    check(f"next resolve got a fresh non-colliding id #{tmp_id} (total {total})",
          tmp_id == max(ids) + 1 and total == 18)
    sc, d = post(f"/api/incidents/{tmp_id}/purge")

    print(f"\n{BOLD}FINAL STATE{END}")
    sc, d = get("/api/incidents")
    qn = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"  Postgres active={d['total']} deleted={d['deleted']} | Qdrant={qn}")
    check("Postgres active == Qdrant points (stores in sync)", d["total"] == qn == 17)
    check("newest id restored correctly", d["incidents"][0]["id"] == test_id)
    print(f"\n{BOLD}ALL LIFECYCLE CHECKS COMPLETE{END}")


if __name__ == "__main__":
    main()
