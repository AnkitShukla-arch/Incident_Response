"""
End-to-end walkthrough: exercises the full memory-compounds loop through the
Flask app's real API endpoints (via Flask's test client — same code paths the
browser uses).

  cycle A: payments/flash-sale incident  -> resolve  -> related checkout incident
  cycle B: notification worker OOMKill   -> resolve  -> related reporting OOMKill

Each cycle proves that resolving an incident makes it retrievable and citable
by the very next related query.

Usage:
  python e2e_walkthrough.py            # requires the memory to be at the 12-seed state
  python e2e_walkthrough.py --reset    # first remove UI-resolved rows, then run
"""

import sys
import time

if "--reset" in sys.argv:
    from new_ingest import COLLECTION_NAME, connect_db, qdrant_client

    with connect_db() as conn:
        rows = conn.execute(
            "SELECT id FROM incidents WHERE title LIKE 'Resolved:%'"
        ).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            conn.execute("DELETE FROM incidents WHERE id = ANY(%s)", (ids,))
            conn.commit()
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=ids)
            print(f"--reset: removed {len(ids)} UI-resolved row(s) {ids}")
        else:
            print("--reset: nothing to remove")

sys.path.insert(0, ".")

import app as appmod
from new_ingest import COLLECTION_NAME, connect_db, qdrant_client

client = appmod.app.test_client()

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
BOLD = "\033[1m"
END = "\033[0m"


def get(route):
    r = client.get(route)
    return r.status_code, r.get_json()


def post(route, payload):
    r = client.post(route, json=payload)
    return r.status_code, r.get_json()


def pg_count():
    with connect_db() as conn:
        return conn.execute("SELECT count(*) FROM incidents").fetchone()[0]


def qdrant_count():
    return qdrant_client.count(collection_name=COLLECTION_NAME, exact=True).count


def check(label, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {label}{('  ' + detail) if detail and not cond else ''}")
    return cond


def show_suggestion(d):
    if d.get("suggestion"):
        print("\n  -------- LLM SUGGESTION --------")
        print(d["suggestion"])
        print("  --------------------------------")
    elif d.get("llm_error"):
        print(f"\n  !! LLM error: {d['llm_error'][:300]}")
    else:
        print("\n  !! no suggestion returned")


def show_matches(d, label):
    print(f"\n  Matches for {label}:")
    for m in d.get("matches", []):
        print(f"    #{m['id']:<4} {m['score']:.3f}  {m['title']}  [{m['service']}/{m['severity']}]")


def run_cycle(name, query1, query2, seed_title_frag, check_frag):
    """Run one full cycle: search -> resolve -> related search -> resolve."""
    print(f"\n{BOLD}=== CYCLE {name} — step 1: search for a NEW incident similar to a seeded one ==={END}")
    t0 = time.time()
    sc, d = post("/api/search", {"description": query1, "top": 5, "threshold": 0.22})
    print(f"  (api latency {time.time()-t0:.1f}s)")
    assert sc == 200, f"search returned {sc}"
    show_matches(d, f"'{query1[:60]}…'")
    check(f"search 1 returns the original seed ('{seed_title_frag}…')",
          any(seed_title_frag in (m["title"] or "") for m in d["matches"]))
    show_suggestion(d)
    assert d.get("suggestion"), "no suggestion to resolve"
    suggestion1 = d["suggestion"]

    print(f"\n{BOLD}=== CYCLE {name} — step 2: engineer edits/approves the fix, marks resolved ==={END}")
    sc, d = post("/api/resolve", {
        "description": query1,
        "suggestion": suggestion1,
        "service": d["matches"][0]["service"],
        "severity": "high",
    })
    assert sc == 200, f"resolve returned {sc}: {d}"
    new_id = d["incident"]["id"]
    new_title = d["incident"]["title"]
    print(f"  -> saved #{new_id} '{new_title}' | Postgres total now {d['total']}")
    check(f"Postgres count incremented (now {d['total']})", d["total"] >= 13)
    check("Qdrant has the same point count as Postgres", qdrant_count() == pg_count())

    print(f"\n{BOLD}=== CYCLE {name} — step 3: search for a RELATED incident ==={END}")
    t0 = time.time()
    sc, d = post("/api/search", {"description": query2, "top": 5, "threshold": 0.22})
    print(f"  (api latency {time.time()-t0:.1f}s)")
    assert sc == 200
    show_matches(d, f"'{query2[:60]}…'")
    check(f"retrieval now ALSO returns the just-resolved incident #{new_id}",
          any(m["id"] == new_id for m in d["matches"]))
    check(f"retrieval still returns the original seed ('{seed_title_frag}…')",
          any(seed_title_frag in (m["title"] or "") for m in d["matches"]))
    show_suggestion(d)
    assert d.get("suggestion"), "no suggestion to resolve"
    suggestion2 = d["suggestion"]
    cites_new = new_title.split("Resolved: ")[-1][:28] in suggestion2 or f"#{new_id}" in suggestion2 or str(new_id) in suggestion2
    check(f"suggestion CITES the just-resolved incident (title fragment or id {new_id})", cites_new)
    check("suggestion cites the original seed",
          seed_title_frag[:28] in suggestion2 or str(check_frag) in suggestion2)

    sc, d = post("/api/resolve", {
        "description": query2,
        "suggestion": suggestion2,
        "service": d["matches"][0]["service"],
        "severity": "high",
    })
    assert sc == 200, f"resolve returned {sc}: {d}"
    print(f"  -> saved #{d['incident']['id']} '{d['incident']['title']}' | Postgres total now {d['total']}")
    check("Postgres count incremented", d["total"] >= 14)
    check("Qdrant in sync with Postgres", qdrant_count() == pg_count())
    return new_id, new_title


def main():
    print(f"{BOLD}INITIAL STATE{END}")
    sc, d = get("/api/incidents")
    print(f"  Postgres: {d['total']} incidents | Qdrant: {qdrant_count()} vectors")
    assert d["total"] == 12, f"expected 12 seeded incidents, found {d['total']}"

    # --- Cycle A: payments family ---
    run_cycle(
        "A",
        "Payment API started returning 504 timeouts again during a flash sale; "
        "latency on the /charge endpoint spiked from 200ms to over 12s for about 15% of requests.",
        "Checkout API throwing 504s during a traffic spike right after the flash sale "
        "where the payment service timed out — same connection pool pattern we fixed before.",
        "Payment service timeout under load",
        7,  # the checkout seed's likely id (used only as a soft reference)
    )

    # --- Cycle B: memory-leak family ---
    run_cycle(
        "B",
        "Notification worker pod memory climbing steadily again until Kubernetes "
        "OOMKills it, roughly every few hours, causing gaps in email delivery.",
        "Nightly report generation job OOMKilled partway through processing the largest "
        "accounts, similar to the notification worker memory leak — the same caching issue.",
        "Memory leak in notification worker",
        3,  # soft reference only
    )

    print(f"\n{BOLD}FINAL STATE{END}")
    sc, d = get("/api/incidents")
    print(f"  Postgres: {d['total']} incidents | Qdrant: {qdrant_count()} vectors")
    print(f"  Newest rows:")
    for inc in d["incidents"][:4]:
        print(f"    #{inc['id']:<4} {inc['title']}  [{inc['service']}/{inc['severity']}]")
    check("memory grew 12 -> 16 (2 cycles x 2 resolutions)", d["total"] == 16)

    print(f"\n{BOLD}ALL CHECKS COMPLETE{END}")


if __name__ == "__main__":
    main()
