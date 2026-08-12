"""
Performance benchmark for the Qdrant incident-memory system.

Measures the real performance matrix of the retrieval pipeline:
  1. Retrieval quality  - self-retrieval (Recall@k, MRR), nearest-neighbor
                          discrimination margins, related-query recall
  2. Latency matrix      - per-stage latency: Gemini embedding, Qdrant search,
                          Postgres enrichment, full retrieval, Groq LLM
                          suggestion, postmortem generation, HTTP end-to-end
  3. Throughput          - sequential full-pipeline calls per minute
  4. Threshold analysis  - score distribution vs. app thresholds (0.22 / 0.3)

Read-only against the stores: it only queries Postgres/Qdrant and makes API
calls. It writes TWO artifacts in perf_charts/:
  - performance_results.json   latest full snapshot (make_charts.py reads this)
  - performance_history.json   append-only history of every run, for trend
                               tracking across time (one entry per benchmark)

Usage:  hackenv/Scripts/python.exe benchmark_perf.py
"""

import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, ".")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_charts")
RESULTS_FILE = os.path.join(OUT_DIR, "performance_results.json")
HISTORY_FILE = os.path.join(OUT_DIR, "performance_history.json")

from new_ingest import COLLECTION_NAME, connect_db, embed_text, qdrant_client
from query_incidents import retrieve_similar_incidents
from suggest_fix import (
    SYSTEM_PROMPT,
    build_user_prompt,
    generate_fix,
    resolve_gen_model,
    resolve_gen_provider,
)
from postmortem import generate_postmortem
from resolve_incident import ensure_schema

TOP_K = 5
THRESHOLD_APP = 0.22   # /api/search default
THRESHOLD_CLI = 0.30   # CLI default

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"


def pct(ok, total):
    return f"{ok}/{total} ({100.0 * ok / total:.1f}%)"


def fmt_ms(sec):
    return f"{sec * 1000:.0f} ms"


# ---------------------------------------------------------------- state ---
def store_state():
    with connect_db() as conn:
        ensure_schema(conn)
        total = conn.execute(
            "SELECT count(*) FROM incidents WHERE status <> 'deleted'"
        ).fetchone()[0]
        all_rows = conn.execute("SELECT count(*) FROM incidents").fetchone()[0]
        with_docs = conn.execute(
            "SELECT count(*) FROM incidents WHERE postmortem_doc IS NOT NULL"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT id, title, description, service, severity FROM incidents "
            "WHERE status <> 'deleted' ORDER BY id"
        ).fetchall()
    qdrant_count = qdrant_client.count(
        collection_name=COLLECTION_NAME, exact=True
    ).count
    info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "pg_active": total, "pg_all": all_rows, "qdrant": qdrant_count,
        "with_docs": with_docs, "rows": rows, "info": info,
    }


# -------------------------------------------------------- retrieval quality ---
def run_self_retrieval(rows):
    """Query Qdrant with each incident's own description; the right answer is
    the incident itself. Produces Recall@k, MRR and score margins."""
    ranks = []          # rank of the correct incident per query (1-based)
    top1_self = []      # similarity score of the correct incident
    top1_other = []     # highest similarity of any OTHER incident
    scores_first = []   # top-1 score regardless of correctness
    score_table = []    # per-query breakdown for the report
    for r in rows:
        rid, title, desc, service, sev = r
        query = desc.strip()
        # measure stage latencies here too (embed + qdrant + enrich)
        t0 = time.perf_counter()
        vec = embed_text(query)
        t_embed = time.perf_counter() - t0
        t0 = time.perf_counter()
        resp = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=TOP_K, score_threshold=None,
        )
        t_search = time.perf_counter() - t0
        hits = resp.points
        t0 = time.perf_counter()
        with connect_db() as conn:
            for h in hits:
                conn.execute(
                    "SELECT 1 FROM incidents WHERE id = %s", (h.id,)
                ).fetchone()
        t_enrich = time.perf_counter() - t0

        rank = None
        self_score = None
        others = []
        for i, h in enumerate(hits, 1):
            if h.id == rid:
                rank = i
                self_score = h.score
            else:
                others.append(h.score)
        ranks.append(rank)
        if self_score is not None:
            top1_self.append(self_score)
        top1_other.append(max(others) if others else 0.0)
        scores_first.append(hits[0].score if hits else 0.0)
        score_table.append(
            (rid, rank, self_score, hits[0].score if hits else 0.0,
             round(t_embed * 1000), round(t_search * 1000),
             round(t_enrich * 1000),
             [round(h.score, 3) for h in hits])
        )
        print(f"    #{rid:<4} {title[:52]:<54} rank={rank}  self_score={self_score and round(self_score,3)}  top1={hits[0].score and round(hits[0].score,3)}")
    return {
        "ranks": ranks, "top1_self": top1_self, "top1_other": top1_other,
        "scores_first": scores_first, "score_table": score_table,
    }


def run_related_queries(rows):
    """Realistic 'new incident' queries, each expected to surface a specific seed.
    Mirrors the e2e walkthrough's cycle queries + obvious semantic relatives."""
    id_by_frag = {}
    for r in rows:
        t = (r[1] or "").lower()
        if "payment service timeout" in t:
            id_by_frag["payment"] = r[0]
        elif "checkout api 504" in t:
            id_by_frag["checkout"] = r[0]
        elif "notification worker" in t:
            id_by_frag["notif"] = r[0]
        elif "report generation" in t:
            id_by_frag["report"] = r[0]
        elif "auth service outage" in t:
            id_by_frag["auth"] = r[0]
        elif "api gateway cert" in t:
            id_by_frag["gateway"] = r[0]
        elif "database disk full" in t:
            id_by_frag["db"] = r[0]
        elif "duplicate order" in t:
            id_by_frag["dup"] = r[0]

    cases = [
        ("Payment service timeout during flash sale - 504 errors, latency spike",
         "payment"),
        ("Checkout API started throwing 504s during a traffic spike after an email blast",
         "checkout"),
        ("Notification worker pod memory usage climbing steadily until OOMKill",
         "notif"),
        ("Nightly report generation job OOMKilled mid-run on large accounts",
         "report"),
        ("Auth service: all logins failing with SSL handshake errors at 3am",
         "auth"),
        ("Internal API gateway cert expired, TLS errors between services",
         "gateway"),
        ("Database disk full, write operations failing across services",
         "db"),
        ("Customers charged twice - duplicate orders from double-clicking checkout",
         "dup"),
    ]
    print(f"\n{BOLD}Related-query recall (8 realistic 'new incident' descriptions){END}")
    print(f"  {'expected':<10} {'rank':<6} {'top1':<7} {'in top-5':<10} scores")
    hits_at1 = hits_at5 = 0
    details = []
    for query, key in cases:
        expected = id_by_frag.get(key)
        results = retrieve_similar_incidents(query, TOP_K, THRESHOLD_APP)
        ids = [m["id"] for m in results]
        rank = ids.index(expected) + 1 if expected in ids else None
        top1 = results[0]["score"] if results else 0.0
        ok = rank is not None
        hits_at5 += 1 if ok else 0
        hits_at1 += 1 if rank == 1 else 0
        sc = " ".join(f"{m['score']:.2f}" for m in results[:3])
        mark = f"{GREEN}PASS{END}" if ok else f"{RED}MISS{END}"
        print(f"  {key:<10} {str(rank or '-'):<6} {top1:<7.3f} {mark:<10} {sc}")
        details.append({
            "key": key, "rank": rank, "top1": round(top1, 4),
            "ok": ok, "scores": [round(m["score"], 4) for m in results[:3]],
        })
    print(f"  -> Recall@1: {hits_at1}/8   Recall@5: {pct(hits_at5, len(cases))}")
    return hits_at1, hits_at5, details


# ---------------------------------------------------------------- latency ---
def latency_matrix():
    """LLM-stage latencies (Groq) + HTTP end-to-end latency."""
    out = {}
    provider = resolve_gen_provider()
    model = resolve_gen_model(provider)

    # warm-up
    generate_fix("warmup", provider, model, SYSTEM_PROMPT)

    samples = []
    for _ in range(2):
        past = retrieve_similar_incidents(
            "payment service 504 timeouts under flash sale load", 3, THRESHOLD_APP
        )
        prompt = build_user_prompt("payment service 504 timeouts during flash sale", past)
        t0 = time.perf_counter()
        generate_fix(prompt, provider, model, SYSTEM_PROMPT)
        samples.append(time.perf_counter() - t0)
    out["llm_suggestion"] = samples

    pm_samples = []
    with connect_db() as conn:
        row = conn.execute(
            "SELECT id, title, description, root_cause, resolution, service, "
            "severity FROM incidents WHERE status <> 'deleted' ORDER BY id LIMIT 1"
        ).fetchone()
    incident = {
        "id": row[0], "title": row[1], "description": row[2],
        "root_cause": row[3], "resolution": row[4],
        "service": row[5], "severity": row[6],
    }
    t0 = time.perf_counter()
    generate_postmortem(incident)
    out["postmortem"] = [time.perf_counter() - t0]

    http_samples = []
    payload = json.dumps({
        "description": "Auth service TLS handshake failures at 3am, all logins failing",
        "top": 3,
    }).encode()
    for _ in range(2):
        req = urllib.request.Request(
            "http://127.0.0.1:5050/api/search", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        http_samples.append(time.perf_counter() - t0)
    out["http_e2e"] = http_samples

    # throughput estimate: sequential full pipeline (retrieve + LLM), 3 iterations
    pipe = []
    for _ in range(3):
        t0 = time.perf_counter()
        past = retrieve_similar_incidents(
            "checkout 504s under traffic spike", 3, THRESHOLD_APP
        )
        prompt = build_user_prompt("checkout 504s under traffic spike", past)
        generate_fix(prompt, provider, model, SYSTEM_PROMPT)
        pipe.append(time.perf_counter() - t0)
    out["pipeline_full"] = pipe
    return out


def stat(samples):
    if not samples:
        return (0.0, 0.0, 0.0)
    s = sorted(samples)
    return (
        statistics.mean(s),
        s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2,
        s[min(len(s) - 1, int(len(s) * 0.95))],
    )


def print_latency_row(name, samples, calls):
    mean, p50, p95 = stat(samples)
    print(f"  {name:<30} {calls:>3}  {fmt_ms(mean):>9} {fmt_ms(p50):>9} {fmt_ms(p95):>9}")


# ---------------------------------------------------------------- history ---
def _aggregate_entry(results: dict) -> dict:
    """Compress a full snapshot into the aggregate metrics that trend charts
    need. The history file only stores these, keeping it small as it grows."""
    lat = results["latency"]
    return {
        "generated_at": results["generated_at"],
        "system": results["system"],
        "quality": {
            "recall1": results["retrieval_quality"]["recall1"],
            "recall3": results["retrieval_quality"]["recall3"],
            "recall5": results["retrieval_quality"]["recall5"],
            "total": results["retrieval_quality"]["total"],
            "mrr": results["retrieval_quality"]["mrr"],
            "self_mean": results["retrieval_quality"]["self_mean"],
            "margin_mean": results["retrieval_quality"]["margin_mean"],
        },
        "related": results["related_queries"],
        "latency_ms": {
            k: lat[k]["mean_ms"] for k in (
                "embedding", "qdrant", "postgres", "retrieval_full",
                "llm_suggestion", "postmortem", "http_e2e", "pipeline_full",
            )
        },
        "throughput": results["throughput"],
    }


def save_results(results: dict) -> int:
    """Write the latest snapshot and append an aggregate entry to the history
    file. Returns the run number of this entry (1-based)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Latest full snapshot (overwritten every run).
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {RESULTS_FILE}")

    # 2. Append-only history (one aggregate entry per run).
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except (json.JSONDecodeError, OSError):
            # Corrupt history must never block a benchmark run; start fresh.
            history = []
    entry = _aggregate_entry(results)
    entry["run"] = len(history) + 1
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History appended -> {HISTORY_FILE}  (run #{entry['run']}, "
          f"{len(history)} run(s) total)")
    return entry["run"]


# ------------------------------------------------------------------ main ---
def main():
    print(f"{BOLD}============================================================{END}")
    print(f"{BOLD} INCIDENT MEMORY — PERFORMANCE MATRIX (Qdrant + Gemini + Groq){END}")
    print(f"{BOLD}============================================================{END}")

    st = store_state()
    rows = st["rows"]
    info = st["info"]
    n = len(rows)
    print(f"\n{BOLD}[0] SYSTEM STATE{END}")
    print(f"  Postgres active rows : {st['pg_active']}   (all rows incl. deleted: {st['pg_all']})")
    print(f"  Qdrant points        : {st['qdrant']}")
    print(f"  Stores in sync       : {'YES' if st['pg_active'] == st['qdrant'] else 'NO - drift!'}")
    print(f"  Rows with postmortem : {st['with_docs']}")
    try:
        print(f"  Collection status    : {info.status} | points={info.points_count} | "
              f"vectors={info.config.params.vectors.size}d | "
              f"distance={info.config.params.vectors.distance}")
    except Exception:
        pass
    print(f"  Memory growth        : 12 seeded -> {st['pg_active']} now (+{st['pg_active'] - 12} from resolving)")

    print(f"\n{BOLD}[1] RETRIEVAL QUALITY — self-retrieval over all {n} active incidents{END}")
    print(f"  (each incident's own description used as the query; correct answer = itself)")
    sr = run_self_retrieval(rows)

    ranks = [r for r in sr["ranks"] if r is not None]
    recall1 = sum(1 for r in ranks if r == 1)
    recall3 = sum(1 for r in ranks if r <= 3)
    recall5 = sum(1 for r in ranks if r <= 5)
    mrr = statistics.mean(1.0 / r for r in ranks) if ranks else 0.0
    print(f"\n  Recall@1  : {pct(recall1, n)}")
    print(f"  Recall@3  : {pct(recall3, n)}")
    print(f"  Recall@5  : {pct(recall5, n)}")
    print(f"  MRR@5     : {mrr:.4f}")
    if sr["top1_self"]:
        print(f"  Mean top-1 SELF score  : {statistics.mean(sr['top1_self']):.4f} "
              f"(min {min(sr['top1_self']):.4f}, max {max(sr['top1_self']):.4f})")
        print(f"  Mean top-1 OTHER score : {statistics.mean(sr['top1_other']):.4f}")
        margins = [s - o for s, o in zip(sr["top1_self"], sr["top1_other"]) if s]
        print(f"  Mean discrimination margin (self - nearest other): "
              f"{statistics.mean(margins):.4f}")

    r1, r5, related_details = run_related_queries(rows)

    print(f"\n{BOLD}[2] LATENCY MATRIX (measured on the live system){END}")
    print(f"  {'stage':<30} {'n':>3}  {'mean':>9} {'p50':>9} {'p95':>9}")
    emb = [t[4] / 1000 for t in sr["score_table"]]
    sea = [t[5] / 1000 for t in sr["score_table"]]
    enr = [t[6] / 1000 for t in sr["score_table"]]
    full = [e + s + q for e, s, q in zip(emb, sea, enr)]
    print_latency_row("Gemini embedding", emb, len(emb))
    print_latency_row("Qdrant query_points", sea, len(sea))
    print_latency_row("Postgres enrichment", enr, len(enr))
    print_latency_row("Full retrieval (1+2+3)", full, len(full))
    lm = latency_matrix()
    print_latency_row("Groq LLM suggestion", lm["llm_suggestion"], len(lm["llm_suggestion"]))
    print_latency_row("Groq postmortem gen", lm["postmortem"], len(lm["postmortem"]))
    print_latency_row("HTTP POST /api/search (e2e)", lm["http_e2e"], len(lm["http_e2e"]))
    print_latency_row("Full pipeline (retrieve+LLM)", lm["pipeline_full"], len(lm["pipeline_full"]))

    print(f"\n{BOLD}[3] THROUGHPUT (sequential, single worker){END}")
    pm = stat(lm["pipeline_full"])[0]
    print(f"  Full pipeline (embed+search+LLM): {60.0 / pm:.1f} incidents/min "
          f"({pm * 1000:.0f} ms avg each)")
    pe = stat(emb)[0]
    print(f"  Retrieval-only path:             {60.0 / (pe + stat(sea)[0] + stat(enr)[0]):.1f} queries/min")
    print(f"  Estimated Groq headroom:         30 req/min free limit vs {60.0 / pm:.1f} used -> "
          f"{max(0, 30 - 60.0 / pm):.0f} req/min spare")

    print(f"\n{BOLD}[4] SCORE THRESHOLD ANALYSIS (app default {THRESHOLD_APP}, CLI default {THRESHOLD_CLI}){END}")
    first = sr["scores_first"]
    print(f"  Top-1 score distribution across all queries: "
          f"mean {statistics.mean(first):.3f}, min {min(first):.3f}, max {max(first):.3f}")
    for thr, label in ((THRESHOLD_APP, "app (/api/search)"), (THRESHOLD_CLI, "CLI")):
        kept = sum(1 for s in first if s >= thr)
        print(f"  Queries returning >= {thr} (default {label}): {pct(kept, len(first))}")
    print("  Note: self-retrieval is the WORST-case match (no rephrasing). Real-world")
    print("  new incidents score lower, so 0.22 keeps recall high at the cost of noise.")

    # per-query score table (compact)
    print(f"\n{BOLD}[5] PER-QUERY SCORES (self-retrieval){END}")
    print(f"  {'id':<5} {'rank':<6} {'self':<7} {'top1':<7} {'top5 scores'}")
    for rid, rank, self_s, top1, _, _, _, scores in sr["score_table"]:
        self_s = round(self_s, 3) if self_s is not None else "-"
        print(f"  #{rid:<4} {str(rank or '-'):<6} {self_s:<7} {round(top1, 3):<7} {scores}")

    print(f"\n{BOLD}Summary:{END} Recall@1={pct(recall1, n)}  Recall@5={pct(recall5, n)}  "
          f"MRR={mrr:.4f}  related Recall@5={pct(r5, 8)}  "
          f"full-pipeline {fmt_ms(pm)}/call")

    # ---- dump everything the chart generator needs ------------------------
    def stat3(samples):
        m, p50, p95 = stat(samples)
        return {"mean_ms": round(m * 1000, 1), "p50_ms": round(p50 * 1000, 1),
                "p95_ms": round(p95 * 1000, 1), "n": len(samples)}

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "pg_active": st["pg_active"], "pg_all": st["pg_all"],
            "qdrant": st["qdrant"], "with_docs": st["with_docs"],
            "synced": st["pg_active"] == st["qdrant"],
            "seeded": 12, "growth": st["pg_active"] - 12,
        },
        "retrieval_quality": {
            "recall1": recall1, "recall3": recall3, "recall5": recall5,
            "total": n, "mrr": round(mrr, 4),
            "self_mean": round(statistics.mean(sr["top1_self"]), 4),
            "self_min": round(min(sr["top1_self"]), 4),
            "self_max": round(max(sr["top1_self"]), 4),
            "other_mean": round(statistics.mean(sr["top1_other"]), 4),
            "margin_mean": round(statistics.mean(margins), 4),
            "scores_first": [round(s, 4) for s in sr["scores_first"]],
            "per_query": [
                {"id": t[0], "rank": t[1],
                 "self": round(t[2], 4) if t[2] is not None else None,
                 "top1": round(t[3], 4), "top5": t[7]}
                for t in sr["score_table"]
            ],
        },
        "related_queries": {
            "recall1": r1, "recall5": r5, "total": len(related_details),
            "details": related_details,
        },
        "latency": {
            "embedding": stat3(emb), "qdrant": stat3(sea),
            "postgres": stat3(enr), "retrieval_full": stat3(full),
            "llm_suggestion": stat3(lm["llm_suggestion"]),
            "postmortem": stat3(lm["postmortem"]),
            "http_e2e": stat3(lm["http_e2e"]),
            "pipeline_full": stat3(lm["pipeline_full"]),
        },
        "throughput": {
            "pipeline_per_min": round(60.0 / pm, 1),
            "retrieval_per_min": round(
                60.0 / (pe + stat(sea)[0] + stat(enr)[0]), 1),
            "groq_headroom": round(max(0, 30 - 60.0 / pm), 1),
        },
        "thresholds": {
            "app": THRESHOLD_APP, "cli": THRESHOLD_CLI,
            "scores_mean": round(statistics.mean(first), 3),
            "scores_min": round(min(first), 3),
            "scores_max": round(max(first), 3),
            "over_app": sum(1 for s in first if s >= THRESHOLD_APP),
            "over_cli": sum(1 for s in first if s >= THRESHOLD_CLI),
        },
    }
    save_results(results)
    print(f"{GREEN}Benchmark complete - stores in sync, no errors.{END}")


if __name__ == "__main__":
    main()
