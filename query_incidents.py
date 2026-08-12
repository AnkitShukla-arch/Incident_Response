"""
Query the incident memory store.

Embeds a description of a NEW incident with the same embedding model used by
new_ingest.py, then searches the Qdrant 'incidents' collection for the most
similar PAST incidents and prints their root causes and resolutions.

Usage:
  python query_incidents.py "payment service timing out during flash sale"
  python query_incidents.py --top 5 "checkout 504s under traffic spike"
  python query_incidents.py            # interactive prompt

Options:
  --top N            Number of similar incidents to return (default: 3)
  --threshold X      Minimum similarity score, 0..1 (default: 0.3)

Run new_ingest.py at least once first so the collection has vectors to search.

The retrieval helper `retrieve_similar_incidents()` (embed -> Qdrant search ->
Postgres fetch) is also used by suggest_fix.py, which goes one step further and
has an LLM turn the retrieved incidents into a suggested fix.
"""

import argparse
import sys

# Reuse the .env loading, provider resolution, embedding and Qdrant client
# from new_ingest.py instead of duplicating that setup here.
# Note: importing new_ingest runs its module-level setup (loads .env, creates
# the embedding + Qdrant clients) and fails fast with its error messages if
# env vars are missing.
from new_ingest import (
    COLLECTION_NAME,
    EMBED_DIM,
    _collection_vector_size,
    connect_db,
    embed_text,
    qdrant_client,
)


def retrieve_similar_incidents(query_text: str, top_k: int, threshold: float) -> list[dict]:
    """Embed the query, search Qdrant, and pull the full records from Postgres.

    Returns a ranked list of dicts (most similar first):
        {id, score, title, description, root_cause, resolution, service, severity}

    A hit whose Qdrant point has no matching Postgres row is included with the
    record fields set to None so callers can decide how to handle it. Used by
    this script's CLI mode and by suggest_fix.py's LLM pipeline.
    """
    # 1. Embed the new incident description (same model/dims as the ingest step)
    print("Embedding the query ...")
    vector = embed_text(query_text)

    # 2. Semantic search over Qdrant (cosine similarity against all past incidents)
    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            score_threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real error with context
        raise SystemExit(f"Qdrant search failed:\n  {exc}") from exc

    hits = response.points

    # 3. Enrich each hit with the full record from Postgres (same id in both stores)
    with connect_db() as conn:
        rows = {}
        for hit in hits:
            row = conn.execute(
                "SELECT id, title, description, root_cause, resolution, service, severity "
                "FROM incidents WHERE id = %s",
                (hit.id,),
            ).fetchone()
            if row:
                rows[hit.id] = row

    results = []
    for hit in hits:
        row = rows.get(hit.id)
        results.append(
            {
                "id": hit.id,
                "score": hit.score,
                "title": row[1] if row else None,
                "description": row[2] if row else None,
                "root_cause": row[3] if row else None,
                "resolution": row[4] if row else None,
                "service": row[5] if row else None,
                "severity": row[6] if row else None,
            }
        )
    return results


def search_incidents(query_text: str, top_k: int, threshold: float) -> None:
    """Retrieve similar incidents and print them (CLI mode)."""
    results = retrieve_similar_incidents(query_text, top_k, threshold)
    if not results:
        print(f"\nNo similar incidents found above the score threshold ({threshold}).")
        print("  - Lower the bar with --threshold, e.g. --threshold 0.1")
        print("  - Make sure the collection has data: python new_ingest.py")
        return

    # 4. Print ranked results
    print()
    for i, r in enumerate(results, 1):
        print("=" * 74)
        print(f"#{i}   similarity: {r['score']:.3f} / 1.0")
        if r["title"] is None:
            print(f"    (point {r['id']} has no matching Postgres row)")
            continue
        print(f"    Title:       {r['title']}")
        print(f"    Service:     {r['service']}      Severity: {r['severity']}")
        print(f"    Description: {r['description']}")
        print(f"    Root cause:  {r['root_cause']}")
        print(f"    Resolution:  {r['resolution']}")
    print("=" * 74)


def validate_collection() -> None:
    """Fail with actionable messages if the collection is missing, empty, or was
    seeded with vectors that don't match the current embedding model."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' was not found.\n"
            "Seed it first:  python new_ingest.py"
        ) from None
    if (info.points_count or 0) == 0:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' exists but is empty.\n"
            "Seed it first:  python new_ingest.py"
        )

    # The stored vectors must match the current embedding model's dimension;
    # e.g. switching providers/models after seeding would break the search.
    stored_size = _collection_vector_size(info)
    if stored_size is not None and stored_size != EMBED_DIM:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' stores {stored_size}-dim vectors but the "
            f"current embedding model produces {EMBED_DIM}-dim vectors.\n"
            "This usually means the embedding provider/model changed after seeding.\n"
            "Re-seed with a matching setup: set EMBED_PROVIDER / GEMINI_EMBED_MODEL in "
            ".env to the values used at ingest time, then delete the collection and run "
            "python new_ingest.py to rebuild it."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find past incidents similar to a new incident description."
    )
    parser.add_argument("query", nargs="*", help="description of the new incident")
    parser.add_argument("--top", type=int, default=3, help="results to return (default: 3)")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="min similarity score (default: 0.3)"
    )
    args = parser.parse_args()
    args.top = max(1, args.top)  # Qdrant rejects limit=0

    query_text = " ".join(args.query).strip()
    if not query_text:
        try:
            query_text = input("Describe the new incident: ").strip()
        except EOFError:
            print()
            return 0
    if not query_text:
        raise SystemExit("No query text provided.")

    # Friendly errors before we bother embedding anything.
    validate_collection()

    search_incidents(query_text, args.top, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
