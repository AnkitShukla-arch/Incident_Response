"""
Store a resolved incident back into the incident memory store — the write-back
step that makes memory compound.

This is the closing loop of the pipeline: after the LLM suggests a fix for a NEW
incident (suggest_fix.py) and an engineer edits/approves it, this module inserts
the incident as a new row in Postgres, embeds it with the same embedding model
used by new_ingest.py, and upserts the vector into Qdrant. From that moment on,
future incidents similar to this one match against it too — the memory grew by
one, and the next suggestion gets smarter because of it.

Pipeline position:
  new_ingest.py        seeds the initial memory (12 incidents)
  query_incidents.py   retrieval only
  suggest_fix.py       retrieval + LLM suggestion
  resolve_incident.py  WRITE-BACK: approved fix -> durable, searchable memory
  app.py               the page that wraps all of the above

Usage (CLI):
  python resolve_incident.py "payment api 504s under flash sale load" \
      --resolution "Bumped the payments connection pool to 100, added a circuit breaker..." \
      --service payments --severity high

  python resolve_incident.py "auth handshake failures at 3am" \
      --resolution "..." --title "My custom title"

  python resolve_incident.py --check     # verify Postgres + Qdrant connectivity
"""

import argparse
import sys

from qdrant_client.models import PointStruct

from new_ingest import COLLECTION_NAME, connect_db, embed_text, qdrant_client


def ensure_schema(conn) -> None:
    """Make the root_cause column nullable.

    Seed data always has a root_cause, but incidents resolved through the UI
    store the LLM's freeform suggestion as `resolution` and don't reliably know
    a separate root cause. This migration is idempotent, so it's safe to run on
    every startup (it's a metadata-only change on first run, a no-op after).
    """
    conn.execute("ALTER TABLE incidents ALTER COLUMN root_cause DROP NOT NULL")
    conn.commit()


def derive_title(description: str, max_words: int = 8) -> str:
    """Build a readable title from the incident description, e.g.
    'Payment api 504s under flash sale load' -> 'Resolved: payment api 504s...'."""
    words = []
    for word in description.strip().split():
        cleaned = word.strip(".,;:!?()[]\"'")
        if cleaned:
            words.append(cleaned)
        if len(words) >= max_words:
            break
    if not words:
        return "Resolved incident"
    # Keep original casing (API, OOMKilled, 504 …), just uppercase the first letter.
    title = " ".join(words)
    title = title[0].upper() + title[1:]
    return f"Resolved: {title}"


def store_resolved_incident(
    description: str,
    resolution: str,
    service: str | None = None,
    severity: str | None = None,
    title: str | None = None,
) -> dict:
    """Insert an approved incident into Postgres, embed it, and upsert it into
    Qdrant. Returns the new record as a dict, e.g.:

        {id, title, description, resolution, service, severity, created_at}

    The write is transactional: the Postgres INSERT stays uncommitted until the
    Qdrant upsert succeeds, so a failure can never leave the two stores out of
    sync (a re-run is always safe).
    """
    description = (description or "").strip()
    resolution = (resolution or "").strip()
    if not description:
        raise ValueError("A description of the resolved incident is required.")
    if not resolution:
        raise ValueError("A resolution (the approved fix) is required.")

    title = ((title or derive_title(description)).strip()) or "Resolved incident"

    # 1. Embed the new knowledge (title + description + resolution gives the
    #    richest signal for matching future incidents — the fix itself is the
    #    most valuable part to remember).
    text_to_embed = f"{title}. {description} {resolution}"
    vector = embed_text(text_to_embed)

    # 2. Insert into Postgres (uncommitted for now) and get the new id back.
    with connect_db() as conn:
        ensure_schema(conn)
        row = conn.execute(
            """
            INSERT INTO incidents
                (title, description, root_cause, resolution, service, severity, status)
            VALUES
                (%(title)s, %(description)s, NULL, %(resolution)s, %(service)s, %(severity)s, 'resolved')
            RETURNING id, created_at
            """,
            {
                "title": title,
                "description": description,
                "resolution": resolution,
                "service": service,
                "severity": severity,
            },
        ).fetchone()
        incident_id, created_at = row

        # 3. Upsert the vector into Qdrant, keyed by the same id as Postgres.
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=incident_id,
                    vector=vector,
                    payload={
                        "service": service,
                        "severity": severity,
                        "title": title,
                    },
                )
            ],
        )
        # 4. Only now commit — if the upsert above threw, this never runs and
        #    the connection close rolls the INSERT back automatically.
        conn.commit()

    print(
        f"Stored resolved incident #{incident_id}: {title} "
        f"({service or '-'}/{severity or '-'}) -> memory now compounds with this fix."
    )
    return {
        "id": incident_id,
        "title": title,
        "description": description,
        "resolution": resolution,
        "service": service,
        "severity": severity,
        "created_at": str(created_at),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store an approved fix back into the incident memory store."
    )
    parser.add_argument("description", nargs="*", help="the resolved incident's description")
    parser.add_argument("--resolution", required=True, help="the approved fix (the LLM suggestion, as-is or edited)")
    parser.add_argument("--title", default=None, help="optional custom title (default: derived from description)")
    parser.add_argument("--service", default=None, help="service tag, e.g. payments")
    parser.add_argument("--severity", default=None, choices=["critical", "high", "medium", "low"], help="severity tag")
    parser.add_argument("--check", action="store_true", help="verify Postgres + Qdrant connectivity, write nothing")
    args = parser.parse_args()

    if args.check:
        with connect_db() as conn:
            conn.execute("SELECT 1")
        print("Postgres OK")
        qdrant_client.get_collections()
        print("Qdrant OK")
        return 0

    description = " ".join(args.description).strip()
    if not description:
        try:
            description = input("Describe the resolved incident: ").strip()
        except EOFError:
            print()
            return 0
    if not description:
        raise SystemExit("No description provided.")

    store_resolved_incident(
        description=description,
        resolution=args.resolution,
        service=args.service,
        severity=args.severity,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
