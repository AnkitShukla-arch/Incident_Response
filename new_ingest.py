"""
Seeds the memory store: inserts each incident into Postgres, embeds it,
and upserts the vector + metadata into Qdrant.

Run once before the demo:  python new_ingest.py
Validate your setup first: python new_ingest.py --check   (no data is written)
Test your embedding key:   python new_ingest.py --test-embed   (embeds one string)

Embeddings:
  - Uses Google Gemini "gemini-embedding-001" (free tier) when GEMINI_API_KEY is set.
    Get a free key (no credit card) at https://aistudio.google.com/apikey
    If Google retires a model name, override it with GEMINI_EMBED_MODEL in .env.
  - Falls back to OpenAI "text-embedding-3-small" only if GEMINI_API_KEY is missing
    but OPENAI_API_KEY is set (note: OpenAI requires paid credits).
  - Set EMBED_PROVIDER=gemini|openai in .env to force a specific provider.
"""

import json
import os
import random
import socket
import sys
import time

from dotenv import load_dotenv

# Reads variables from a .env file in the same folder into os.environ.
# override=True so the project's .env always wins over stale OS-level variables
# (e.g. an old PGHOST exported on Windows that would otherwise shadow the file).
load_dotenv(override=True)

import psycopg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# --- Configuration (from .env) --------------------------------------------
# Postgres can be configured two ways:
#   1. A full connection string (recommended for Supabase):
#        DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
#   2. Individual variables: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
#
# For Supabase, use the CURRENT host format (e.g. aws-0-<region>.pooler.supabase.com).
# The legacy "db.<project-ref>.supabase.co" hostname only resolves over IPv6 and
# fails from most networks. With the pooler hosts, PGUSER must be
# "postgres.<project-ref>".
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")

PG_HOST = os.environ.get("PGHOST")
PG_PORT = os.environ.get("PGPORT", "5432")
PG_DATABASE = os.environ.get("PGDATABASE", "postgres")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD")

QDRANT_URL = os.environ.get("QDRANT_URL")               # e.g. http://localhost:6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")       # only needed for Qdrant Cloud

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Providers: "gemini-embedding-001" (Gemini, 768 dims) or "text-embedding-3-small" (OpenAI, 1536 dims)
EMBED_PROVIDER = (os.environ.get("EMBED_PROVIDER") or "auto").strip().lower()
# Override with GEMINI_EMBED_MODEL in .env if Google retires the default model name.
GEMINI_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_DIM = 768
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536

COLLECTION_NAME = "incidents"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry (with jitter)


def resolve_embed_provider() -> str:
    """Decide which embedding provider to use, with clear errors."""
    if EMBED_PROVIDER in ("gemini", "google"):
        provider = "gemini"
    elif EMBED_PROVIDER == "openai":
        provider = "openai"
    elif EMBED_PROVIDER == "auto":
        if GEMINI_API_KEY:
            provider = "gemini"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            raise SystemExit(
                "No embedding API key found. Add GEMINI_API_KEY (free, recommended) "
                "to your .env file — get one at https://aistudio.google.com/apikey. "
                "Alternatively set OPENAI_API_KEY (requires paid OpenAI credits)."
            )
    else:
        raise SystemExit(
            f"Unknown EMBED_PROVIDER '{EMBED_PROVIDER}' in .env. "
            "Use 'gemini', 'openai', or leave it unset for auto-detection."
        )

    key = GEMINI_API_KEY if provider == "gemini" else OPENAI_API_KEY
    if not key:
        raise SystemExit(
            f"EMBED_PROVIDER is '{provider}' but the matching key is missing in .env "
            f"({'GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'})."
        )
    return provider


if not QDRANT_URL:
    raise SystemExit(
        "Missing QDRANT_URL. Add it to your .env file "
        "(e.g. http://localhost:6333 for local Qdrant, or your Qdrant Cloud URL)."
    )

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

EMBED_PROVIDER = resolve_embed_provider()
EMBED_MODEL = GEMINI_MODEL if EMBED_PROVIDER == "gemini" else OPENAI_MODEL
EMBED_DIM = GEMINI_DIM if EMBED_PROVIDER == "gemini" else OPENAI_DIM

# Create the embedding client once (not per call).
if EMBED_PROVIDER == "gemini":
    try:
        from google import genai  # noqa: E402
        from google.genai import types  # noqa: E402

        embed_client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        raise SystemExit(
            "google-genai SDK is not installed. Run: pip install google-genai"
        ) from None
else:
    from openai import OpenAI  # noqa: E402

    embed_client = OpenAI(api_key=OPENAI_API_KEY)


# --- Embedding (with retry / backoff) -------------------------------------
def _gemini_embed(text: str) -> list[float]:
    # Pin the output dimensionality so the vector size always matches the
    # Qdrant collection, regardless of the model's default.
    response = embed_client.models.embed_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=GEMINI_DIM),
    )
    return list(response.embeddings[0].values)


def _openai_embed(text: str) -> list[float]:
    response = embed_client.embeddings.create(model=OPENAI_MODEL, input=text)
    return response.data[0].embedding


def _retryable(exc: Exception) -> bool:
    """True for transient failures worth retrying (rate limits / server errors)."""
    # OpenAI: openai.RateLimitError (429) / APIStatusError with 5xx
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
            return True
    except ImportError:
        pass
    # Gemini: google.genai.errors.APIError carries a .code
    code = getattr(exc, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    # Generic fallback: retry any exception a couple of times to ride out hiccups
    return isinstance(exc, (ConnectionError, TimeoutError))


def embed_text(text: str) -> list[float]:
    """Embed text with exponential backoff + jitter on transient errors."""
    embed_fn = _gemini_embed if EMBED_PROVIDER == "gemini" else _openai_embed
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return embed_fn(text)
        except Exception as exc:  # noqa: BLE001 - we inspect the error below
            last_exc = exc
            if not _retryable(exc):
                break
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"  Embedding call failed ({exc.__class__.__name__}), retrying in {delay:.1f}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(delay)

    raise SystemExit(
        f"Embedding failed after {MAX_RETRIES} attempt(s): {last_exc}\n\n"
        "Troubleshooting:\n"
        "  - If the message mentions 'no credits' or 'quota': the API key has no paid credits.\n"
        "    Switch to Gemini (free): add a free GEMINI_API_KEY to .env, then delete the old\n"
        "    'incidents' Qdrant collection or start with a fresh vector DB.\n"
        "  - For transient 'rate limit' errors the script already retried automatically.\n"
        "  - If the error is a 404 'model not found' (e.g. text-embedding-004): Google\n"
        "    periodically retires model names. Set GEMINI_EMBED_MODEL=gemini-embedding-001\n"
        "    (or gemini-embedding-2) in .env, or change GEMINI_MODEL in this script.\n"
        "  - Verify your key is valid: python new_ingest.py --test-embed"
    ) from last_exc


def check_host_resolves(host: str) -> None:
    """Fail fast with an actionable message when a DB hostname won't resolve."""
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror:
        hint = ""
        if host.endswith(".supabase.co"):
            hint = (
                f"\n\n'{host}' is the legacy Supabase hostname, which now only "
                "resolves over IPv6 and cannot be reached from most networks.\n"
                "Use the current host from Supabase Dashboard > Project Settings > "
                "Database, e.g. 'aws-0-<region>.pooler.supabase.com', with PGUSER "
                "set to 'postgres.<project-ref>'."
            )
        raise SystemExit(
            f"Could not resolve database host '{host}' (DNS lookup failed).\n"
            f"Check the PGHOST value in your .env file.{hint}"
        )


def connect_db():
    """Open a Postgres connection from the .env config, with clear errors."""
    if DATABASE_URL:
        conninfo = DATABASE_URL
        if "sslmode" not in conninfo:
            sep = "&" if "?" in conninfo else "?"
            conninfo = f"{conninfo}{sep}sslmode=require"
        try:
            params = psycopg.conninfo.conninfo_to_dict(conninfo)
        except psycopg.ProgrammingError as exc:
            raise SystemExit(
                f"DATABASE_URL is not a valid Postgres connection string:\n  {exc}"
            ) from exc
    else:
        if not (PG_HOST and PG_PASSWORD):
            raise SystemExit(
                "Missing database credentials in .env. Either set DATABASE_URL to "
                "your full Supabase connection string, or set PGHOST, PGPORT, "
                "PGDATABASE, PGUSER and PGPASSWORD."
            )
        params = {
            "host": PG_HOST,
            "port": PG_PORT,
            "dbname": PG_DATABASE,
            "user": PG_USER,
            "password": PG_PASSWORD,
            "sslmode": "require",
        }

    host = params.get("host")
    if host and "," not in host and not host.startswith("/"):
        check_host_resolves(host)

    try:
        return psycopg.connect(connect_timeout=15, **params)
    except psycopg.OperationalError as exc:
        raise SystemExit(f"Could not connect to Postgres:\n  {exc}") from exc


def setup_postgres(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            resolution TEXT NOT NULL,
            service TEXT,
            severity TEXT,
            status TEXT DEFAULT 'resolved',
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    conn.commit()


def existing_titles(conn) -> set[str]:
    """Titles already seeded, so re-running the script never duplicates data."""
    rows = conn.execute("SELECT title FROM incidents").fetchall()
    return {row[0] for row in rows}


def _collection_vector_size(info) -> int | None:
    """Extract vector size from a CollectionInfo, across qdrant-client versions.

    qdrant-client >= 1.19 stores vectors under config.params.vectors; older
    versions expose info.vectors directly. Named vectors come back as a dict.
    """
    vectors_info = getattr(info, "vectors", None)
    if vectors_info is None:
        params = getattr(getattr(info, "config", None), "params", None)
        vectors_info = getattr(params, "vectors", None)
    if vectors_info is None:
        return None
    if isinstance(vectors_info, dict):  # named vectors: {name: VectorParams}
        vec = (vectors_info.get("") or vectors_info.get("default")
               or next(iter(vectors_info.values()), None))
        return getattr(vec, "size", None) if vec else None
    return getattr(vectors_info, "size", None)


def setup_qdrant() -> bool:
    """Ensure the Qdrant collection exists with the right vector size.

    Returns True if the collection was (re)created, False if it already existed.
    """
    existing_names = {c.name for c in qdrant_client.get_collections().collections}
    if COLLECTION_NAME not in existing_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        return True
    # Collection already exists — make sure its vector size matches the provider.
    current_size = _collection_vector_size(qdrant_client.get_collection(COLLECTION_NAME))
    if current_size is not None and current_size != EMBED_DIM:
        # If the old collection is empty (e.g. a failed earlier run), recreate it
        # automatically; otherwise ask the user before touching data.
        try:
            point_count = qdrant_client.count(
                collection_name=COLLECTION_NAME, exact=True
            ).count
        except Exception:  # noqa: BLE001 - count is only used for the safety check
            point_count = None
        if point_count == 0:
            print(f"Recreating empty '{COLLECTION_NAME}' collection with {EMBED_DIM}-dim vectors "
                  f"(old size was {current_size}).")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            return True
        raise SystemExit(
            f"Qdrant collection '{COLLECTION_NAME}' already exists with vector size "
            f"{current_size}, but '{EMBED_MODEL}' produces {EMBED_DIM}-dim vectors."
            "\nDelete the old collection so it can be recreated (it contains data): "
            "see the Qdrant dashboard/CLI, or run:  qdrant_client.collection delete 'incidents'"
        )
    return False


def ingest(conn, incidents: list[dict]):
    points = []
    already = existing_titles(conn)
    skipped = 0
    for incident in incidents:
        if incident["title"] in already:
            skipped += 1
            print(f"Skipping (already seeded): {incident['title']}")
            continue

        # 1. Insert into Postgres, get back the generated id
        row = conn.execute(
            """
            INSERT INTO incidents (title, description, root_cause, resolution, service, severity)
            VALUES (%(title)s, %(description)s, %(root_cause)s, %(resolution)s, %(service)s, %(severity)s)
            RETURNING id
            """,
            incident,
        ).fetchone()
        incident_id = row[0]

        # 2. Embed a combined text representation (title + description + root_cause
        #    gives the richest signal for matching future incidents)
        text_to_embed = f"{incident['title']}. {incident['description']} {incident['root_cause']}"
        vector = embed_text(text_to_embed)

        # 3. Prepare the Qdrant point, keyed by the same id as the Postgres row
        points.append(
            PointStruct(
                id=incident_id,
                vector=vector,
                payload={
                    "service": incident["service"],
                    "severity": incident["severity"],
                    "title": incident["title"],
                },
            )
        )
        print(f"Ingested #{incident_id}: {incident['title']}")

    # Upsert into Qdrant FIRST, then commit Postgres: if the upsert fails the
    # Postgres transaction rolls back, so a re-run can safely re-seed everything.
    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    conn.commit()
    if points:
        print(f"\nUpserted {len(points)} vectors into Qdrant collection '{COLLECTION_NAME}'.")
    if skipped:
        print(f"Skipped {skipped} incident(s) that were already seeded.")
    if not points:
        print("\nNothing new to ingest — all incidents were already seeded.")


def check_connections():
    """Open and immediately close both external connections; writes no data."""
    print(f"Embedding provider: {EMBED_PROVIDER} ({EMBED_MODEL}, {EMBED_DIM} dims)")
    if EMBED_PROVIDER == "openai":
        print("  Note: using OpenAI embeddings requires paid credits on your OpenAI account.")
        print("  Add a free GEMINI_API_KEY to .env (https://aistudio.google.com/apikey) to switch.")
    print("Checking Postgres connection...")
    with connect_db() as conn:
        conn.execute("SELECT 1")
    print("  -> Postgres OK")
    print("Checking Qdrant connection...")
    qdrant_client.get_collections()
    print("  -> Qdrant OK")
    print("\nPostgres and Qdrant connections are healthy. You can now run: python new_ingest.py")


def test_embed():
    """Verify the embedding API key actually works by embedding one string."""
    print(f"Testing embedding with provider '{EMBED_PROVIDER}' ({EMBED_MODEL})...")
    vector = embed_text("test: payment service timeout")
    print(f"  -> OK, embedding has {len(vector)} dimensions")
    if len(vector) != EMBED_DIM:
        print(f"  Warning: expected {EMBED_DIM} dims but got {len(vector)}. "
              "If you changed providers, delete the old Qdrant 'incidents' collection.")
    else:
        print("  -> Dimension matches the Qdrant collection config.")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--check":
        check_connections()
        return
    if args and args[0] == "--test-embed":
        test_embed()
        return

    with open("seed_incidents.json") as f:
        incidents = json.load(f)

    recreated = setup_qdrant()

    with connect_db() as conn:
        setup_postgres(conn)
        if recreated:
            # Fresh collection: clear any previously seeded rows so Postgres and
            # Qdrant stay in sync and the seed runs cleanly from scratch.
            deleted = conn.execute("DELETE FROM incidents").rowcount
            conn.commit()
            if deleted:
                print(f"Cleared {deleted} previously seeded row(s) for a clean re-seed.")
        ingest(conn, incidents)


if __name__ == "__main__":
    main()
