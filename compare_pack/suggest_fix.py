"""
Generate a suggested fix for a NEW incident by retrieving similar PAST
incidents from the memory store and asking an LLM to reason over them.

Pipeline (the core of the project):
  1. Take a new incident description as input (plain string — no UI yet).
  2. Embed it the same way the seed data was embedded (see new_ingest.py).
  3. Query Qdrant for the top K most similar past incidents (cosine similarity).
  4. Use the returned ids to pull the full records back out of Postgres.
  5. Build a prompt containing the new incident + the retrieved past incidents.
  6. Call the LLM and print a suggested fix that cites which past incident(s)
     it matched on.

`retrieve_and_suggest()` below packages steps 1-6 into one function — it is the
single pipeline entry point used by both the app's /api/search route (the UI
button) and the /webhook/incident route (see webhook.py).

Usage:
  python suggest_fix.py "payment service timing out during flash sale"
  python suggest_fix.py --top 5 --threshold 0.2 "checkout 504s under traffic spike"
  python suggest_fix.py --dry-run "payment service timing out"   # print the prompt, skip the LLM
  python suggest_fix.py --provider groq "auth TLS handshake errors at 3am"
  python suggest_fix.py            # interactive prompt

Options:
  --top N          Number of similar incidents to retrieve (default: 3)
  --threshold X    Minimum similarity score, 0..1 (default: 0.3)
  --provider P     LLM provider: groq | gemini | openai (default: auto)
  --model NAME     Override the LLM model name
  --dry-run        Show the prompt and retrieved incidents without calling the LLM

LLM provider resolution (mirrors the embedding setup in new_ingest.py):
  - 'auto' (default) uses Groq (llama-3.3-70b-versatile) when GROQ_API_KEY is set
    — free tier, no credit card, fast; then Gemini when GEMINI_API_KEY is set
    (free tier); otherwise OpenAI when OPENAI_API_KEY is set (paid credits).
  - Force a provider with GEN_PROVIDER=groq|gemini|openai in .env, or --provider.
  - Default models: llama-3.3-70b-versatile (groq) / gemini-3.6-flash /
    gpt-4.1-mini. If a provider retires the defaults, override with
    GROQ_LLM_MODEL, GEN_LLM_MODEL or OPENAI_LLM_MODEL in .env, or --model.

Groq's API is OpenAI-compatible, so the same openai SDK (base_url
https://api.groq.com/openai/v1) handles both Groq and OpenAI calls.
Gemini calls use Google's Interactions API (client.interactions.create); on
older google-genai SDKs without it, the script falls back to generate_content.

Run new_ingest.py at least once first so the collection has vectors to search.
"""

import argparse
import os
import random
import re
import sys
import time

# Reuse the .env loading, embedding, and Qdrant client from new_ingest.py.
# Note: importing new_ingest runs its module-level setup (loads .env, creates
# the embedding + Qdrant clients) and fails fast with its error messages if
# env vars are missing.
from new_ingest import GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY
from query_incidents import retrieve_similar_incidents, validate_collection

# --- Configuration (from .env) --------------------------------------------
# LLM generation provider: "groq" | "gemini" | "openai" | "auto" (default: auto).
GEN_PROVIDER = (os.environ.get("GEN_PROVIDER") or "auto").strip().lower()
# Generation models (separate from the embedding models in new_ingest.py).
# If a provider retires the defaults, override in .env.
GEN_MODEL_DEFAULT = os.environ.get("GEN_LLM_MODEL", "gemini-3.6-flash")
GROQ_MODEL_DEFAULT = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-mini")
# Groq's API is OpenAI-compatible — the same openai SDK talks to it via this base URL.
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

TEMPERATURE = 0.4  # low-ish for deterministic, evidence-based suggestions

# Retry transient LLM failures (rate limits / server errors) the same way
# new_ingest.embed_text does — a live webhook must never hard-fail on a 429.
# The waits are kept SHORT on purpose: the free tier's rate limit resets every
# ~minute, so when it is exhausted a retry usually fails again anyway, and a
# long stall (15s+) would make every webhook/search/resolve request hang for
# tens of seconds. One quick retry rides out blips; sustained throttling
# fails fast (~5s) with a clean llm_error instead of blocking the caller.
MAX_GEN_RETRIES = 2
GEN_RETRY_BASE = 5.0  # seconds; one short backoff before giving up


def _is_transient_gen_error(exc: Exception) -> bool:
    """True for transient LLM failures worth retrying (rate limits / 5xx)."""
    # Gemini: google.genai errors carry EITHER a .code (older SDKs) or a
    # .status_code (google-genai 2.x sets status_code=429 on RateLimitError,
    # with no .code at all) — checking both catches rate limits on any version.
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    # OpenAI: openai.APIStatusError with a status_code
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
            return True
    except ImportError:
        pass
    # Last resort: google.genai error strings start with 'Error code: 429 - ...'
    return bool(re.search(r"error code: (429|5\d\d)", str(exc), re.IGNORECASE))


def _quota_retry_delay(exc: Exception) -> float | None:
    """Extract the 'Please retry in Ns' hint Google embeds in rate-limit errors.

    The free tier resets roughly every minute, so when a 429 names a concrete
    retry-in we wait exactly that long and let the retry actually succeed.
    Returns None when there is no usable hint (callers then use normal
    exponential backoff instead).
    """
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if not m:
        return None
    try:
        delay = float(m.group(1))
    except ValueError:
        return None
    # Cap at 30s: longer stalls would hang every search/webhook/resolve request.
    return delay if 0 < delay <= 30 else None

SYSTEM_PROMPT = (
    "You are an on-call incident response assistant for a SaaS platform. "
    "You will be given a description of a NEW incident and a ranked list of PAST "
    "incidents retrieved from an incident memory store by similarity search. "
    "Using the past incidents as evidence, propose a likely root cause and "
    "concrete, actionable fix steps for the new incident. Cite which past "
    "incident(s) you matched on (by their id and title) and explain why. Be "
    "specific and concise. Some past incidents were themselves written to memory "
    "by resolving an earlier similar incident (their titles begin with "
    "'Resolved:'). Treat those as direct evidence that a pattern is recurring, "
    "and cite them by id and title whenever they appear in the retrieved list, "
    "even when an older incident also matches. If none of the past incidents are "
    "truly relevant, say so explicitly and suggest the next diagnostic steps "
    "instead of forcing a match."
)


def resolve_gen_provider() -> str:
    """Decide which LLM provider to use for fix generation, with clear errors."""
    if GEN_PROVIDER in ("gemini", "google"):
        provider = "gemini"
    elif GEN_PROVIDER == "groq":
        provider = "groq"
    elif GEN_PROVIDER == "openai":
        provider = "openai"
    elif GEN_PROVIDER == "auto":
        # Preferred order: Groq (free tier, no credit card) -> Gemini (free
        # tier) -> OpenAI (paid credits).
        if GROQ_API_KEY:
            provider = "groq"
        elif GEMINI_API_KEY:
            provider = "gemini"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            raise SystemExit(
                "No LLM API key found for fix generation. Add GROQ_API_KEY "
                "(free, recommended — get one at https://console.groq.com/keys) "
                "or GEMINI_API_KEY (free — https://aistudio.google.com/apikey) "
                "to your .env file. Alternatively set OPENAI_API_KEY "
                "(requires paid OpenAI credits)."
            )
    else:
        raise SystemExit(
            f"Unknown GEN_PROVIDER '{GEN_PROVIDER}' in .env. "
            "Use 'groq', 'gemini', 'openai', or leave it unset for auto-detection."
        )

    key = {
        "gemini": GEMINI_API_KEY,
        "groq": GROQ_API_KEY,
        "openai": OPENAI_API_KEY,
    }[provider]
    if not key:
        raise SystemExit(
            f"GEN_PROVIDER is '{provider}' but the matching key is missing in .env "
            f"({provider.upper()}_API_KEY)."
        )
    return provider


def resolve_gen_model(provider: str) -> str:
    """Pick the default generation model for a provider (.env overrides win)."""
    if provider == "gemini":
        return GEN_MODEL_DEFAULT
    if provider == "groq":
        return GROQ_MODEL_DEFAULT
    return OPENAI_MODEL_DEFAULT


def build_user_prompt(new_incident: str, past: list[dict]) -> str:
    """Build the prompt: the new incident + the retrieved past incidents."""
    lines = [
        f"NEW INCIDENT:\n{new_incident}",
        "",
        "MOST SIMILAR PAST INCIDENTS (from incident memory):",
    ]
    for i, r in enumerate(past, 1):
        lines.append(
            f"\n[{i}] id {r['id']}, similarity {r['score']:.3f} - \"{r['title']}\" "
            f"(service: {r['service']}, severity: {r['severity']})\n"
            f"    Root cause:  {r['root_cause'] or '(not recorded)'}\n"
            f"    Resolution:  {r['resolution'] or '(not recorded)'}"
        )
    lines.append(
        "\nRespond in this format:\n"
        "1. Likely root cause (hypothesis)\n"
        "2. Suggested fix (concrete steps)\n"
        "3. Matched past incident(s): id / title, and why it matches\n"
        "4. If no past incident is a good match, say so explicitly and suggest "
        "the next diagnostic steps instead of forcing one."
    )
    return "\n".join(lines)


def _generate_fix_once(prompt: str, provider: str, model: str, system_prompt: str) -> str:
    """The actual single LLM call (no retries)."""
    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        if hasattr(client, "interactions"):
            # Interactions API: the current recommended Gemini API. Note this
            # endpoint does not accept a temperature parameter.
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=system_prompt,
            )
            return interaction.output_text
        # Older google-genai SDKs without the Interactions API
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
            ),
        )
        return response.text

    # OpenAI-compatible APIs: Groq's endpoint accepts the same /chat/completions
    # shape as OpenAI's, so one SDK call serves both providers.
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from None

    client = (
        OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        if provider == "groq"
        else OpenAI(api_key=OPENAI_API_KEY)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def generate_fix(
    prompt: str, provider: str, model: str, system_prompt: str = SYSTEM_PROMPT
) -> str:
    """Call the LLM and return the suggested fix text, retrying transient
    failures (rate limits / 5xx) with exponential backoff.

    `system_prompt` defaults to this module's fix-suggestion prompt; pass a
    different one (e.g. postmortem.py's) to reuse the same call machinery for
    another generation task.

    Raises the underlying exception if every retry fails (callers like
    retrieve_and_suggest surface it as llm_error instead of crashing).
    """
    last_exc = None
    for attempt in range(1, MAX_GEN_RETRIES + 1):
        try:
            return _generate_fix_once(prompt, provider, model, system_prompt)
        except Exception as exc:  # noqa: BLE001 - we classify the error below
            last_exc = exc
            if not _is_transient_gen_error(exc) or attempt >= MAX_GEN_RETRIES:
                break
            # On a rate limit, wait out the exact 'retry in Ns' window Google
            # reports (capped at 30s); otherwise exponential backoff + jitter.
            delay = _quota_retry_delay(exc) or (
                GEN_RETRY_BASE * (2 ** (attempt - 1)) + random.uniform(0, 2.0)
            )
            print(f"  LLM call failed ({exc.__class__.__name__}), retrying in {delay:.0f}s "
                  f"(attempt {attempt}/{MAX_GEN_RETRIES})...")
            time.sleep(delay)
    raise last_exc


def retrieve_and_suggest(
    description: str, top: int = 5, threshold: float = 0.22
) -> dict:
    """The shared retrieve-and-suggest pipeline: embed the new incident
    description, retrieve similar PAST incidents from Qdrant/Postgres, then ask
    the LLM for a suggested fix.

    Returns a dict:
        {matches: [past-incident dicts], suggestion: str|None,
         llm_error: str|None, note: str|None}

    This is the single entry point used by BOTH the page's /api/search route
    (the button in the UI) and the /webhook/incident route — a webhook call and
    a button click exercise the exact same code path.

    Raises SystemExit (with an actionable message) when the memory store is
    missing or empty; LLM failures are reported in llm_error instead of being
    fatal, so retrieval results are always returned.
    """
    validate_collection()
    # 1 + 2: embed the query and retrieve similar past incidents from Qdrant,
    # enriched with their full records from Postgres.
    results = retrieve_similar_incidents(description, top, threshold)
    past = [r for r in results if r["title"] is not None]

    if not past:
        return {
            "matches": [],
            "suggestion": None,
            "llm_error": None,
            "note": "No similar past incidents found above the threshold.",
        }

    # 3: build the prompt and ask the LLM for a suggested fix. LLM failures
    # (including exhausted rate limits) are reported in llm_error, never fatal
    # — retrieval results are still returned.
    prompt = build_user_prompt(description, past)
    suggestion, llm_error = None, None
    try:
        provider = resolve_gen_provider()
        model = resolve_gen_model(provider)
        suggestion = generate_fix(prompt, provider, model)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - non-fatal by design
        llm_error = str(exc)

    return {"matches": past, "suggestion": suggestion, "llm_error": llm_error, "note": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest a fix for a new incident using similar past incidents."
    )
    parser.add_argument("query", nargs="*", help="description of the new incident")
    parser.add_argument(
        "--top", type=int, default=3, help="past incidents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="min similarity score (default: 0.3)",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini", "openai"],
        default=None,
        help="LLM provider (default: auto)",
    )
    parser.add_argument("--model", default=None, help="LLM model name override")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompt without calling the LLM",
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

    if args.dry_run:
        # Dry runs only build the prompt; they must not require an LLM key.
        provider = model = None
        print(f"Dry run - retrieving top {args.top} past incident(s)...\n")
    else:
        provider = resolve_gen_provider()
        if args.provider:
            provider = args.provider
            key = {
                "gemini": GEMINI_API_KEY,
                "groq": GROQ_API_KEY,
                "openai": OPENAI_API_KEY,
            }[provider]
            if not key:
                raise SystemExit(
                    f"--provider {provider} but the matching key is missing in .env "
                    f"({provider.upper()}_API_KEY)."
                )
        model = args.model or resolve_gen_model(provider)

        print(f"LLM: {provider} ({model}) - retrieving top {args.top} past incident(s)...\n")

    # 1-4. Embed the new incident, find the most similar past incidents in
    # Qdrant, and pull their full records back out of Postgres.
    results = retrieve_similar_incidents(query_text, args.top, args.threshold)
    if not results:
        print(f"\nNo similar incidents found above the score threshold ({args.threshold}).")
        print("  - Lower the bar with --threshold, e.g. --threshold 0.1")
        print("  - Make sure the collection has data: python new_ingest.py")
        return 0
    past = [r for r in results if r["title"] is not None]
    if not past:
        raise SystemExit(
            "Similar points were found in Qdrant, but none have a matching "
            "Postgres row. Re-seed to sync the stores: python new_ingest.py"
        )
    if len(past) < len(results):
        print(
            f"Note: {len(results) - len(past)} hit(s) had no matching Postgres "
            "row and were skipped.\n"
        )

    print("=" * 74)
    print(f"Retrieved {len(past)} similar past incident(s):")
    for r in past:
        print(f"  [{r['id']}] similarity {r['score']:.3f} - {r['title']} ({r['service']})")
    print("=" * 74)

    # 5. Build the prompt with the new incident + the retrieved past incidents.
    prompt = build_user_prompt(query_text, past)
    if args.dry_run:
        print("\n--- PROMPT (dry run, no LLM call) ---\n")
        print(prompt)
        print("\n--- END PROMPT ---")
        return 0

    # 6. Call the LLM and print the suggested fix.
    try:
        fix = generate_fix(prompt, provider, model)
    except Exception as exc:  # noqa: BLE001 - surface the real error with context
        hint = ""
        if provider == "gemini" and (
            "not_found" in str(exc).lower() or "no longer available" in str(exc).lower()
        ):
            hint = (
                "\n  The model name may have been retired. Set GEN_LLM_MODEL to a "
                "current model in .env (e.g. gemini-3.6-flash or gemini-3.5-flash), "
                "or pass --model."
            )
        raise SystemExit(f"LLM call failed:\n  {exc}{hint}") from exc

    print("\n" + "=" * 74)
    print("SUGGESTED FIX")
    print("=" * 74)
    print(fix)
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
