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

Usage:
  python suggest_fix.py "payment service timing out during flash sale"
  python suggest_fix.py --top 5 --threshold 0.2 "checkout 504s under traffic spike"
  python suggest_fix.py --dry-run "payment service timing out"   # print the prompt, skip the LLM
  python suggest_fix.py --provider openai "auth TLS handshake errors at 3am"
  python suggest_fix.py            # interactive prompt

Options:
  --top N          Number of similar incidents to retrieve (default: 3)
  --threshold X    Minimum similarity score, 0..1 (default: 0.3)
  --provider P     LLM provider: gemini | openai (default: auto)
  --model NAME     Override the LLM model name
  --dry-run        Show the prompt and retrieved incidents without calling the LLM

LLM provider resolution (mirrors the embedding setup in new_ingest.py):
  - 'auto' (default) uses Gemini when GEMINI_API_KEY is set (free tier,
    recommended), otherwise OpenAI when OPENAI_API_KEY is set (paid credits).
  - Force a provider with GEN_PROVIDER=gemini|openai in .env, or --provider.
  - Default models: gemini-3.6-flash / gpt-4.1-mini. If a model is retired,
    override with GEN_LLM_MODEL or OPENAI_LLM_MODEL in .env, or --model.

Gemini calls use Google's Interactions API (client.interactions.create); on
older google-genai SDKs without it, the script falls back to generate_content.

Run new_ingest.py at least once first so the collection has vectors to search.
"""

import argparse
import os
import sys

# Reuse the .env loading, embedding, and Qdrant client from new_ingest.py.
# Note: importing new_ingest runs its module-level setup (loads .env, creates
# the embedding + Qdrant clients) and fails fast with its error messages if
# env vars are missing.
from new_ingest import GEMINI_API_KEY, OPENAI_API_KEY
from query_incidents import retrieve_similar_incidents, validate_collection

# --- Configuration (from .env) --------------------------------------------
# LLM generation provider: "gemini" | "openai" | "auto" (default: auto).
GEN_PROVIDER = (os.environ.get("GEN_PROVIDER") or "auto").strip().lower()
# Generation models (separate from the embedding models in new_ingest.py).
# If Google/OpenAI retire the defaults, override in .env.
GEN_MODEL_DEFAULT = os.environ.get("GEN_LLM_MODEL", "gemini-3.6-flash")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-mini")

TEMPERATURE = 0.4  # low-ish for deterministic, evidence-based suggestions

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
    elif GEN_PROVIDER == "openai":
        provider = "openai"
    elif GEN_PROVIDER == "auto":
        if GEMINI_API_KEY:
            provider = "gemini"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            raise SystemExit(
                "No LLM API key found for fix generation. Add GEMINI_API_KEY "
                "(free, recommended) to your .env file — get one at "
                "https://aistudio.google.com/apikey. Alternatively set "
                "OPENAI_API_KEY (requires paid OpenAI credits)."
            )
    else:
        raise SystemExit(
            f"Unknown GEN_PROVIDER '{GEN_PROVIDER}' in .env. "
            "Use 'gemini', 'openai', or leave it unset for auto-detection."
        )

    key = GEMINI_API_KEY if provider == "gemini" else OPENAI_API_KEY
    if not key:
        raise SystemExit(
            f"GEN_PROVIDER is '{provider}' but the matching key is missing in .env "
            f"({'GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'})."
        )
    return provider


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


def generate_fix(prompt: str, provider: str, model: str) -> str:
    """Call the LLM and return the suggested fix text."""
    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        if hasattr(client, "interactions"):
            # Interactions API: the current recommended Gemini API. Note this
            # endpoint does not accept a temperature parameter.
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=SYSTEM_PROMPT,
            )
            return interaction.output_text
        # Older google-genai SDKs without the Interactions API
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
            ),
        )
        return response.text

    # OpenAI
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from None

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content or ""


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
        choices=["gemini", "openai"],
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
            key = GEMINI_API_KEY if provider == "gemini" else OPENAI_API_KEY
            if not key:
                raise SystemExit(
                    f"--provider {provider} but the matching key is missing in .env "
                    f"({'GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'})."
                )
        model = args.model or (
            GEN_MODEL_DEFAULT if provider == "gemini" else OPENAI_MODEL_DEFAULT
        )

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
