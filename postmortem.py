"""
Generate and persist postmortem documents — the final stage of the pipeline.

Pipeline position (after the write-back in resolve_incident.py):

  feed -> webhook -> retrieval (query_incidents.py)
                   -> suggestion (suggest_fix.py)
                   -> write-back (resolve_incident.py)   <-- incident row exists
                   -> POSTMORTEM (this module)           <-- expands that row

Given the recorded facts of a single incident (title, description, root_cause,
resolution, service, severity — the same fields already stored on the incidents
row), this module asks the LLM to write a structured Markdown postmortem using
ONLY those facts. The prompt explicitly forbids inventing anything: fields that
were not recorded are reported as "not recorded" rather than guessed.

  generate_postmortem(incident: dict) -> str
      Builds the prompt and calls the same LLM machinery as suggest_fix.py
      (provider auto-detection from .env, same retry/error conventions).
      Returns the full Markdown document as a single string.

  save_postmortem(conn, incident_id: int, postmortem_markdown: str) -> None
      Writes the document into the postmortem_doc column of the incident row
      (UPDATE incidents SET postmortem_doc = ... WHERE id = ...). Returns
      nothing — its job is the side effect of permanently saving the doc.

In the app, both run immediately after write-back: /api/resolve stores the
short resolution text via store_resolved_incident(), then calls
generate_postmortem() and save_postmortem() on the same row — one extra LLM
call and one extra UPDATE that expands the short text into a full document.

Usage (CLI):
  python postmortem.py --incident 12            # generate + print the doc
  python postmortem.py --incident 12 --save     # ... and write it to the row
  python postmortem.py --incident 12 --dry-run  # print the prompt, no LLM call
"""

import argparse
import sys

# Reuse the LLM provider resolution, model selection and call machinery from
# suggest_fix.py instead of duplicating the Groq/Gemini/OpenAI setup here.
from suggest_fix import (
    generate_fix,
    resolve_gen_model,
    resolve_gen_provider,
)

# --- Configuration ----------------------------------------------------------
# Required sections, in order. (The spec lists five headings; the doc must
# contain all of them, each as a level-2 Markdown heading.)
POSTMORTEM_SECTIONS = [
    "Summary",
    "Timeline",
    "Root Cause",
    "Resolution",
    "Prevention / Follow-ups",
]

POSTMORTEM_SYSTEM_PROMPT = (
    "You are an incident postmortem writer for a SaaS platform. You will be "
    "given the recorded facts of a single incident: title, service, severity, "
    "description, root cause, and resolution. Write a structured postmortem "
    "document in Markdown using ONLY those facts. You must NOT invent, "
    "extrapolate, or fabricate anything — no timestamps, metrics, names, "
    "events, or steps that were not provided. When a fact was not recorded, "
    "state explicitly that it was not recorded instead of guessing. The "
    "document must contain exactly these five sections, each as a level-2 "
    "Markdown heading (##), in this order: Summary, Timeline, Root Cause, "
    "Resolution, Prevention / Follow-ups. For the Timeline section, restate "
    "only the sequence implied by the description (if any); if no timeline was "
    "recorded, say so. For Prevention / Follow-ups, propose only actions that "
    "follow logically from the recorded root cause and resolution; if none can "
    "be inferred, say so. Do not add sections beyond these five."
)


def build_postmortem_prompt(incident: dict) -> str:
    """Turn the incident's recorded fields into the LLM user prompt."""
    title = (incident.get("title") or "").strip() or "(no title recorded)"
    service = (incident.get("service") or "").strip() or "not recorded"
    severity = (incident.get("severity") or "").strip() or "not recorded"
    description = (incident.get("description") or "").strip() or "(not recorded)"
    root_cause = (incident.get("root_cause") or "").strip() or "not recorded"
    resolution = (incident.get("resolution") or "").strip() or "not recorded"

    return (
        "Write a postmortem document for the incident described by the facts "
        "below. Use ONLY these facts. If a fact was not recorded, say so in "
        "the document — do not invent it.\n"
        "\n"
        "INCIDENT FACTS (recorded):\n"
        f"- Title: {title}\n"
        f"- Service: {service}\n"
        f"- Severity: {severity}\n"
        f"- Description: {description}\n"
        f"- Root cause: {root_cause}\n"
        f"- Resolution: {resolution}\n"
        "\n"
        "Output the complete document in Markdown, with exactly these five "
        "level-2 headings in this order:\n"
        + "\n".join(f"## {s}" for s in POSTMORTEM_SECTIONS)
    )


def generate_postmortem(incident: dict) -> str:
    """Ask the LLM for a full postmortem document built from the incident's
    recorded facts only. Returns the Markdown document as a single string.

    Raises SystemExit (with an actionable message) if no LLM key is configured
    or the LLM call fails — callers decide whether that is fatal (the CLI) or
    recoverable (the /api/resolve route reports postmortem_error but still
    returns the successful write-back).
    """
    incident = {k: v for k, v in (incident or {}).items()}
    description = (incident.get("description") or "").strip()
    if not description:
        raise ValueError("A description is required to write a postmortem.")

    provider = resolve_gen_provider()
    model = resolve_gen_model(provider)
    try:
        return generate_fix(
            build_postmortem_prompt(incident),
            provider,
            model,
            system_prompt=POSTMORTEM_SYSTEM_PROMPT,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - surface the real error with context
        raise SystemExit(f"LLM call failed:\n  {exc}") from exc


def save_postmortem(conn, incident_id: int, postmortem_markdown: str) -> None:
    """Persist the document into the incident row's postmortem_doc column.
    Returns nothing — the side effect (the UPDATE) is the whole point.
    """
    conn.execute(
        "UPDATE incidents SET postmortem_doc = %s WHERE id = %s",
        (postmortem_markdown, incident_id),
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate (and optionally save) a postmortem document."
    )
    parser.add_argument(
        "--incident", type=int, required=True,
        help="id of the incident row to read the recorded facts from",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="write the generated document into the row's postmortem_doc column",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the prompt only — no LLM call, nothing written",
    )
    args = parser.parse_args()

    # Read the recorded facts for the row from Postgres (same columns the app
    # stores on every incident).
    from new_ingest import connect_db

    with connect_db() as conn:
        row = conn.execute(
            "SELECT id, title, description, root_cause, resolution, service, "
            "severity FROM incidents WHERE id = %s",
            (args.incident,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"No incident #{args.incident} in Postgres.")
    incident = {
        "id": row[0],
        "title": row[1],
        "description": row[2],
        "root_cause": row[3],
        "resolution": row[4],
        "service": row[5],
        "severity": row[6],
    }

    if args.dry_run:
        # Dry runs only build the prompt; they must not require an LLM key.
        print(build_postmortem_prompt(incident))
        return 0

    doc = generate_postmortem(incident)
    print(doc)
    if args.save:
        with connect_db() as conn:
            save_postmortem(conn, args.incident, doc)
        print(f"\nSaved postmortem for incident #{args.incident} "
              f"({len(doc)} chars).")
    else:
        print(f"\n(Not saved — re-run with --save to write it to "
              f"incident #{args.incident}.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
