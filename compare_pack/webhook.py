"""
Webhook entry point: make the agent reachable as a live service.

The page (app.py) is the first way to trigger the retrieve-and-suggest
pipeline — clicking the button in the UI calls /api/search. This module adds a
second, HTTP-first entry point: POST /webhook/incident, which any caller can
hit with a plain JSON body — a script, a curl command, a monitoring tool, or
the incident simulator used in the demo.

  IncidentIn                 Pydantic schema — the incoming payload
                             (title, description, service, severity) is
                             validated before any code runs, so malformed
                             requests are rejected with 422 without touching
                             the pipeline.

  receive_incident(incident) Takes the validated incident, passes its
                             description straight into the SAME
                             retrieve_and_suggest() pipeline the UI button
                             calls (embed -> Qdrant search -> Postgres fetch
                             -> LLM suggestion), and returns the matched past
                             incidents + suggested fix as a JSON-ready dict.

This does NOT replace retrieve_and_suggest() — it's a second entry point into
it. Nothing is written to Postgres or Qdrant here: the write-back and the
postmortem happen later, via /api/resolve (as before).
"""

from pydantic import BaseModel, ConfigDict, Field

from suggest_fix import retrieve_and_suggest

# Retrieval defaults — the same settings the UI's /api/search route uses.
TOP_K = 5
THRESHOLD = 0.22


class IncidentIn(BaseModel):
    """The validated shape of an incoming webhook incident.

    `description` is the only required field — it is what gets embedded and
    searched. `title`, `service` and `severity` are optional tags that are
    echoed back in the response (and can later be used by the write-back step).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(default=None, description="Short incident title")
    description: str = Field(
        min_length=1, description="What happened — embedded and searched"
    )
    service: str | None = Field(default=None, description="Affected service")
    severity: str | None = Field(default=None, description="critical/high/medium/low")


def receive_incident(incident: IncidentIn) -> dict:
    """Handle a webhook incident: run the shared retrieve-and-suggest pipeline
    on its description and return the matched past incidents + suggested fix.

    Returns a JSON-ready dict:
        {ok, incident: {title, description, service, severity},
         matches: [past-incident dicts (id, score, title, ...)],
         suggestion: str|None, llm_error: str|None, note: str|None}

    Raises SystemExit (with an actionable message) when the memory store is
    missing or empty; LLM failures are returned in `llm_error`, never fatal.
    """
    out = retrieve_and_suggest(incident.description, top=TOP_K, threshold=THRESHOLD)
    return {"ok": True, "incident": incident.model_dump(), **out}
