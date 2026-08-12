Incident Memory — Performance Comparison Pack
=============================================

This folder contains everything needed to run the SAME performance benchmark
and dashboard on another machine, so you can compare the parameters
(retrieval quality, latency matrix, throughput, thresholds) across systems.

SETUP ON THE NEW SYSTEM (Windows, Git Bash):
1. Copy this whole folder into your project, then create the venv:
     python -m venv hackenv
     hackenv/Scripts/python.exe -m pip install flask pydantic "psycopg[binary]" qdrant-client python-dotenv google-genai groq openai matplotlib numpy

2. Create a .env file (see .env.example). Use the NEW system's Postgres and
   Qdrant endpoints and your own API keys. NEVER share .env files — they
   contain secrets.

3. Seed the same memory (12 incidents):
     hackenv/Scripts/python.exe new_ingest.py
   (validate setup first:  hackenv/Scripts/python.exe new_ingest.py --check)

4. Start the app (port 5050 — the benchmark's e2e test needs it):
     hackenv/Scripts/python.exe app.py --port 5050

5. Run the benchmark in a second terminal:
     hackenv/Scripts/python.exe benchmark_perf.py

6. Open the dashboard:
     http://localhost:5050/perf

WHY THE TREND CHART COMPARES BOTH SYSTEMS:
perf_charts/performance_history.json is append-only. The runs from the
original machine are already inside it, so the new machine's runs append
after them — the trend chart on /perf plots both machines side by side.

TIP for an apples-to-apples retrieval-quality comparison: both systems must
search the same memory. Seeding with the same seed_incidents.json gives the
same 12 incidents. If you also RESOLVED incidents on the original machine,
copy incidents_backup.json over and restore it (Import button on the page, or
`python memory_backup.py import incidents_backup.json`) so both stores match.

NOT COPIED (recreate / regenerate locally):
  hackenv/            (venv — recreated in step 1)
  .env                (secrets — recreate from .env.example)
  server.log
  perf_charts/*.png, perf_charts/report.html   (regenerate with make_charts.py)
  incidents_backup.json  (optional, huge — only needed for exact memory state)
