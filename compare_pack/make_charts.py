"""
Generate the performance-matrix charts from performance_results.json.

Reads  perf_charts/performance_results.json   (written by benchmark_perf.py)
Writes perf_charts/chart_*.png                (7 charts)
Writes perf_charts/report.html                (self-contained dashboard)

Usage:  hackenv/Scripts/python.exe make_charts.py
"""

import base64
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "perf_charts")
DATA = os.path.join(OUT, "performance_results.json")
HISTORY = os.path.join(OUT, "performance_history.json")

# ---- dark theme palette ---------------------------------------------------
BG = "#0e1526"
PANEL = "#151f38"
TEXT = "#e8eefc"
MUTED = "#8fa3c8"
GRID = "#243252"
CYAN = "#00d4ff"
PURPLE = "#7b61ff"
GREEN = "#2ecc71"
AMBER = "#f5a623"
RED = "#ff5c5c"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": TEXT, "font.size": 11,
    "font.family": "DejaVu Sans",
})


def style_ax(ax, ylabel="", title=""):
    ax.set_title(title)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)


# ------------------------------------------------------------------ charts ---
def chart_latency(d):
    """Grouped horizontal bar: mean / p50 / p95 per stage (log-ish scale)."""
    stages = {
        "Gemini embedding": d["latency"]["embedding"],
        "Qdrant query_points": d["latency"]["qdrant"],
        "Postgres enrichment": d["latency"]["postgres"],
        "Full retrieval (1+2+3)": d["latency"]["retrieval_full"],
        "Groq LLM suggestion": d["latency"]["llm_suggestion"],
        "Postmortem gen": d["latency"]["postmortem"],
        "HTTP /api/search (e2e)": d["latency"]["http_e2e"],
        "Full pipeline": d["latency"]["pipeline_full"],
    }
    names = list(stages)
    mean = [stages[k]["mean_ms"] for k in names]
    p50 = [stages[k]["p50_ms"] for k in names]
    p95 = [stages[k]["p95_ms"] for k in names]

    y = np.arange(len(names))
    h = 0.27
    fig, ax = plt.subplots(figsize=(11, 6.4))
    b1 = ax.barh(y + h, mean, h, label="mean", color=CYAN)
    b2 = ax.barh(y, p50, h, label="p50", color=PURPLE)
    b3 = ax.barh(y - h, p95, h, label="p95", color=AMBER, alpha=0.85)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_width() + 30, b.get_y() + b.get_height() / 2,
                    f"{b.get_width():.0f}", va="center", fontsize=8.5,
                    color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace(" ", "\n") if len(s) > 18 else s for s in names],
                       fontsize=10)
    ax.set_xlabel("milliseconds (lower is better)", color=MUTED)
    ax.set_xlim(0, max(p95) * 1.16)
    ax.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, fontsize=10)
    style_ax(ax, title="Latency matrix — every stage of the retrieval + LLM pipeline")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_latency.png"), dpi=150)
    plt.close(fig)


def chart_recall(d):
    """Retrieval quality: Recall@1/3/5 self-retrieval + related-query recall."""
    rq = d["retrieval_quality"]
    tot = rq["total"]
    rl = d["related_queries"]
    labels = ["Recall@1", "Recall@3", "Recall@5",
              "Related\nRecall@1", "Related\nRecall@5"]
    vals = [100 * rq["recall1"] / tot, 100 * rq["recall3"] / tot,
            100 * rq["recall5"] / tot,
            100 * rl["recall1"] / rl["total"], 100 * rl["recall5"] / rl["total"]]
    colors = [PURPLE, CYAN, GREEN, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_ylim(0, 115)
    ax.set_ylabel("recall (%)", color=MUTED)
    style_ax(ax, title="Retrieval quality — how often the right past incident surfaces")
    ax.axhline(100, color=GREEN, ls="--", lw=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_recall.png"), dpi=150)
    plt.close(fig)


def chart_margins(d):
    """Per-incident: self score vs nearest-other score (discrimination)."""
    per = d["retrieval_quality"]["per_query"]
    ids = [str(p["id"]) for p in per]
    self_s = []
    other_s = []
    for p in per:
        others = [s for s in p["top5"] if p["self"] is None or abs(s - p["self"]) > 1e-9]
        self_s.append(p["self"] if p["self"] is not None else p["top1"])
        other_s.append(max(others) if others else 0.0)
    x = np.arange(len(ids))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w / 2, self_s, w, label="self score (correct incident)", color=GREEN)
    ax.bar(x + w / 2, other_s, w, label="nearest other incident", color=MUTED, alpha=0.7)
    ax.axhline(d["thresholds"]["app"], color=AMBER, ls="--", lw=1.2,
               label=f"app threshold ({d['thresholds']['app']})")
    ax.axhline(d["thresholds"]["cli"], color=RED, ls=":", lw=1.2,
               label=f"CLI threshold ({d['thresholds']['cli']})")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=8)
    ax.set_xlabel("incident id", color=MUTED)
    ax.set_ylabel("cosine similarity", color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, fontsize=9)
    style_ax(ax, title="Discrimination — correct incident vs nearest look-alike, per incident")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_margins.png"), dpi=150)
    plt.close(fig)


def chart_scores(d):
    """Score distribution histogram with threshold lines."""
    scores = d["retrieval_quality"]["scores_first"]
    th = d["thresholds"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bins = np.arange(0.6, 1.01, 0.02)
    n, _, patches = ax.hist(scores, bins=bins, color=CYAN, edgecolor=BG,
                            alpha=0.9)
    for p, c in zip(patches, np.linspace(0.25, 1, len(patches))):
        p.set_facecolor(plt.cm.viridis(c))
    ax.axvline(th["app"], color=AMBER, ls="--", lw=2,
               label=f"app threshold {th['app']}  ({th['over_app']}/{len(scores)} above)")
    ax.axvline(th["cli"], color=RED, ls=":", lw=2,
               label=f"CLI threshold {th['cli']}  ({th['over_cli']}/{len(scores)} above)")
    ax.set_xlabel("top-1 cosine similarity score", color=MUTED)
    ax.set_ylabel("queries", color=MUTED)
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=9)
    style_ax(ax, title=f"Score distribution — mean {th['scores_mean']:.3f}, "
                       f"min {th['scores_min']:.3f}, max {th['scores_max']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_scores.png"), dpi=150)
    plt.close(fig)


def chart_related(d):
    """Related-query: top-1 score per realistic query, pass/fail colour."""
    det = d["related_queries"]["details"]
    keys = [x["key"] for x in det]
    top1 = [x["top1"] for x in det]
    colors = [GREEN if x["ok"] else RED for x in det]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(keys, top1, color=colors, width=0.62)
    for b, x, v in zip(bars, det, top1):
        rank = x["rank"]
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f"rank {rank}" if rank else "MISS", ha="center", fontsize=9,
                fontweight="bold", color=TEXT)
    ax.axhline(d["thresholds"]["app"], color=AMBER, ls="--", lw=1.2,
               label=f"app threshold {d['thresholds']['app']}")
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("top-1 similarity", color=MUTED)
    ax.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, fontsize=9)
    style_ax(ax, title="Related-query recall — realistic 'new incident' queries find their seed")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_related.png"), dpi=150)
    plt.close(fig)


def chart_breakdown(d):
    """Share of the full-pipeline latency budget per stage (donut)."""
    lat = d["latency"]
    parts = [
        ("Postgres enrichment", lat["postgres"]["mean_ms"]),
        ("Gemini embedding", lat["embedding"]["mean_ms"]),
        ("Groq LLM", lat["llm_suggestion"]["mean_ms"]),
        ("Qdrant search", lat["qdrant"]["mean_ms"]),
    ]
    labels = [p[0] for p in parts]
    sizes = [p[1] for p in parts]
    colors = [RED, AMBER, PURPLE, CYAN]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                       counterclock=False, autopct="%.0f%%", pctdistance=0.78,
                       wedgeprops=dict(width=0.42, edgecolor=BG, linewidth=2),
                       textprops=dict(color=TEXT, fontsize=11, weight="bold"))
    total = sum(sizes)
    ax.text(0, 0.08, f"{total:.0f} ms", ha="center", fontsize=22,
            fontweight="bold", color=TEXT)
    ax.text(0, -0.16, "retrieval + LLM", ha="center", fontsize=10, color=MUTED)
    style_ax(ax, title="Where the time goes — full /api/search latency budget")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_breakdown.png"), dpi=150)
    plt.close(fig)


def chart_throughput(d):
    """Throughput: pipeline/min vs retrieval-only/min vs Groq free limit."""
    tp = d["throughput"]
    labels = ["Full pipeline\n(retrieve + LLM)", "Retrieval-only\n(no LLM)",
              "Groq free limit\n(30 req/min)"]
    vals = [tp["pipeline_per_min"], tp["retrieval_per_min"], 30.0]
    colors = [PURPLE, CYAN, MUTED]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("calls per minute", color=MUTED)
    style_ax(ax, title=f"Throughput (single worker) — headroom {tp['groq_headroom']:.1f} req/min")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_throughput.png"), dpi=150)
    plt.close(fig)


def chart_trend(history):
    """Two-panel trend across benchmark runs: retrieval quality (left) and
    pipeline speed (right). One point per run in performance_history.json."""
    runs = [h["run"] for h in history]
    labels = [f"#{h['run']}\n{h['generated_at'][11:16]}" for h in history]
    recall5 = [100.0 * h["quality"]["recall5"] / h["quality"]["total"]
               for h in history]
    mrr = [h["quality"]["mrr"] for h in history]
    pipe_ms = [h["latency_ms"]["pipeline_full"] for h in history]
    thru = [h["throughput"]["pipeline_per_min"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left panel — quality over time
    l1, = ax1.plot(runs, recall5, "o-", color=GREEN, lw=2.2, ms=8,
                   label="Recall@5 (%)")
    l2, = ax1.plot(runs, [m * 100 for m in mrr], "s--", color=CYAN, lw=2,
                   ms=7, label="MRR × 100")
    for x, v in zip(runs, recall5):
        ax1.annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10,
                     fontweight="bold", color=GREEN)
    for x, v in zip(runs, [m * 100 for m in mrr]):
        ax1.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9, color=CYAN)
    ax1.set_xticks(runs)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(55, 110)
    ax1.set_ylabel("recall (%) / MRR ×100", color=MUTED)
    ax1.legend(handles=[l1, l2], loc="lower right", facecolor=PANEL,
               edgecolor=GRID, fontsize=9)
    style_ax(ax1, title="Retrieval quality across runs")

    # Right panel — speed over time (pipeline latency + throughput, twin axis)
    l3, = ax2.plot(runs, pipe_ms, "o-", color=PURPLE, lw=2.2, ms=8,
                   label="full pipeline (ms)")
    ax2.set_ylabel("milliseconds", color=MUTED)
    ax2b = ax2.twinx()
    l4, = ax2b.plot(runs, thru, "^--", color=AMBER, lw=2, ms=8,
                    label="throughput (/min)")
    ax2b.set_ylabel("incidents per minute", color=AMBER)
    ax2b.tick_params(colors=AMBER)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_color(GRID)
    for x, v in zip(runs, pipe_ms):
        ax2.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10,
                     fontweight="bold", color=PURPLE)
    for x, v in zip(runs, thru):
        ax2b.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                      xytext=(0, -16), ha="center", fontsize=9, color=AMBER)
    ax2.set_xticks(runs)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(handles=[l3, l4], loc="lower right", facecolor=PANEL,
               edgecolor=GRID, fontsize=9)
    style_ax(ax2, title="Pipeline speed across runs")

    if len(history) < 2:
        fig.suptitle("Trend needs 2+ benchmark runs — run benchmark_perf.py again "
                     "to extend this line", fontsize=11, color=AMBER)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_trend.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report ---
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(d):
    rq = d["retrieval_quality"]
    rl = d["related_queries"]
    lat = d["latency"]
    tp = d["throughput"]
    sys = d["system"]
    th = d["thresholds"]
    tot = rq["total"]

    cards = [
        ("Recall@5", f"{rq['recall5']}/{tot}  (100%)", GREEN),
        ("MRR@5", f"{rq['mrr']:.4f}", CYAN),
        ("Full pipeline", f"{lat['pipeline_full']['mean_ms']:.0f} ms/call", PURPLE),
        ("Throughput", f"{tp['pipeline_per_min']:.1f} incidents/min", AMBER),
        ("Related Recall@5", f"{rl['recall5']}/{rl['total']}", GREEN),
        ("Stores in sync", f"{sys['pg_active']} PG = {sys['qdrant']} Qdrant", CYAN),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="card-val" style="color:{c}">{v}</div>'
        f'<div class="card-label">{k}</div></div>' for k, v, c in cards
    )

    def img_card(path, caption):
        return (
            f'<figure><img src="data:image/png;base64,{b64(os.path.join(OUT, path))}" '
            f'alt="{caption}"/><figcaption>{caption}</figcaption></figure>'
        )

    charts = "".join([
        img_card("chart_trend.png", "Performance trend across benchmark runs — quality and speed"),
        img_card("chart_latency.png", "Latency matrix — mean / p50 / p95 per stage"),
        img_card("chart_recall.png", "Retrieval quality — self-retrieval and related queries"),
        img_card("chart_margins.png", "Discrimination — correct incident vs nearest look-alike"),
        img_card("chart_scores.png", "Similarity score distribution vs thresholds"),
        img_card("chart_related.png", "Related-query recall — each query's top match"),
        img_card("chart_breakdown.png", "Latency budget — where /api/search time goes"),
        img_card("chart_throughput.png", "Throughput vs Groq free-tier limit"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Incident Memory — Performance Matrix</title>
<style>
  :root {{ --bg:#0e1526; --panel:#151f38; --line:#243252; --text:#e8eefc; --muted:#8fa3c8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text);
         font-family:"Segoe UI", system-ui, sans-serif; padding:32px 40px 60px; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-end;
            border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:26px; }}
  h1 {{ font-size:26px; letter-spacing:.3px; }}
  h1 span {{ color:var(--muted); font-weight:400; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
           gap:14px; margin-bottom:30px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
           padding:18px 16px; text-align:center; }}
  .card-val {{ font-size:22px; font-weight:700; }}
  .card-label {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  figure {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
            padding:14px; margin:0 0 22px; }}
  img {{ width:100%; height:auto; border-radius:8px; display:block; }}
  figcaption {{ color:var(--muted); font-size:12px; padding:10px 4px 2px; }}
  .pill {{ display:inline-block; background:var(--panel); border:1px solid var(--line);
           border-radius:999px; padding:4px 12px; font-size:12px; color:var(--muted);
           margin-right:6px; }}
</style>
</head>
<body>
  <header>
    <div>
      <h1>Incident Memory — <span>Performance Matrix</span></h1>
      <div class="sub">Qdrant vector search &middot; Gemini embeddings &middot; Groq LLM
        &middot; measured {d['generated_at']}</div>
    </div>
    <div>
      <span class="pill">{sys['pg_active']} active incidents</span>
      <span class="pill">{sys['growth']} from memory loop</span>
      <span class="pill">threshold {th['app']}</span>
    </div>
  </header>
  <div class="grid">{cards_html}</div>
  {charts}
</body>
</html>"""


# ------------------------------------------------------------------ main ---
def main():
    if not os.path.exists(DATA):
        sys.exit(f"Run benchmark_perf.py first (no {DATA}).")
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)

    os.makedirs(OUT, exist_ok=True)

    history = []
    if os.path.exists(HISTORY):
        try:
            with open(HISTORY, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    chart_trend(history)
    chart_latency(d)
    chart_recall(d)
    chart_margins(d)
    chart_scores(d)
    chart_related(d)
    chart_breakdown(d)
    chart_throughput(d)

    html = build_html(d)
    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("Charts written to:", OUT)
    for name in ("chart_trend.png", "chart_latency.png", "chart_recall.png",
                 "chart_margins.png", "chart_scores.png", "chart_related.png",
                 "chart_breakdown.png", "chart_throughput.png", "report.html"):
        size = os.path.getsize(os.path.join(OUT, name))
        print(f"  {name:<24} {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
