"""
Evaluates sonata.rag.ask() on 10 fixed queries, 3 runs each.

Per query it checks:
  (a) grounding    every recommended song was in that run's retrieved set
  (b) consistency  mean pairwise overlap (Jaccard) of the recommended songs across runs >= 60%
  (c) graceful     the query was handled as expected without an exception:
                   normal requests get 3-5 songs (a fallback must explain itself),
                   bad input (empty / too long) is rejected with a message and no songs,
                   nonsense gets a weak-match notice instead of a model answer
It writes reports/eval_results.md, then runs every query with use_audio=False and
True and writes the before/after comparison to reports/rag_comparison.md.

The provider comes from .env (LLM_PROVIDER, mock by default). With Gemini, runs 2-3
are served from the response cache unless --fresh is given, which makes the
consistency check meaningful but costs a real call per run; the script refuses to
start if the expected calls exceed what is left of today's limit.

Run from the project root with: python -m sonata.evaluate [--fresh]
"""

import argparse
import datetime as dt
import itertools
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from sonata import rag
from sonata.features import FEATURES_PATH
from sonata.catalog import LIBRARY_PATH, load_library

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_REPORT = REPORTS_DIR / "eval_results.md"
COMPARISON_REPORT = REPORTS_DIR / "rag_comparison.md"

RUNS = 3
CONSISTENCY_THRESHOLD = 0.6
MODEL_SOURCES = {"gemini", "cache", "mock"}

# (query, kind): kind is "normal", "bad_input" or "nonsense"
TEST_QUERIES = [
    ("high energy reggaeton for a party", "normal"),
    ("chill songs to study to", "normal"),
    ("sad slow songs for a rainy night", "normal"),
    ("dark rap for the gym", "normal"),
    ("songs like Bad Bunny", "normal"),
    ("bright happy pop", "normal"),
    ("music for a late night drive", "normal"),
    ("", "bad_input"),
    ("asdkjh qwpoeiru zzxv", "nonsense"),
    ("play something " * 40, "bad_input"),     # 599 characters once trimmed
]

log = logging.getLogger("evaluate")

AskFn = Callable[..., Dict]


# ---------- checks (pure functions, unit-tested) ----------

def returned_ids(result: Dict) -> List[int]:
    return [pick["song"]["deezer_id"] for pick in result.get("recommendations", [])]


def is_grounded(result: Dict) -> bool:
    """Every recommended song is in the retrieved set (trivially true when nothing is returned)."""
    return set(returned_ids(result)) <= set(result.get("retrieved_ids", []))


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def consistency(runs: List[Dict]) -> float:
    """Mean pairwise Jaccard overlap of the recommended songs across runs (1.0 for a single run)."""
    sets = [set(returned_ids(run)) for run in runs]
    pairs = list(itertools.combinations(sets, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else 1.0


def handled_gracefully(result: Optional[Dict], kind: str) -> bool:
    """Whether one run did what this kind of query should do. None means ask() raised."""
    if result is None:
        return False
    picks, source, message = result.get("recommendations", []), result.get("source"), result.get("message")
    if kind == "bad_input":
        return source == "rejected" and bool(message) and not picks
    if kind == "nonsense":
        return source in {"fallback", "rejected"} and bool(message)
    # normal: a usable answer; a fallback is fine as long as it says why. With fewer than 3
    # strong matches, a shorter answer is right as long as it says there's no strong match.
    few_strong = result.get("strong_matches", rag.MIN_PICKS) < rag.MIN_PICKS
    fewest = rag.MIN_PICKS_FEW_STRONG if few_strong else rag.MIN_PICKS
    return (fewest <= len(picks) <= rag.MAX_PICKS
            and (not few_strong or bool(message))
            and (source in MODEL_SOURCES or (source == "fallback" and bool(message))))


def average_similarity(results: List[Dict]) -> Optional[float]:
    values = [pick["retrieval_similarity"] for result in results for pick in result.get("recommendations", [])]
    return sum(values) / len(values) if values else None


# ---------- running ----------

def safe_ask(ask: AskFn, query: str, **kwargs) -> Dict:
    """Calls ask(); an exception is recorded instead of stopping the evaluation."""
    try:
        return ask(query, **kwargs)
    except Exception as error:  # the point of check (c) is to catch these
        log.error("ask(%r) raised %s: %s", query[:40], type(error).__name__, error)
        return {"query": query, "error": f"{type(error).__name__}: {error}", "recommendations": [],
                "retrieved_ids": [], "dropped": [], "source": "error", "message": None}


def evaluate(ask: AskFn, queries=TEST_QUERIES, runs: int = RUNS, fresh: bool = False, **ask_kwargs) -> List[Dict]:
    """Runs each query `runs` times and returns one row of verdicts per query."""
    rows = []
    for query, kind in queries:
        results = [safe_ask(ask, query, use_cache=(run == 0 or not fresh), **ask_kwargs) for run in range(runs)]
        crashed = [r for r in results if r.get("source") == "error"]
        rows.append({
            "query": query, "kind": kind, "results": results,
            "grounded": all(is_grounded(r) for r in results),
            "consistency": consistency(results),
            "consistent": consistency(results) >= CONSISTENCY_THRESHOLD,
            "graceful": all(handled_gracefully(None if r in crashed else r, kind) for r in results),
            "avg_similarity": average_similarity(results),
            "sources": sorted({r.get("source") for r in results}),
            "hallucinations": sum(1 for r in results for d in r.get("dropped", []) if d["why"] == "hallucination"),
            "messages": sorted({r["message"] for r in results if r.get("message")}),
            "errors": [r["error"] for r in crashed],
        })
    return rows


def compare_audio(ask: AskFn, audio_map: Dict[int, float], queries=TEST_QUERIES, **ask_kwargs) -> List[Dict]:
    """Runs each query once with use_audio=False and once with True."""
    rows = []
    for query, kind in queries:
        before = safe_ask(ask, query, use_audio=False, **ask_kwargs)
        after = safe_ask(ask, query, use_audio=True, **ask_kwargs)

        def summary(result):
            picks = result.get("recommendations", [])
            audio = [audio_map.get(p["song"]["deezer_id"]) for p in picks]
            audio = [a for a in audio if a is not None]
            return {"result": result, "picks": picks, "avg_similarity": average_similarity([result]),
                    "avg_audio_match": sum(audio) / len(audio) if audio else None,
                    "liked": sum(p["song"]["liked"] for p in picks)}

        rows.append({"query": query, "kind": kind, "before": summary(before), "after": summary(after),
                     "overlap": jaccard(set(returned_ids(before)), set(returned_ids(after)))})
    return rows


# ---------- reports ----------

def short(query: str, width: int = 38) -> str:
    if not query.strip():
        return "*(empty)*"
    query = query.strip().replace("|", "\\|")
    return query if len(query) <= width else f"{query[:width - 1]}… ({len(query)} chars)"


def fmt(value: Optional[float], pattern: str = "{:.2f}") -> str:
    return "—" if value is None else pattern.format(value)


def mark(ok: bool) -> str:
    return "✅ pass" if ok else "❌ fail"


def settings_line() -> str:
    settings = rag._settings()
    model = f", model `{settings['model']}`" if settings["provider"] == "gemini" and settings["model"] else ""
    return f"provider `{settings['provider']}`{model}"


def eval_markdown(rows: List[Dict], fresh: bool, runs: int = RUNS) -> str:
    deterministic = all(set(row["sources"]) <= {"mock", "cache", "fallback", "rejected"} for row in rows)
    lines = [
        "# Sonata RAG evaluation",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} with {settings_line()}; "
        f"{len(rows)} queries × {runs} runs{' (fresh model call per run)' if fresh else ''}.",
        "",
        "| # | Query | Kind | Grounding | Consistency | Graceful | Avg retrieval similarity | Sources |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for number, row in enumerate(rows, start=1):
        lines.append(
            f"| {number} | {short(row['query'])} | {row['kind']} | {mark(row['grounded'])} | "
            f"{mark(row['consistent'])} ({row['consistency']:.0%}) | {mark(row['graceful'])} | "
            f"{fmt(row['avg_similarity'])} | {', '.join(row['sources'])} |")

    similarities = [row["avg_similarity"] for row in rows if row["avg_similarity"] is not None]
    total = len(rows)
    lines += [
        f"| | **Total** | | **{sum(r['grounded'] for r in rows)}/{total}** | "
        f"**{sum(r['consistent'] for r in rows)}/{total}** | **{sum(r['graceful'] for r in rows)}/{total}** | "
        f"**{fmt(sum(similarities) / len(similarities) if similarities else None)}** | |",
        "",
        f"Hallucinated songs dropped by the validator: {sum(r['hallucinations'] for r in rows)}. "
        f"Exceptions: {sum(len(r['errors']) for r in rows)}.",
        "",
        "**Checks.** Grounding: every recommended song was in that run's retrieved set. "
        f"Consistency: mean pairwise overlap of the recommended songs across the {runs} runs is at least "
        f"{CONSISTENCY_THRESHOLD:.0%}. Graceful: normal requests get {rag.MIN_PICKS}-{rag.MAX_PICKS} songs, "
        f"or {rag.MIN_PICKS_FEW_STRONG}-{rag.MAX_PICKS} with a no-strong-match note when fewer than "
        f"{rag.MIN_PICKS} songs match strongly (a fallback must say why), empty or over-500-character "
        "input is rejected with a message, and nonsense gets a weak-match notice instead of a model answer. "
        "Retrieval similarity is each recommended song's retrieval score (0-1): how closely it matched "
        "the request, not whether the pick is right.",
    ]
    if deterministic:
        lines += ["", "> **Note:** every run came from the mock provider, the response cache, a fallback or a "
                      "rejection, which are all deterministic, so the consistency column is 100% by "
                      "construction here. Run `python -m sonata evaluate --fresh` with Gemini to measure "
                      "the model's real run-to-run consistency."]
    notes = [(number, message) for number, row in enumerate(rows, start=1) for message in row["messages"]]
    notes += [(number, f"Exception: {error}") for number, row in enumerate(rows, start=1) for error in row["errors"]]
    if notes:
        lines += ["", "## Messages shown to the user", ""]
        lines += [f"- **#{number}:** {message}" for number, message in notes]
    return "\n".join(lines) + "\n"


def comparison_markdown(rows: List[Dict]) -> str:
    def picks_cell(side):
        if not side["picks"]:
            return f"*{side['result'].get('source')}*"
        return "<br>".join(f"{p['song']['title']} — {p['song']['artist']}{' ♥' if p['song']['liked'] else ''}"
                           for p in side["picks"]).replace("|", "\\|")

    lines = [
        "# Text-only vs. text + audio retrieval",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} with {settings_line()}. **Before:** `use_audio=False`, "
        f"songs ranked by text similarity only. **After:** `use_audio=True`, the top {rag.RERANK_POOL} text matches "
        f"re-ranked by {rag.TEXT_WEIGHT:.0%} text + {rag.AUDIO_WEIGHT:.0%} audio similarity to your closest liked song.",
        "",
        "*Audio match* is the recommended songs' average audio similarity (cosine on standardized features) "
        "to their closest liked song, computed the same way for both columns; ♥ marks songs you already like.",
        "",
        "| # | Query | Before: picks | After: picks | Overlap | Retrieval similarity before → after | "
        "Audio match before → after | ♥ before → after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for number, row in enumerate(rows, start=1):
        b, a = row["before"], row["after"]
        lines.append(
            f"| {number} | {short(row['query'], 30)} | {picks_cell(b)} | {picks_cell(a)} | {row['overlap']:.0%} | "
            f"{fmt(b['avg_similarity'])} → {fmt(a['avg_similarity'])} | "
            f"{fmt(b['avg_audio_match'])} → {fmt(a['avg_audio_match'])} | {b['liked']} → {a['liked']} |")

    scored = [row for row in rows if row["before"]["picks"] and row["after"]["picks"]]

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    if scored:
        lines += [
            "",
            f"**Averages over the {len(scored)} queries that returned songs:** overlap "
            f"{fmt(mean([r['overlap'] for r in scored]), '{:.0%}')}, retrieval similarity "
            f"{fmt(mean([r['before']['avg_similarity'] for r in scored]))} → "
            f"{fmt(mean([r['after']['avg_similarity'] for r in scored]))}, audio match "
            f"{fmt(mean([r['before']['avg_audio_match'] for r in scored]))} → "
            f"{fmt(mean([r['after']['avg_audio_match'] for r in scored]))}, liked songs per answer "
            f"{fmt(mean([r['before']['liked'] for r in scored]), '{:.1f}')} → "
            f"{fmt(mean([r['after']['liked'] for r in scored]), '{:.1f}')}.",
            "",
            "Retrieval similarity is each column's own ranking score (text similarity before, the "
            f"{rag.TEXT_WEIGHT:.0%}/{rag.AUDIO_WEIGHT:.0%} blend after), so the two numbers are not on "
            "exactly the same scale.",
        ]
    return "\n".join(lines) + "\n"


# ---------- quota preflight ----------

def expected_calls(fresh: bool, queries=TEST_QUERIES, runs: int = RUNS) -> int:
    """Upper bound on real Gemini calls: normal queries only (bad input and nonsense never reach the model)."""
    normal = sum(kind == "normal" for _, kind in queries)
    evaluation = normal * (runs if fresh else 1)
    comparison = normal  # the use_audio=False runs; the use_audio=True runs hit the cache
    return evaluation + comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sonata's RAG recommendations")
    parser.add_argument("--fresh", action="store_true",
                        help="bypass the response cache for every run (real consistency test; costs API calls)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    print(run(fresh=args.fresh))


def run(fresh: bool = False) -> str:
    """Runs the evaluation and comparison, saves both reports, and returns the printable summary.

    Raises SystemExit with a clear message if Gemini's remaining daily calls aren't enough.
    """
    lines = []
    settings = rag._settings()
    if settings["provider"] == "gemini":
        needed, left = expected_calls(fresh), rag.MAX_CALLS_PER_DAY - rag.calls_today()
        lines.append(f"Gemini: up to {needed} calls needed (fewer if cached), {left} left today.")
        if needed > left:
            raise SystemExit(f"Not starting: up to {needed} Gemini calls could be needed but only {left} are "
                             f"left today, so later runs would silently turn into fallbacks. Try again "
                             f"tomorrow, drop --fresh, or set LLM_PROVIDER=mock.")

    library = load_library(LIBRARY_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    ask_kwargs = {"library": library, "features": features}

    started = time.perf_counter()
    rows = evaluate(rag.ask, fresh=fresh, **ask_kwargs)
    report = eval_markdown(rows, fresh)
    comparison = comparison_markdown(compare_audio(rag.ask, rag.audio_similarity(features, library), **ask_kwargs))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(report, encoding="utf-8")
    COMPARISON_REPORT.write_text(comparison, encoding="utf-8")
    lines += [report, f"Saved {EVAL_REPORT.relative_to(PROJECT_ROOT)} and "
                      f"{COMPARISON_REPORT.relative_to(PROJECT_ROOT)} ({time.perf_counter() - started:.1f}s)"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
