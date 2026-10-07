"""Track C — efficiency: latency, documents scored, and what the speed-ups cost.

Member 2 (Suvrat).  Everything is timed through ``nyaya.search.search()``, the
same entry point the CLI and the UI use, so the latencies are the system's and
not a benchmark harness's.

What is measured, and why each number is here rather than a nicer-looking one:

* **p50 and p95, not the mean.**  Query latency is long-tailed — a query whose
  terms are all common scores ten times as many documents as one with a rare term
  — and a mean hides exactly the queries a user would complain about.
* **Cold versus warm.**  The first query in a process pays for loading the
  dictionary (16 MB of JSON), memory-mapping ``postings.bin`` and reading
  ``authority.json`` (9 MB).  Reporting only warm latency would be flattering and
  useless; reporting only cold would misrepresent steady state.  Both are here,
  and cold is split into store-load time and first-query time.
* **Documents scored**, from ``SearchReport.scored``.  Latency depends on the
  machine; documents scored is the algorithmic quantity, and it is what index
  elimination and the tiered index actually change.
* **Tier fidelity is folded in, not recomputed.**
  ``nyaya.index.tiers.evaluate`` already measured it into
  ``index/tiers_eval.json``; measuring it again with slightly different code
  would be a second implementation of the same number.

CLI::

    python -m nyaya.eval.efficiency run [--repeats 3]
    python -m nyaya.eval.efficiency cold
    python -m nyaya.eval.efficiency report
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.utils import ensure_utf8_stdout, human, timed

RUNS = config.RUNS
RESULTS_JSON = RUNS / "efficiency.json"
LATENCY_CSV = RUNS / "efficiency_latency.csv"
PERQUERY_CSV = RUNS / "efficiency_per_query.csv"
STORES_CSV = RUNS / "efficiency_stores.csv"

#: The query set.  Reused from the S5 evaluation so the efficiency and fidelity
#: tables describe the same queries — comparing a speed-up measured on one set
#: against a fidelity measured on another would be meaningless.
from nyaya.index.tiers import DEFAULT_QUERIES  # noqa: E402

MODES = ("B0", "S1", "S2", "S2p", "S3", "S4")

#: Repeats per query, to take a median rather than one noisy sample.  Three is
#: enough for a median and keeps the whole run under a minute per mode.
REPEATS = 3


# ==========================================================================
# 1. Percentiles
# ==========================================================================


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile.  ``q`` in [0, 100]."""
    v = sorted(float(x) for x in values)
    if not v:
        return 0.0
    if len(v) == 1:
        return v[0]
    return float(np.percentile(np.asarray(v), q))


def summarise(values: Sequence[float]) -> Dict[str, float]:
    v = [float(x) for x in values]
    if not v:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": len(v),
        "p50": percentile(v, 50),
        "p95": percentile(v, 95),
        "mean": float(statistics.fmean(v)),
        "min": min(v),
        "max": max(v),
    }


# ==========================================================================
# 2. Warm latency per mode
# ==========================================================================


def measure_mode(mode: str, queries: Sequence[str], repeats: int = REPEATS,
                 k: int = config.TOP_K, verbose: bool = True
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Time every query ``repeats`` times in one mode, warm."""
    from nyaya.search import SearchReport, search, load_context

    ctx = load_context(mode)              # warm the stores before timing
    search("warmup query", k=k, mode=mode, explain=False, ctx=ctx)

    per_query: List[Dict[str, Any]] = []
    for raw in queries:
        samples: List[float] = []
        rep = SearchReport()
        for _ in range(repeats):
            rep = SearchReport()
            t0 = time.perf_counter()
            search(raw, k=k, mode=mode, explain=False, ctx=ctx, report=rep)
            samples.append((time.perf_counter() - t0) * 1000.0)
        per_query.append({
            "mode": mode, "query": raw,
            "ms_median": percentile(samples, 50),
            "ms_min": min(samples), "ms_max": max(samples),
            "candidates": rep.candidates, "scored": rep.scored,
            "returned": rep.returned, "path": rep.path,
            "parse_ms": round(rep.parse_ms, 3),
            "plan_ms": round(rep.plan_ms, 3),
            "score_ms": round(rep.score_ms, 3),
            "scorer": rep.scorer,
        })
        if verbose:
            print(f"  {mode} {raw[:34]:<36} "
                  f"{per_query[-1]['ms_median']:>7.1f} ms  "
                  f"scored {rep.scored:,}", file=sys.stderr)

    lat = [r["ms_median"] for r in per_query]
    scored = [r["scored"] for r in per_query]
    cands = [r["candidates"] for r in per_query]
    entry = {
        "mode": mode,
        "queries": len(per_query),
        "repeats": repeats,
        "latency_ms": summarise(lat),
        "scored": summarise(scored),
        "candidates": summarise(cands),
        "total_scored": int(sum(scored)),
        "scorer": per_query[0]["scorer"] if per_query else "",
    }
    return entry, per_query


# ==========================================================================
# 3. Cold start — measured in a fresh process, which is the only honest way
# ==========================================================================

_COLD_SCRIPT = r"""
import json, time, sys
t0 = time.perf_counter()
from nyaya.search import load_context, search, SearchReport
import_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
ctx = load_context(MODE)
load_ms = (time.perf_counter() - t0) * 1000.0

rep = SearchReport()
t0 = time.perf_counter()
search(QUERY, k=10, mode=MODE, explain=False, ctx=ctx, report=rep)
first_ms = (time.perf_counter() - t0) * 1000.0

rep2 = SearchReport()
t0 = time.perf_counter()
search(QUERY, k=10, mode=MODE, explain=False, ctx=ctx, report=rep2)
second_ms = (time.perf_counter() - t0) * 1000.0

print("COLDJSON" + json.dumps({
    "import_ms": import_ms, "load_ms": load_ms,
    "first_query_ms": first_ms, "second_query_ms": second_ms,
    "scored": rep.scored,
}))
"""


def measure_cold(mode: str = "S4", query: str = "right to privacy",
                 verbose: bool = True) -> Dict[str, Any]:
    """Cold start in a **subprocess**, so nothing is already cached.

    Timing a "cold" start inside a process that has already imported numpy and
    read the dictionary would measure nothing at all; the OS page cache still
    helps the second run, which is noted rather than pretended away.
    """
    script = (f"MODE = {mode!r}\nQUERY = {query!r}\n" + _COLD_SCRIPT)
    proc = subprocess.run([sys.executable, "-c", script],
                          cwd=str(config.ROOT), capture_output=True, text=True)
    out = {"mode": mode, "query": query}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("COLDJSON"):
            out.update(json.loads(line[len("COLDJSON"):]))
            break
    else:
        out["error"] = (proc.stderr or "no COLDJSON line")[-400:]
    if verbose and "error" not in out:
        print(f"  cold {mode}: import {out['import_ms']:.0f} ms + load "
              f"{out['load_ms']:.0f} ms + first query "
              f"{out['first_query_ms']:.0f} ms (warm again "
              f"{out['second_query_ms']:.0f} ms)", file=sys.stderr)
    out["note"] = ("measured in a fresh interpreter; the OS page cache is still "
                   "warm from previous runs, so this is a lower bound on a "
                   "genuinely cold machine")
    return out


# ==========================================================================
# 4. Store sizes and build times
# ==========================================================================

#: Which file belongs to which store, for the report's size table.
STORE_FILES: Dict[str, Sequence[str]] = {
    "S1 paragraphs": ("paragraphs.jsonl", "paragraphs.idx.json",
                      "paragraphs.off.npz"),
    "S2 parametric": ("parametric.json", "parametric_year.npy",
                      "parametric_year_ord.npy", "parametric_bench.npy",
                      "parametric_bench_ord.npy"),
    "S3 positional": ("dictionary.json", "postings.bin", "docmeta.npz"),
    "S3 baseline": ("baseline/dictionary.json", "baseline/postings.bin",
                    "baseline/docmeta.npz"),
    "S4 citations": ("citations.jsonl", "citations_stats.json"),
    "S4 authority": ("authority.json", "authority_treatment.json",
                     "pagerank_stats.json"),
    "S5 champions": ("champions.json", "tiers.npz"),
}


def store_sizes() -> Dict[str, Any]:
    out: Dict[str, Any] = {"stores": {}, "total_bytes": 0}
    for store, names in STORE_FILES.items():
        files: Dict[str, int] = {}
        for name in names:
            p = config.INDEX / name
            if p.exists():
                files[name] = int(p.stat().st_size)
        total = sum(files.values())
        out["stores"][store] = {"bytes": total, "human": human(total),
                               "files": files}
        out["total_bytes"] += total
    out["total_human"] = human(out["total_bytes"])
    # The corpus the index was built from, for context.
    for label, p in (("judgments.jsonl", config.JUDGMENTS_JSONL),
                     ("universe.parquet", config.UNIVERSE_PARQUET),
                     ("manifest.parquet", config.MANIFEST_PARQUET)):
        if Path(p).exists():
            out.setdefault("corpus", {})[label] = int(Path(p).stat().st_size)
    return out


def build_times() -> Dict[str, Any]:
    """Build times recorded by the builders themselves, where they saved them."""
    out: Dict[str, Any] = {}
    candidates = {
        "S2 parametric": ("parametric.json", ("header", "build_seconds")),
        "S4 citations": ("citations_stats.json", ("build_seconds",)),
        "S4 pagerank": ("pagerank_stats.json", ("iterations",)),
        "S5 champions": ("champions.json", ("stats", "r")),
    }
    for label, (name, _path) in candidates.items():
        p = config.INDEX / name
        if p.exists():
            out[label] = {"artefact": name,
                          "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(p.stat().st_mtime))}
    # Measured directly where it is cheap enough to just re-run.
    try:
        from nyaya.index import tiers as T
        t0 = time.perf_counter()
        T.TieredIndex.load()
        out["S5 load_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    except Exception:
        pass
    return out


# ==========================================================================
# 5. Tier fidelity — folded in from the S5 evaluation
# ==========================================================================


def tier_fidelity() -> Dict[str, Any]:
    """Read ``index/tiers_eval.json`` rather than recomputing it."""
    p = config.INDEX / "tiers_eval.json"
    if not p.exists():
        return {"available": False,
                "note": "run `python -m nyaya.index.tiers evaluate` first"}
    s = json.loads(p.read_text(encoding="utf-8"))
    worst = min(s.get("rows", []), key=lambda r: r["overlap"], default=None)
    return {
        "available": True,
        "k": s["k"], "r": s["r"], "queries": s["queries"],
        "mean_scored_exhaustive": s["mean_scored_exhaustive"],
        "mean_scored_tier1": s["mean_scored_tier1"],
        "mean_reduction": s["mean_reduction"],
        "mean_speedup": s["mean_speedup"],
        "identical_topk": s["identical_topk"],
        "identical_fraction": s["identical_fraction"],
        "mean_recall_at_k": s["mean_recall_at_k"],
        "top1_same": s["top1_same"],
        "fell_through": s["fell_through"],
        "worst_query": ({"query": worst["query"], "overlap": worst["overlap"],
                         "reduction": worst["reduction"]} if worst else None),
        "note": ("S5 is now wired into the live free-text path "
                 "(search() -> tiers.free_text_candidates), so the by_mode "
                 "latency table above measures the tiered path end to end.  "
                 "This fidelity block still comes from "
                 "nyaya.index.tiers.evaluate, which drives the same net.rank "
                 "and the same BM25F scorer but bypasses search(), because it "
                 "needs the exhaustive arm as a reference and search() no "
                 "longer offers one."),
    }


# RESOLVED: M3 wired S5 into `search()` (`load_context` now calls
# `TieredIndex.load()` and `_free_text_candidates` routes through
# `tiers.free_text_candidates`).  The live-path consequence is measured in the
# by_mode table: documents scored per query fall 8,032 -> 1,030 at the median and
# p50 latency falls 1.3x-2.4x depending on the scorer.  The fidelity block below
# still runs `nyaya.index.tiers.evaluate` rather than `search()`, on purpose: it
# compares tier-1 against the exhaustive union, and the wired `search()` no
# longer exposes an exhaustive arm to compare against.


# ==========================================================================
# 6. Elimination contribution, measured through the live path
# ==========================================================================


def elimination_effect(queries: Sequence[str] = None,
                       mode: str = "S4", k: int = config.TOP_K,
                       verbose: bool = True) -> Dict[str, Any]:
    """How many documents index elimination removes, on the Track C query set."""
    from nyaya.query.parser import parse
    from nyaya.query.planner import plan_query
    from nyaya.rank.net import measure_elimination
    from nyaya.search import load_context

    ctx = load_context(mode)
    rows: List[Dict[str, Any]] = []
    for raw in (queries or DEFAULT_QUERIES):
        q = parse(raw)
        terms = list(getattr(q, "scoring_terms", None) or q.free_terms)
        if not terms:
            continue
        if q.is_free_text and not q.filters:
            from nyaya.search import _free_text_candidates
            cands = _free_text_candidates(terms, ctx)
        else:
            cands, _p, _s = plan_query(q, ctx)
        m = measure_elimination(cands, terms, ctx, k)
        rows.append({"query": raw, **{kk: m[kk] for kk in (
            "scored_without_elimination", "scored_with_elimination",
            "documents_saved", "fraction_saved", "min_matches_required",
            "top_k_identical", "top_k_overlap")}})
        if verbose:
            print(f"  elim {raw[:32]:<34} "
                  f"{m['scored_without_elimination']:>7,} -> "
                  f"{m['scored_with_elimination']:>6,}", file=sys.stderr)
    tot_w = sum(r["scored_without_elimination"] for r in rows)
    tot_e = sum(r["scored_with_elimination"] for r in rows)
    return {
        "queries": len(rows),
        "total_scored_without": tot_w,
        "total_scored_with": tot_e,
        "fraction_saved": (tot_w - tot_e) / tot_w if tot_w else 0.0,
        "identical_topk": sum(1 for r in rows if r["top_k_identical"]),
        "mean_overlap": (sum(r["top_k_overlap"] for r in rows) / len(rows) / k
                         if rows else 0.0),
        "rows": rows,
    }


# ==========================================================================
# 7. Run everything
# ==========================================================================


def run(queries: Sequence[str] = None, modes: Sequence[str] = MODES,
        repeats: int = REPEATS, k: int = config.TOP_K,
        verbose: bool = True) -> Dict[str, Any]:
    qs = list(queries or DEFAULT_QUERIES)
    by_mode: Dict[str, Any] = {}
    per_query: List[Dict[str, Any]] = []
    for mode in modes:
        if verbose:
            print(f"timing {mode} ({len(qs)} queries x {repeats})", file=sys.stderr)
        entry, rows = measure_mode(mode, qs, repeats, k, verbose=verbose)
        by_mode[mode] = entry
        per_query.extend(rows)

    cold = {m: measure_cold(m, qs[0], verbose=verbose) for m in ("B0", "S4")}

    return {
        "track": "C (efficiency)",
        "queries": len(qs),
        "repeats": repeats,
        "k": k,
        "modes": list(modes),
        "by_mode": by_mode,
        "cold_start": cold,
        "stores": store_sizes(),
        "build": build_times(),
        "tiers": tier_fidelity(),
        "elimination": elimination_effect(qs, "S4", k, verbose=verbose),
        "per_query": per_query,
        "notes": [
            "p50/p95 over the median of 3 repeats per query; latency is "
            "long-tailed, so the mean is not the number to quote.",
            "`scored` is the algorithmic quantity; latency is machine-dependent.",
            "search() now consults ctx.tiers, so the by_mode table measures "
            "the tiered path; the S5 fidelity numbers still come from "
            "nyaya.index.tiers.evaluate because that is the only place an "
            "exhaustive arm survives to compare tier-1 against.",
        ],
    }


# ==========================================================================
# 8. Persistence and reporting
# ==========================================================================


def write_results(s: Dict[str, Any]) -> List[Path]:
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(s, indent=2), encoding="utf-8")

    lines = ["mode,queries,latency_p50_ms,latency_p95_ms,latency_mean_ms,"
             "latency_max_ms,scored_p50,scored_p95,scored_mean,candidates_p50,"
             "total_scored,scorer"]
    for mode in s["modes"]:
        e = s["by_mode"][mode]
        lat, sc, ca = e["latency_ms"], e["scored"], e["candidates"]
        lines.append(",".join(str(x) for x in [
            mode, e["queries"], f"{lat['p50']:.2f}", f"{lat['p95']:.2f}",
            f"{lat['mean']:.2f}", f"{lat['max']:.2f}",
            f"{sc['p50']:.0f}", f"{sc['p95']:.0f}", f"{sc['mean']:.0f}",
            f"{ca['p50']:.0f}", e["total_scored"], e["scorer"],
        ]))
    LATENCY_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pq = ["mode,query,ms_median,candidates,scored,returned,path,parse_ms,"
          "plan_ms,score_ms"]
    for r in s["per_query"]:
        q = r["query"].replace(",", ";")
        pq.append(",".join(str(x) for x in [
            r["mode"], q, f"{r['ms_median']:.3f}", r["candidates"], r["scored"],
            r["returned"], r["path"], r["parse_ms"], r["plan_ms"],
            r["score_ms"]]))
    PERQUERY_CSV.write_text("\n".join(pq) + "\n", encoding="utf-8")

    st = ["store,bytes,human"]
    for name, v in s["stores"]["stores"].items():
        st.append(f"{name.replace(',', ';')},{v['bytes']},{v['human']}")
    st.append(f"TOTAL,{s['stores']['total_bytes']},{s['stores']['total_human']}")
    STORES_CSV.write_text("\n".join(st) + "\n", encoding="utf-8")
    return [RESULTS_JSON, LATENCY_CSV, PERQUERY_CSV, STORES_CSV]


def print_report(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 82)
    p(f"Track C — efficiency   ({s['queries']} queries x {s['repeats']} repeats, "
      f"k={s['k']})")
    p("=" * 82)
    p(f"  {'mode':<5}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}"
      f"{'scored p50':>12}{'scored p95':>12}  scorer")
    for mode in s["modes"]:
        e = s["by_mode"][mode]
        lat, sc = e["latency_ms"], e["scored"]
        p(f"  {mode:<5}{lat['p50']:>9.1f}{lat['p95']:>9.1f}{lat['max']:>9.1f}"
          f"{sc['p50']:>12,.0f}{sc['p95']:>12,.0f}  {e['scorer']}")
    p("")
    p("cold start (fresh interpreter)")
    for mode, c in s["cold_start"].items():
        if "error" in c:
            p(f"  {mode}: FAILED {c['error'][:60]}")
            continue
        p(f"  {mode}: import {c['import_ms']:>6.0f} ms + stores "
          f"{c['load_ms']:>7.0f} ms + first query {c['first_query_ms']:>7.1f} ms"
          f"  -> warm {c['second_query_ms']:.1f} ms")
    p("")
    el = s["elimination"]
    p("index elimination, same query set")
    p(f"  documents scored  : {el['total_scored_without']:,} -> "
      f"{el['total_scored_with']:,}  ({el['fraction_saved']:.1%} fewer)")
    p(f"  top-k identical   : {el['identical_topk']}/{el['queries']}, "
      f"mean overlap {el['mean_overlap']:.3f}")
    p("")
    t = s["tiers"]
    if t.get("available"):
        p(f"two-tier index (r={t['r']}, k={t['k']}, {t['queries']} queries)")
        p(f"  documents scored  : {t['mean_scored_exhaustive']:,.0f} -> "
          f"{t['mean_scored_tier1']:,.0f}  ({t['mean_reduction']:.1%} fewer)")
        p(f"  wall-clock        : {t['mean_speedup']:.2f}x faster")
        p(f"  top-{t['k']} identical  : {t['identical_topk']}/{t['queries']} "
          f"({t['identical_fraction']:.0%}), mean recall@{t['k']} "
          f"{t['mean_recall_at_k']:.3f}")
        p(f"  rank-1 unchanged  : {t['top1_same']}/{t['queries']}")
        if t.get("worst_query"):
            w = t["worst_query"]
            p(f"  worst case        : {w['query']!r} kept only "
              f"{w['overlap']}/{t['k']} of the exhaustive top-{t['k']} "
              f"({w['reduction']:.0%} cut) — the honest failure mode of a "
              f"non-safe optimisation")
    p("")
    p("store sizes")
    for name, v in s["stores"]["stores"].items():
        if v["bytes"]:
            p(f"  {name:<16}{v['human']:>10}")
    p(f"  {'TOTAL':<16}{s['stores']['total_human']:>10}")
    p("=" * 82)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.eval.efficiency",
                                 description="Track C: latency and fidelity.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repeats", type=int, default=REPEATS)
    r.add_argument("--modes", default=",".join(MODES))
    r.add_argument("-k", type=int, default=config.TOP_K)
    c = sub.add_parser("cold")
    c.add_argument("--mode", default="S4")
    sub.add_parser("stores")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    if args.command == "run":
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        with timed("track C"):
            s = run(modes=modes, repeats=args.repeats, k=args.k)
            paths = write_results(s)
        print_report(s)
        print("wrote " + ", ".join(str(p) for p in paths))
    elif args.command == "cold":
        print(json.dumps(measure_cold(args.mode), indent=2))
    elif args.command == "stores":
        s = store_sizes()
        for name, v in s["stores"].items():
            print(f"  {name:<16}{v['human']:>10}")
        print(f"  {'TOTAL':<16}{s['total_human']:>10}")
    else:
        if not RESULTS_JSON.exists():
            raise SystemExit("no efficiency.json — run `run` first")
        print_report(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
