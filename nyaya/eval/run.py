"""Track A: the retrieval-effectiveness harness  (Member 1, wave 4).

Runs the judged query set through **the live system** - every system is driven by
``nyaya.search.search(q, k, mode=...)``, never by a private pipeline assembled
for the evaluation - and writes the tables the report and the video read.

Why that matters: a harness that re-implements scoring measures the harness.
Driving ``search()`` means every number here describes the thing a user actually
gets, and the efficiency columns come from the ``SearchReport`` the same call
fills in.

Outputs, all under ``runs/``
---------------------------
``runs/trackA_summary.csv``    one row per system: the five metrics plus timings
``runs/trackA_per_query.csv``  one row per (system, query): every metric
``runs/trackA_runs.jsonl``     the actual rankings, so a result can be re-checked
``runs/trackA.json``           everything above plus the comparisons, for data.ts

Key names are stable because the report and the video's ``data.ts`` read them
directly; renaming a column here silently breaks both.

Splits
------
``eval/queries/queries.jsonl`` marks 5 queries ``dev`` (used to tune the zone
weights and BM25F parameters) and 20 ``test``.  The default is ``--split test``
and the dev queries are **excluded**, because quoting a tuned-on number as a
result is the single easiest way to overstate a system.  ``--split all`` prints
both, labelled.

    python -m nyaya.eval.run --systems B0 S1 S2 S2p S3 S4
    python -m nyaya.eval.run --split all --plots
    python -m nyaya.eval.run --systems B0 S4 --compare S4 B0
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nyaya import config
from nyaya.contracts import MODES
from nyaya.eval import metrics as M
from nyaya.utils import ensure_utf8_stdout, timed

QUERIES = config.EVAL_DIR / "queries" / "queries.jsonl"
QRELS = config.EVAL_DIR / "queries" / "qrels.jsonl"

SUMMARY_CSV = config.RUNS / "trackA_summary.csv"
PER_QUERY_CSV = config.RUNS / "trackA_per_query.csv"
RUNS_JSONL = config.RUNS / "trackA_runs.jsonl"
REPORT_JSON = config.RUNS / "trackA.json"

#: How deep to retrieve.  20 because R@20 is one of the reported metrics; the
#: precision cutoffs are 5 and 10 and nDCG is at 10, so one run serves all five.
#: It is also the AP cutoff — AP is averaged over the whole submitted ranking —
#: which is why this harness reports MAP@20 and `nyaya.rank.tune` reports MAP@10.
RETRIEVE_K = 20
assert RETRIEVE_K == M.AP_CUTOFF, "metrics.MAP_NAME would mislabel the cutoff"

DEFAULT_SYSTEMS = list(MODES)


# ==========================================================================
# 1. One (system, query) run
# ==========================================================================

@dataclass
class RunRow:
    """One query under one system: the ranking and what it cost."""

    system: str
    qid: str
    query: str
    kind: str = "free"
    split: str = "test"
    ranked: List[int] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    parse_ms: float = 0.0
    plan_ms: float = 0.0
    score_ms: float = 0.0
    candidates: int = 0
    scored: int = 0
    path: str = ""
    scorer: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    def as_json(self) -> Dict[str, Any]:
        return {
            "system": self.system, "qid": self.qid, "query": self.query,
            "kind": self.kind, "split": self.split, "ranked": self.ranked,
            "titles": self.titles, "scores": [round(s, 6) for s in self.scores],
            "elapsed_ms": round(self.elapsed_ms, 2),
            "parse_ms": round(self.parse_ms, 2),
            "plan_ms": round(self.plan_ms, 2),
            "score_ms": round(self.score_ms, 2),
            "candidates": self.candidates, "scored": self.scored,
            "path": self.path, "scorer": self.scorer,
            "warnings": self.warnings, "error": self.error,
        }


def run_one(system: str, q: Dict[str, Any], k: int = RETRIEVE_K) -> RunRow:
    """One query through ``search()``, with the report it fills in."""
    from nyaya import search as S

    row = RunRow(system=system, qid=str(q["qid"]), query=str(q["query"]),
                 kind=str(q.get("kind") or "free"),
                 split=str(q.get("split") or "test"))
    report = S.SearchReport(mode=system)
    t0 = time.perf_counter()
    try:
        results = S.search(row.query, k=k, mode=system, explain=False,
                           report=report)
    except Exception as exc:                       # a broken system is a result
        row.error = f"{type(exc).__name__}: {exc}"
        row.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return row
    row.ranked = [int(r.doc_id) for r in results]
    row.titles = [r.title[:70] for r in results]
    row.scores = [float(r.score) for r in results]
    row.elapsed_ms = report.elapsed_ms or (time.perf_counter() - t0) * 1000.0
    row.parse_ms = report.parse_ms
    row.plan_ms = report.plan_ms
    row.score_ms = report.score_ms
    row.candidates = report.candidates
    row.scored = report.scored
    row.path = report.path
    row.scorer = report.scorer
    row.warnings = list(report.warnings or ())
    return row


# ==========================================================================
# 2. The whole grid
# ==========================================================================

def _select(queries: Sequence[Dict[str, Any]], split: str
            ) -> List[Dict[str, Any]]:
    if split == "all":
        return list(queries)
    return [q for q in queries if str(q.get("split") or "test") == split]


def run_grid(systems: Sequence[str], split: str = "test", k: int = RETRIEVE_K,
             verbose: bool = True) -> Tuple[List[RunRow], List[Dict[str, Any]],
                                            Dict[str, Dict[int, int]]]:
    queries = M.load_queries(QUERIES)
    qrels = M.load_qrels(QRELS)
    chosen = _select(queries, split)
    if not chosen:
        raise SystemExit(f"no queries with split={split!r}")
    rows: List[RunRow] = []
    for system in systems:
        t0 = time.perf_counter()
        for q in chosen:
            rows.append(run_one(system, q, k=k))
        if verbose:
            bad = sum(1 for r in rows if r.system == system and r.error)
            print(f"  {system:<4} {len(chosen)} queries in "
                  f"{time.perf_counter() - t0:.1f}s"
                  + (f"  ({bad} errored)" if bad else ""),
                  file=sys.stderr, flush=True)
    return rows, chosen, qrels


def score_runs(rows: Sequence[RunRow], qrels: Dict[str, Dict[int, int]]
               ) -> Dict[str, List[M.QueryScore]]:
    out: Dict[str, List[M.QueryScore]] = {}
    for row in rows:
        rels = qrels.get(row.qid, {})
        out.setdefault(row.system, []).append(
            M.score_query(row.qid, row.system, row.ranked, rels))
    return out


def efficiency(rows: Sequence[RunRow]) -> Dict[str, Dict[str, float]]:
    """Median and p90 latency per system, plus the candidate/scored counts.

    Median rather than mean: with 20 queries one slow Boolean plan drags a mean
    around and the typical latency is what a user experiences.
    """
    by_system: Dict[str, List[RunRow]] = {}
    for r in rows:
        by_system.setdefault(r.system, []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for system, rs in by_system.items():
        lat = sorted(r.elapsed_ms for r in rs if not r.error)
        if not lat:
            out[system] = {}
            continue
        out[system] = {
            "median_ms": round(statistics.median(lat), 2),
            "mean_ms": round(statistics.fmean(lat), 2),
            "p90_ms": round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 2),
            "max_ms": round(lat[-1], 2),
            "mean_candidates": round(statistics.fmean(
                [r.candidates for r in rs if not r.error]), 1),
            "mean_scored": round(statistics.fmean(
                [r.scored for r in rs if not r.error]), 1),
            "mean_parse_ms": round(statistics.fmean(
                [r.parse_ms for r in rs if not r.error]), 2),
            "mean_plan_ms": round(statistics.fmean(
                [r.plan_ms for r in rs if not r.error]), 2),
            "mean_score_ms": round(statistics.fmean(
                [r.score_ms for r in rs if not r.error]), 2),
            "errors": float(sum(1 for r in rs if r.error)),
        }
    return out


# ==========================================================================
# 3. Writing the artefacts
# ==========================================================================

def write_outputs(rows: Sequence[RunRow],
                  scores: Dict[str, List[M.QueryScore]],
                  eff: Dict[str, Dict[str, float]],
                  queries: Sequence[Dict[str, Any]],
                  qrels: Dict[str, Dict[int, int]],
                  split: str,
                  comparisons: Sequence[Tuple[str, str]] = (("S4", "B0"),),
                  ) -> Dict[str, Any]:
    config.RUNS.mkdir(parents=True, exist_ok=True)

    systems = list(scores)
    agg = {s: M.aggregate(scores[s]) for s in systems}

    # ---- summary table ------------------------------------------------
    fields = (["system"] + list(M.METRIC_NAMES) + ["RR", "n_queries"]
              + ["median_ms", "p90_ms", "mean_candidates", "mean_scored",
                 "errors"])
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in systems:
            row = {"system": s}
            row.update({m: round(agg[s].get(m, float("nan")), 4)
                        for m in M.METRIC_NAMES})
            row["RR"] = round(agg[s].get("RR", float("nan")), 4)
            row["n_queries"] = int(agg[s].get("n_queries", 0))
            for key in ("median_ms", "p90_ms", "mean_candidates", "mean_scored",
                        "errors"):
                row[key] = eff.get(s, {}).get(key, "")
            w.writerow(row)

    # ---- per-query table ----------------------------------------------
    pq_fields = ["qid", "system", "returned", "relevant_judged"] + \
        list(M.METRIC_NAMES) + ["RR"]
    with PER_QUERY_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pq_fields)
        w.writeheader()
        for s in systems:
            for qs in scores[s]:
                row = qs.as_row()
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in row.items() if k in pq_fields})

    # ---- the rankings themselves --------------------------------------
    with RUNS_JSONL.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.as_json(), ensure_ascii=False))
            fh.write("\n")

    # ---- comparisons ---------------------------------------------------
    comps: List[Dict[str, Any]] = []
    for a, b in comparisons:
        if a in scores and b in scores:
            for metric in ("nDCG@10", M.MAP_NAME, "P@10"):
                comps.append(M.per_query_comparison(scores[a], scores[b],
                                                    metric=metric))

    report = {
        "split": split,
        "retrieve_k": RETRIEVE_K,
        "systems": systems,
        "queries": len(queries),
        "qrels": M.qrels_summary(qrels),
        "summary": {s: {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in agg[s].items()} for s in systems},
        "efficiency": eff,
        "comparisons": comps,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {"summary_csv": str(SUMMARY_CSV),
                  "per_query_csv": str(PER_QUERY_CSV),
                  "runs_jsonl": str(RUNS_JSONL)},
    }
    REPORT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    return report


# ==========================================================================
# 4. Printing
# ==========================================================================

def print_table(report: Dict[str, Any]) -> None:
    systems = report["systems"]
    print("=" * 86)
    print(f"Track A - retrieval effectiveness, split={report['split']}, "
          f"{report['summary'][systems[0]]['n_queries']:.0f} queries, "
          f"k={report['retrieve_k']}")
    print("=" * 86)
    head = f"{'system':<7}" + "".join(f"{m:>10}" for m in M.METRIC_NAMES) + \
        f"{'med ms':>9}{'cands':>9}{'scored':>8}"
    print(head)
    for s in systems:
        agg = report["summary"][s]
        eff = report["efficiency"].get(s, {})
        line = f"{s:<7}" + "".join(
            f"{agg.get(m, float('nan')):>10.4f}" for m in M.METRIC_NAMES)
        line += f"{eff.get('median_ms', 0):>9.1f}"
        line += f"{eff.get('mean_candidates', 0):>9.0f}"
        line += f"{eff.get('mean_scored', 0):>8.0f}"
        print(line)

    for comp in report["comparisons"]:
        if comp["metric"] != "nDCG@10":
            continue
        c = comp["counts"]
        st = comp["sign_test"]
        bs = comp["bootstrap"]
        print()
        print(f"{comp['a_system']} vs {comp['b_system']} on {comp['metric']}: "
              f"mean {comp['mean_a']:.4f} vs {comp['mean_b']:.4f} "
              f"(delta {comp['mean_a'] - comp['mean_b']:+.4f})")
        print(f"  per query: {c['improved']} improved, {c['tied']} tied, "
              f"{c['worsened']} worsened")
        print(f"  sign test: {st['wins']}-{st['losses']} "
              f"(ties {st['ties']}), p = {st['p_value']:.4f}"
              f"{'  SIGNIFICANT' if st['significant_at_05'] else '  not significant'}")
        print(f"  bootstrap 95% CI on the mean difference: "
              f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}]"
              f"{'  excludes 0' if bs['excludes_zero'] else '  includes 0'}")
        print(f"  With {st['wins'] + st['losses'] + st['ties']} queries a few "
              f"points of mean difference is not evidence on its own; the "
              f"per-query breakdown above is the result.")


def print_per_query(report: Dict[str, Any], metric: str = "nDCG@10") -> None:
    comp = next((c for c in report["comparisons"] if c["metric"] == metric), None)
    if comp is None:
        return
    print()
    print(f"per-query {metric}: {comp['a_system']} vs {comp['b_system']}")
    print(f"{'qid':<6}{comp['a_system']:>9}{comp['b_system']:>9}{'delta':>9}"
          f"  outcome")
    for r in comp["per_query"]:
        print(f"{r['qid']:<6}{r['a']:>9.4f}{r['b']:>9.4f}{r['delta']:>+9.4f}"
              f"  {r['outcome']}")


# ==========================================================================
# 5. CLI
# ==========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.eval.run",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS,
                    help=f"any of {' '.join(MODES)}")
    ap.add_argument("--split", default="test", choices=("test", "dev", "all"))
    ap.add_argument("--k", type=int, default=RETRIEVE_K)
    ap.add_argument("--compare", nargs=2, action="append", default=None,
                    metavar=("A", "B"),
                    help="per-query comparison of A against B (default S4 B0)")
    ap.add_argument("--per-query", action="store_true",
                    help="print the per-query table as well")
    ap.add_argument("--plots", action="store_true",
                    help="also write the report's charts to runs/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    unknown = [s for s in args.systems if s not in MODES]
    if unknown:
        raise SystemExit(f"unknown system(s) {unknown}; expected {list(MODES)}")
    comparisons = [tuple(c) for c in (args.compare or [["S4", "B0"]])]

    if not QRELS.exists():
        raise SystemExit(f"{QRELS} missing - the judged query set is Member 3's")

    with timed("trackA"):
        rows, queries, qrels = run_grid(args.systems, split=args.split,
                                        k=args.k, verbose=not args.quiet)
        scores = score_runs(rows, qrels)
        eff = efficiency(rows)
        report = write_outputs(rows, scores, eff, queries, qrels,
                               split=args.split, comparisons=comparisons)

    print_table(report)
    if args.per_query:
        print_per_query(report)
    print(f"\nwrote {SUMMARY_CSV}")
    print(f"wrote {PER_QUERY_CSV}")
    print(f"wrote {RUNS_JSONL}")
    print(f"wrote {REPORT_JSON}")

    if args.plots:
        from nyaya.eval import plots

        for p in plots.all_plots(report):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
