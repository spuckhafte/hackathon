"""Fitting the zone weights, alpha and beta — on the five dev queries only.

The discipline
--------------
`eval/queries/queries.jsonl` marks five queries ``"split": "dev"`` and twenty
``"split": "test"``.  **Only the dev five are read here**, and there is an
assertion to that effect rather than a comment, because a parameter fitted on
the test set makes every number in the report meaningless and the mistake is
invisible once made.

"We tuned it" without the search range is unfalsifiable, so this prints the grid
it searched, the baseline, every improving step and the final values.

Why coordinate ascent rather than an exhaustive grid
-----------------------------------------------------
The full cross product of five zone weights at five values each is 3,125
configurations, times five queries, times a search each — hours.  Coordinate
ascent (optimise one parameter at a time, repeat until nothing improves) costs
about forty configurations and is the standard choice for this shape of problem.
It finds a local optimum, not a global one; that is stated rather than hidden.
``body`` is pinned at 1.0 as the reference scale, because only the *ratios*
between zone weights matter once the score is divided by their sum.

Metrics
-------
Imports `nyaya.eval.metrics` when M1's module exists.  Until then it uses the
small implementation at the bottom of this file, which is standard nDCG and MAP
and is labelled as a stand-in; the import is tried first on every run, so the
moment his lands this defers to it with no edit.

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyaya import config

__all__ = ["load_dev", "evaluate", "coordinate_ascent", "GRID"]

QUERY_FILE = config.EVAL_DIR / "queries" / "queries.jsonl"
QRELS_FILE = config.EVAL_DIR / "queries" / "qrels.jsonl"

#: The search range.  Printed with the result, because a tuned value without
#: its range says nothing about whether the optimum is interior or at an edge.
GRID: Dict[str, List[float]] = {
    "title": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "parties": [0.0, 0.25, 0.5, 1.0],
    "statutes": [0.0, 1.0, 2.0, 3.0, 4.0],
    "order": [0.5, 1.0, 1.5, 2.0, 3.0],
    "alpha": [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00],
    "beta": [0.00, 0.025, 0.05, 0.10, 0.20],
}
#: Pinned: only the ratios matter, and `bench` is filter-only in every mode.
FIXED = {"body": 1.0, "bench": 0.0}


# ==========================================================================
# Data
# ==========================================================================


def load_dev(split: str = "dev") -> Tuple[List[Dict], Dict[str, Dict[int, int]]]:
    """The queries of one split, and the judgments for them."""
    queries = [json.loads(l) for l in
               QUERY_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = [q for q in queries if q.get("split") == split]
    if split == "dev":
        assert chosen, "no dev queries found"
        assert all(q["split"] == "dev" for q in chosen), \
            "a test query leaked into the tuning set"
    ids = {q["qid"] for q in chosen}
    qrels: Dict[str, Dict[int, int]] = {}
    for line in QRELS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["qid"] in ids:
            qrels.setdefault(r["qid"], {})[int(r["doc_id"])] = int(r["rel"])
    return chosen, qrels


# ==========================================================================
# Metrics
# ==========================================================================


def _metrics_module():
    try:
        from nyaya.eval import metrics                       # type: ignore

        return metrics
    except ImportError:
        return None


def ndcg_at_k(ranked: Sequence[int], rels: Dict[int, int], k: int = 10) -> float:
    m = _metrics_module()
    if m is not None and hasattr(m, "ndcg_at_k"):
        return float(m.ndcg_at_k(ranked, rels, k))
    gains = [(2 ** rels.get(int(d), 0) - 1) for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** v - 1 for v in rels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def average_precision(ranked: Sequence[int], rels: Dict[int, int],
                      k: int = 10) -> float:
    m = _metrics_module()
    if m is not None and hasattr(m, "average_precision"):
        return float(m.average_precision(ranked, rels, k))
    hits, total = 0, 0.0
    n_rel = sum(1 for v in rels.values() if v > 0)
    for i, d in enumerate(ranked[:k], 1):
        if rels.get(int(d), 0) > 0:
            hits += 1
            total += hits / i
    return float(total / min(n_rel, k)) if n_rel else 0.0


def precision_at_k(ranked: Sequence[int], rels: Dict[int, int],
                   k: int = 10) -> float:
    if not ranked:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(int(d), 0) > 0) / min(k, len(ranked))


# ==========================================================================
# Evaluation
# ==========================================================================


@dataclass
class Score:
    ndcg: float = 0.0
    map: float = 0.0
    p10: float = 0.0
    per_query: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def __repr__(self) -> str:                               # pragma: no cover
        return f"nDCG@10={self.ndcg:.4f} MAP={self.map:.4f} P@10={self.p10:.4f}"


def evaluate(queries: Sequence[Dict], qrels: Dict[str, Dict[int, int]],
             mode: str = "S4", k: int = 10, ctx=None) -> Score:
    """Run the live system over a query set and score it."""
    from nyaya.search import search

    t0 = time.perf_counter()
    s = Score()
    nd, ap, p10 = [], [], []
    for q in queries:
        rels = qrels.get(q["qid"], {})
        ranked = [r.doc_id for r in search(q["query"], k=k, mode=mode,
                                           explain=False, ctx=ctx)]
        n = ndcg_at_k(ranked, rels, k)
        nd.append(n)
        ap.append(average_precision(ranked, rels, k))
        p10.append(precision_at_k(ranked, rels, k))
        s.per_query[q["qid"]] = n
    s.ndcg = sum(nd) / max(1, len(nd))
    s.map = sum(ap) / max(1, len(ap))
    s.p10 = sum(p10) / max(1, len(p10))
    s.seconds = time.perf_counter() - t0
    return s


def _apply(params: Dict[str, float]) -> None:
    """Install one configuration into `config`, which every scorer reads live."""
    for zone in config.ZONES:
        if zone in FIXED:
            config.ZONE_WEIGHTS[zone] = FIXED[zone]
        elif zone in params:
            config.ZONE_WEIGHTS[zone] = float(params[zone])
    config.ALPHA = float(params.get("alpha", config.ALPHA))
    config.BETA = float(params.get("beta", config.BETA))


def _snapshot() -> Dict[str, float]:
    d = {z: float(w) for z, w in config.ZONE_WEIGHTS.items()}
    d["alpha"] = float(config.ALPHA)
    d["beta"] = float(config.BETA)
    return d


# ==========================================================================
# Coordinate ascent
# ==========================================================================


def coordinate_ascent(queries: Sequence[Dict], qrels: Dict[str, Dict[int, int]],
                      mode: str = "S4", passes: int = 2, k: int = 10,
                      order: Optional[Sequence[str]] = None,
                      verbose: bool = True) -> Tuple[Dict[str, float], Score, List[Dict]]:
    """Optimise one parameter at a time until a whole pass stops improving."""
    from nyaya.search import load_context

    ctx = load_context(mode)
    start = _snapshot()
    best_params = dict(start)
    best = evaluate(queries, qrels, mode=mode, k=k, ctx=ctx)
    trail = [{"step": "baseline", "params": dict(best_params),
              "ndcg": best.ndcg, "map": best.map}]
    if verbose:
        print(f"  baseline  {best}   {_fmt(best_params)}", file=sys.stderr)

    names = list(order or ["title", "statutes", "order", "parties",
                           "alpha", "beta"])
    n_evals = 1
    for p in range(passes):
        improved = False
        for name in names:
            values = GRID[name]
            local_best, local_value = best, best_params.get(name)
            for v in values:
                if v == best_params.get(name):
                    continue
                trial = dict(best_params)
                trial[name] = v
                _apply(trial)
                sc = evaluate(queries, qrels, mode=mode, k=k, ctx=ctx)
                n_evals += 1
                if sc.ndcg > local_best.ndcg + 1e-9:
                    local_best, local_value = sc, v
            if local_value != best_params.get(name):
                best_params[name] = local_value
                best = local_best
                improved = True
                trail.append({"step": f"pass{p + 1}:{name}",
                              "params": dict(best_params),
                              "ndcg": best.ndcg, "map": best.map})
                if verbose:
                    print(f"  {name:<9} -> {local_value:<6}  {best}",
                          file=sys.stderr)
            _apply(best_params)
        if not improved:
            break
    _apply(start)                       # never leave config mutated
    trail.append({"step": "evaluations", "n": n_evals})
    return best_params, best, trail


def _fmt(p: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:g}" for k, v in p.items() if k != "bench")


# ==========================================================================
# CLI
# ==========================================================================


def _main(argv: Optional[Sequence[str]] = None) -> int:       # pragma: no cover
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.rank.tune")
    ap.add_argument("--mode", default="S4")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--split", default="dev",
                    help="which split to FIT on; 'test' is refused")
    ap.add_argument("--report-test", action="store_true",
                    help="after fitting on dev, score the test set once")
    args = ap.parse_args(argv)

    if args.split != "dev":
        print("refusing to fit on anything but the dev split", file=sys.stderr)
        return 2

    dev, dev_qrels = load_dev("dev")
    print(f"fitting on {len(dev)} dev queries: "
          f"{', '.join(q['qid'] for q in dev)}")
    m = _metrics_module()
    print(f"metrics: {'nyaya.eval.metrics' if m else 'local stand-in (M1 has not landed)'}")
    print("\nsearch range")
    for name, values in GRID.items():
        print(f"  {name:<9} {values}")
    print(f"  fixed     {FIXED}  (only the ratios between zone weights matter)")
    print()

    before = _snapshot()
    best_params, best, trail = coordinate_ascent(dev, dev_qrels, mode=args.mode,
                                                 passes=args.passes, k=args.k)
    n_evals = next((t["n"] for t in trail if t["step"] == "evaluations"), 0)

    print(f"\n{'parameter':<10}{'before':>10}{'after':>10}")
    print("-" * 30)
    for name in ("title", "parties", "statutes", "order", "body", "bench",
                 "alpha", "beta"):
        b = before.get(name)
        a = best_params.get(name, FIXED.get(name, b))
        flag = "  <-" if b is not None and abs(float(a) - float(b)) > 1e-9 else ""
        print(f"{name:<10}{b:>10g}{float(a):>10g}{flag}")

    base = trail[0]
    print(f"\n{'':<10}{'baseline':>10}{'tuned':>10}{'delta':>10}")
    print("-" * 40)
    print(f"{'nDCG@10':<10}{base['ndcg']:>10.4f}{best.ndcg:>10.4f}"
          f"{best.ndcg - base['ndcg']:>+10.4f}")
    print(f"{'MAP':<10}{base['map']:>10.4f}{best.map:>10.4f}"
          f"{best.map - base['map']:>+10.4f}")
    print(f"\n{n_evals} configurations evaluated by coordinate ascent "
          f"({args.passes} passes max). This is a local optimum, not a global one.")

    out = config.RUNS / "tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fitted_on": [q["qid"] for q in dev], "mode": args.mode,
        "grid": GRID, "fixed": FIXED, "before": before, "after": best_params,
        "baseline_ndcg": base["ndcg"], "tuned_ndcg": best.ndcg,
        "baseline_map": base["map"], "tuned_map": best.map,
        "evaluations": n_evals, "trail": trail,
        "metrics": "nyaya.eval.metrics" if m else "local stand-in",
    }, indent=1), encoding="utf-8")
    print(f"\nwritten to {out}")

    if args.report_test:
        test, test_qrels = load_dev("test")
        _apply(before)
        b_test = evaluate(test, test_qrels, mode=args.mode, k=args.k)
        _apply(best_params)
        a_test = evaluate(test, test_qrels, mode=args.mode, k=args.k)
        _apply(before)
        print(f"\ntest set ({len(test)} queries), scored ONCE, never fitted on")
        print(f"  nDCG@10  {b_test.ndcg:.4f} -> {a_test.ndcg:.4f}  "
              f"({a_test.ndcg - b_test.ndcg:+.4f})")
        print(f"  MAP      {b_test.map:.4f} -> {a_test.map:.4f}  "
              f"({a_test.map - b_test.map:+.4f})")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
