"""Store S4, part 2 — PageRank over the citation graph, and g(d).

Member 2 (Suvrat).  Our own power iteration, no library in the loop:

    PR(d) = (1-lambda)/N  +  lambda * [ sum_{c -> d} PR(c) * w(c,d)/W_out(c)
                                        + (dangling mass) / N ]

with ``lambda = config.PAGERANK_LAMBDA``.

**Dangling nodes are the whole difficulty here, not a footnote.**  Edges point
into the 37,210-node *universe*, but only the 13,942 sampled judgments have text
we could read citations out of, and only ~34% of those cite anything our
extractor recognises.  So the overwhelming majority of nodes have **no
out-links** — they are sinks.  A sink's share of the probability mass has
nowhere to go; if it is simply dropped, every iteration loses mass, the vector
decays towards zero and the ranking it induces is meaningless.  The standard fix
is applied: each iteration collects the mass sitting on dangling nodes and
redistributes it uniformly over all N nodes, which is exactly the behaviour of
the random surfer who teleports when it lands somewhere with no outgoing link.
The implementation asserts the vector still sums to 1.

**Edge weighting is pluggable, because it is the S3 -> S4 ablation.**

* ``uniform``   — every citation counts the same.  Plain PageRank.  **System S3.**
* ``treatment`` — a citation is weighted by how the citing judgment *treated* the
  precedent (``config.TREATMENT_WEIGHTS``: followed 1.0, mentioned 0.6,
  distinguished 0.3, overruled 0.0).  Being overruled contributes no authority,
  which is the point: a much-cited-but-overruled judgment should not outrank live
  law.  **System S4.**  M1's classifier labels ``CitationEdge.treatment`` in wave
  3; until then every edge is unlabelled and this mode falls back to the
  ``mentioned`` weight, so running it now would just be uniform PageRank with a
  constant factor — hence S3 is what we run today and S4 is a re-run of the same
  code path once the labels land.

Then the static quality score of the contract:

    g(d) = penalty(d) * [ G_PAGERANK_WEIGHT * PR_norm(d)
                        + G_BENCH_WEIGHT    * bench_norm(d) ]

``PR_norm`` is min-maxed **on a log scale**, because raw PageRank over a citation
graph is heavily skewed (a power law): min-maxing the raw values puts almost
every judgment within a hair of 0 and lets one outlier own the whole range, so
the authority term would stop discriminating at all.  Both the raw and the logged
values are persisted so the report can show the skew.

CLI::

    python -m nyaya.authority.pagerank build [--weighting uniform|treatment]
    python -m nyaya.authority.pagerank top [--n 20]
    python -m nyaya.authority.pagerank validate       # vs networkx
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nyaya import config
from nyaya.contracts import Authority, CitationEdge
from nyaya.metadata import load_manifest, load_universe
from nyaya.utils import ensure_utf8_stdout, timed

#: The shipped store that ``AuthorityStore`` (M3) loads by default.  Plain
#: PageRank, i.e. system S3.
AUTHORITY_JSON = config.INDEX / "authority.json"
#: The treatment-weighted store, i.e. system S4.  Written to its own file so
#: **both systems are live at once** and the S3 -> S4 ablation is a flag rather
#: than a rebuild.
AUTHORITY_TREATMENT_JSON = config.INDEX / "authority_treatment.json"
PAGERANK_STATS = config.INDEX / "pagerank_stats.json"
PAGERANK_TREATMENT_STATS = config.INDEX / "pagerank_treatment_stats.json"

WEIGHTINGS = ("uniform", "treatment")


def authority_path(weighting: str = "uniform") -> Path:
    """Where the store for ``weighting`` lives."""
    return (AUTHORITY_TREATMENT_JSON if weighting == "treatment"
            else AUTHORITY_JSON)


def stats_path(weighting: str = "uniform") -> Path:
    return (PAGERANK_TREATMENT_STATS if weighting == "treatment"
            else PAGERANK_STATS)

#: Universe-only nodes have no PDF, hence no CORAM line, hence no real bench
#: size.  Using `bench_size_meta` would hand 87% of them a bench of 1 and
#: systematically penalise every judgment we happen not to have downloaded.
#: Instead they get the corpus median PDF-derived bench as a neutral prior.
#: This is honest because g(d) for a universe-only node is never used to rank
#: anything: only indexed documents appear in results.  The value exists so the
#: Authority record is complete and so the bench term is not silently zero.
BENCH_PRIOR_FALLBACK = 2


# ==========================================================================
# 1. The graph
# ==========================================================================


class CitationGraph:
    """Weighted directed graph over universe ``node_id``s.

    ``src`` on a :class:`~nyaya.contracts.CitationEdge` is a *doc_id* (only an
    indexed document can cite), so it is mapped to its ``node_id`` here — the
    graph is single-id-space, which is what makes PageRank's matrix square.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = int(n_nodes)
        #: (src_node, dst_node) -> accumulated weight
        self.weights: Dict[Tuple[int, int], float] = defaultdict(float)
        self.treatment_counts: Dict[int, Counter] = defaultdict(Counter)
        #: cited node -> the nodes that overruled it (fills in wave 3)
        self.overruled_by: Dict[int, List[int]] = defaultdict(list)
        self.dropped_self = 0
        self.dropped_oob = 0
        self.parallel_collapsed = 0

    @classmethod
    def from_edges(cls, edges: Iterable[CitationEdge], n_nodes: int,
                   doc_to_node: Dict[int, int],
                   weighting: str = "uniform") -> "CitationGraph":
        g = cls(n_nodes)
        for e in edges:
            if e.dst < 0:
                continue                           # unresolved, not in the graph
            src = doc_to_node.get(int(e.src), -1)
            dst = int(e.dst)
            if src < 0 or dst < 0 or src >= n_nodes or dst >= n_nodes:
                g.dropped_oob += 1
                continue
            if src == dst:
                g.dropped_self += 1                # a judgment citing itself
                continue
            w = edge_weight(e, weighting)
            key = (src, dst)
            if key in g.weights:
                # The same precedent cited in ten paragraphs is still one
                # citing-cited relationship; weights accumulate rather than
                # creating ten parallel edges, and we count how often that
                # happens so the report can show it.
                g.parallel_collapsed += 1
            g.weights[key] += w
            treatment = (e.treatment or "").strip().lower() or "mentioned"
            g.treatment_counts[dst][treatment] += 1
            if treatment == "overruled" and src not in g.overruled_by[dst]:
                g.overruled_by[dst].append(src)
        return g

    # -- CSR-ish arrays for the iteration ---------------------------------
    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(src, dst, w, w_out)``: edge lists plus per-node out-weight."""
        if not self.weights:
            return (np.empty(0, np.int32), np.empty(0, np.int32),
                    np.empty(0, np.float64), np.zeros(self.n_nodes))
        keys = np.fromiter((k for pair in self.weights for k in pair),
                           dtype=np.int64, count=2 * len(self.weights))
        keys = keys.reshape(-1, 2)
        src = keys[:, 0].astype(np.int32)
        dst = keys[:, 1].astype(np.int32)
        w = np.fromiter(self.weights.values(), dtype=np.float64,
                        count=len(self.weights))
        w_out = np.zeros(self.n_nodes, dtype=np.float64)
        np.add.at(w_out, src, w)
        return src, dst, w, w_out

    @property
    def n_edges(self) -> int:
        return len(self.weights)

    def in_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes, dtype=np.int32)
        for (_s, d) in self.weights:
            deg[d] += 1
        return deg

    def out_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes, dtype=np.int32)
        for (s, _d) in self.weights:
            deg[s] += 1
        return deg


def edge_weight(edge: CitationEdge, weighting: str = "uniform") -> float:
    """The pluggable part: how much authority one citation transfers.

    ``uniform`` is plain PageRank (system S3).  ``treatment`` reads
    ``config.TREATMENT_WEIGHTS`` (system S4) and treats an unlabelled edge as
    ``mentioned``, which is what every edge is until M1's classifier runs.
    """
    if weighting == "uniform":
        return 1.0
    if weighting == "treatment":
        t = (edge.treatment or "").strip().lower() or "mentioned"
        return float(config.TREATMENT_WEIGHTS.get(t, config.TREATMENT_WEIGHTS["mentioned"]))
    raise ValueError(f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")


# ==========================================================================
# 2. Power iteration
# ==========================================================================


def pagerank(graph: CitationGraph,
             lam: float = config.PAGERANK_LAMBDA,
             iterations: int = config.PAGERANK_ITERATIONS,
             tol: float = config.PAGERANK_TOL,
             verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Weighted PageRank by power iteration, with dangling-mass redistribution."""
    n = graph.n_nodes
    if n == 0:
        return np.empty(0), {"iterations": 0, "converged": True}
    src, dst, w, w_out = graph.arrays()
    # A node is a sink when its total OUT-WEIGHT is zero, which is not the same
    # as having no out-edges.  Under `treatment` weighting an "overruled" edge
    # carries weight 0.0, so a judgment whose only citation was an overruling has
    # edges but no mass to send.  Such edges must be dropped from the iteration:
    # keeping them divides by a zero out-weight and the whole vector goes NaN.
    live = w_out[src] > 0.0
    src, dst, w = src[live], dst[live], w[live]
    dangling = w_out == 0.0
    teleport = (1.0 - lam) / n

    pr = np.full(n, 1.0 / n, dtype=np.float64)
    info: Dict[str, Any] = {"n_nodes": n, "n_edges": graph.n_edges,
                            "dangling_nodes": int(dangling.sum()),
                            "lambda": lam, "delta_history": []}
    it = 0
    for it in range(1, iterations + 1):
        # Mass that would otherwise be lost on sinks, spread over everyone.
        dangling_mass = float(pr[dangling].sum())
        nxt = np.full(n, teleport + lam * dangling_mass / n, dtype=np.float64)
        if src.size:
            # contribution of each edge: PR(src) * w / W_out(src)
            np.add.at(nxt, dst, lam * pr[src] * w / w_out[src])
        delta = float(np.abs(nxt - pr).sum())
        pr = nxt
        info["delta_history"].append(delta)
        if verbose:
            print(f"  iter {it:>3}  L1 delta {delta:.3e}  sum {pr.sum():.12f}",
                  file=sys.stderr)
        if delta < tol:
            break
    info["iterations"] = it
    info["converged"] = bool(info["delta_history"] and info["delta_history"][-1] < tol)
    info["final_delta"] = info["delta_history"][-1] if info["delta_history"] else 0.0
    info["mass"] = float(pr.sum())
    # The redistribution is what keeps this true; if it ever fails, the ranking
    # below is not a probability distribution and must not be trusted.
    assert abs(pr.sum() - 1.0) < 1e-9, f"PageRank mass leaked: {pr.sum()}"
    return pr, info


def pagerank_networkx(graph: CitationGraph,
                      lam: float = config.PAGERANK_LAMBDA) -> np.ndarray:
    """Validation only — never used to produce a ranking we ship."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.n_nodes))
    for (s, d), w in graph.weights.items():
        G.add_edge(s, d, weight=w)
    scores = nx.pagerank(G, alpha=lam, weight="weight",
                         tol=1e-12, max_iter=500)
    out = np.zeros(graph.n_nodes, dtype=np.float64)
    for k, v in scores.items():
        out[k] = v
    return out


# ==========================================================================
# 3. g(d)
# ==========================================================================


def log_minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalise on a log scale.

    PageRank over a citation graph is power-law distributed.  Plain min-max
    leaves the median document at ~1e-4 of the range, so the authority term
    becomes a constant for all but a handful of outliers.  ``log`` first, then
    min-max, keeps the ordering identical while spreading the mass across [0, 1].
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    floor = v[v > 0].min() if np.any(v > 0) else 1.0
    lv = np.log(np.maximum(v, floor))
    lo, hi = float(lv.min()), float(lv.max())
    if hi - lo < 1e-15:
        return np.zeros_like(v)
    return (lv - lo) / (hi - lo)


def bench_norm(bench: np.ndarray) -> np.ndarray:
    """Bench size scaled by the largest bench the Court has ever sat (13)."""
    b = np.asarray(bench, dtype=np.float64)
    return np.clip(b, 0.0, config.MAX_BENCH) / config.MAX_BENCH


def treatment_status(graph: CitationGraph) -> Dict[int, Tuple[str, float, List[int]]]:
    """``node -> (status, penalty, overruled_by)`` from the in-edge treatments.

    A judgment is ``overruled`` once any in-edge says so, and
    ``distinguished`` once ``config.DISTINGUISHED_SHARE`` of at least
    ``config.DISTINGUISHED_MIN_EDGES`` in-edges distinguish it.  Every edge is
    unlabelled until M1's classifier runs, so today this returns ``ok`` with
    penalty 1.0 everywhere — the code path is live and the numbers will move when
    the labels land.
    """
    out: Dict[int, Tuple[str, float, List[int]]] = {}
    for node, counts in graph.treatment_counts.items():
        total = sum(counts.values())
        if counts.get("overruled", 0) > 0:
            out[node] = ("overruled", config.TREATMENT_PENALTY["overruled"],
                         sorted(graph.overruled_by.get(node, [])))
        elif (total >= config.DISTINGUISHED_MIN_EDGES
              and counts.get("distinguished", 0) / total >= config.DISTINGUISHED_SHARE):
            out[node] = ("distinguished", config.TREATMENT_PENALTY["distinguished"], [])
        else:
            out[node] = ("ok", config.TREATMENT_PENALTY["ok"], [])
    return out


def build_authority(graph: CitationGraph,
                   pr: np.ndarray,
                   bench: np.ndarray,
                   use_treatment: bool = False) -> Dict[int, Authority]:
    """Assemble the :class:`~nyaya.contracts.Authority` record for every node.

    ``use_treatment`` is what separates the two systems, and it has to gate the
    *penalty* as well as the edge weights:

    * **S3** is plain PageRank.  It knows nothing about how a precedent was
      treated, so every judgment is ``status="ok"`` with ``penalty=1.0`` even
      though the edges happen to carry labels.  Applying the penalty here would
      make S3 treatment-aware and the S3 -> S4 ablation would under-report the
      difference — the two stores would differ only in edge weights, and the
      headline "overruled judgments drop out of the ranking" effect would already
      be present in the baseline.
    * **S4** reads ``config.TREATMENT_PENALTY`` through
      :func:`treatment_status`.
    """
    pr_norm = log_minmax(pr)
    b_norm = bench_norm(bench)
    g = (config.G_PAGERANK_WEIGHT * pr_norm + config.G_BENCH_WEIGHT * b_norm)
    status = treatment_status(graph) if use_treatment else {}
    indeg, outdeg = graph.in_degree(), graph.out_degree()

    out: Dict[int, Authority] = {}
    for node in range(graph.n_nodes):
        st, penalty, overruled_by = status.get(node, ("ok", config.TREATMENT_PENALTY["ok"], []))
        out[node] = Authority(
            doc_id=node,                 # node_id; the field name is the contract's
            g=float(penalty * g[node]),
            pagerank=float(pr[node]),
            pagerank_norm=float(pr_norm[node]),
            bench_norm=float(b_norm[node]),
            status=st,
            penalty=float(penalty),
            in_degree=int(indeg[node]),
            out_degree=int(outdeg[node]),
            overruled_by=list(overruled_by),
        )
    return out


def bench_vector(universe: pd.DataFrame,
                 manifest: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Per-node bench size: the **merged** (PDF-derived) value where we have it.

    ``bench_size_meta`` is never read here.  It is one name in 87% of rows, so
    using it would tell g(d) that essentially the entire Supreme Court sits
    alone, and the bench term (``config.G_BENCH_WEIGHT`` = 0.25 of g) would turn
    into noise.  Indexed documents take M1's CORAM-parsed bench out of store S2;
    universe-only nodes have no PDF at all and take
    :data:`BENCH_PRIOR_FALLBACK` — see the note on that constant.
    """
    n = len(universe)
    bench = np.zeros(n, dtype=np.float64)
    provenance = {"pdf": 0, "prior": 0}

    from nyaya.index.parametric import ParametricIndex

    man = load_manifest() if manifest is None else manifest
    try:
        s2 = ParametricIndex.load()
        bench_by_doc = s2.column("bench")
    except FileNotFoundError:
        bench_by_doc = None

    median_bench = BENCH_PRIOR_FALLBACK
    if bench_by_doc is not None:
        nz = bench_by_doc[bench_by_doc > 0]
        if nz.size:
            median_bench = int(np.median(nz))
        for doc_id, node_id in zip(man["doc_id"], man["node_id"]):
            b = int(bench_by_doc[int(doc_id)]) if int(doc_id) < bench_by_doc.size else 0
            if b > 0:
                bench[int(node_id)] = b
                provenance["pdf"] += 1

    missing = bench == 0.0
    bench[missing] = median_bench
    provenance["prior"] = int(missing.sum())
    provenance["prior_value"] = int(median_bench)
    return bench, provenance


# ==========================================================================
# 4. Build / persist
# ==========================================================================


def build(weighting: str = "uniform",
          validate: bool = True,
          verbose: bool = True) -> Tuple[Dict[int, Authority], Dict[str, Any]]:
    from nyaya.authority.citations import load_edges

    uni = load_universe()
    man = load_manifest()
    doc_to_node = {int(d): int(n) for d, n in zip(man["doc_id"], man["node_id"])}
    edges = load_edges(resolved_only=True)

    graph = CitationGraph.from_edges(edges, len(uni), doc_to_node, weighting)
    if verbose:
        print(f"graph: {graph.n_nodes:,} nodes, {graph.n_edges:,} distinct edges "
              f"({graph.parallel_collapsed:,} repeat citations folded in, "
              f"{graph.dropped_self:,} self-citations dropped)", file=sys.stderr)

    with timed("power iteration"):
        pr, info = pagerank(graph, verbose=verbose)

    bench, provenance = bench_vector(uni, man)
    authority = build_authority(graph, pr, bench,
                                use_treatment=(weighting == "treatment"))

    stats: Dict[str, Any] = {
        "weighting": weighting,
        "system": "S3 (plain PageRank)" if weighting == "uniform" else "S4 (treatment-weighted)",
        "edges_loaded": len(edges),
        "distinct_edges": graph.n_edges,
        "repeat_citations_folded": graph.parallel_collapsed,
        "self_citations_dropped": graph.dropped_self,
        "out_of_range_dropped": graph.dropped_oob,
        "bench_provenance": provenance,
        **{k: v for k, v in info.items() if k != "delta_history"},
        "delta_first5": info["delta_history"][:5],
        "delta_last": info["delta_history"][-1] if info["delta_history"] else None,
    }

    indeg = graph.in_degree()
    stats["nodes_with_in_edges"] = int((indeg > 0).sum())
    stats["in_degree_max"] = int(indeg.max()) if indeg.size else 0
    stats["in_degree_mean_over_cited"] = (float(indeg[indeg > 0].mean())
                                          if (indeg > 0).any() else 0.0)
    stats["in_degree_histogram"] = _histogram(indeg)
    stats["pagerank_skew"] = {
        "min": float(pr.min()), "median": float(np.median(pr)),
        "p99": float(np.percentile(pr, 99)), "max": float(pr.max()),
        "max_over_median": float(pr.max() / np.median(pr)) if np.median(pr) else 0.0,
    }
    stats["top"] = _top_table(uni, pr, indeg, authority, n=20)
    stats["recency_bias"] = _recency_bias(uni, indeg)

    if validate:
        try:
            nx_pr = pagerank_networkx(graph)
            delta = float(np.abs(nx_pr - pr).max())
            stats["networkx_max_abs_diff"] = delta
            stats["networkx_spearman_top100"] = _rank_agreement(pr, nx_pr, 100)
        except Exception as exc:                       # pragma: no cover
            stats["networkx_error"] = repr(exc)
    return authority, stats


def _histogram(indeg: np.ndarray) -> Dict[str, int]:
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49),
            (50, 99), (100, 10 ** 9)]
    out: Dict[str, int] = {}
    for lo, hi in bins:
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}")
        out[label] = int(((indeg >= lo) & (indeg <= hi)).sum())
    return out


def _top_table(uni: pd.DataFrame, pr: np.ndarray, indeg: np.ndarray,
               authority: Dict[int, Authority], n: int = 20) -> List[Dict[str, Any]]:
    order = np.argsort(-pr)[:n]
    title = dict(zip(uni["node_id"], uni["title"]))
    year = dict(zip(uni["node_id"], uni["year"]))
    scr = dict(zip(uni["node_id"], uni["scr_cite"]))
    out = []
    for rank, node in enumerate(order, 1):
        node = int(node)
        a = authority[node]
        out.append({
            "rank": rank, "node_id": node, "year": int(year.get(node, 0)),
            "title": str(title.get(node, "")), "scr_cite": str(scr.get(node, "")),
            "pagerank": float(pr[node]), "in_degree": int(indeg[node]),
            "g": a.g, "bench_norm": a.bench_norm,
        })
    return out


def _recency_bias(uni: pd.DataFrame, indeg: np.ndarray) -> List[Dict[str, Any]]:
    """Citations received per node, by decade.

    Old judgments have had longer to be cited, *but* our citing documents are a
    sample spread evenly over 76 years, so the dominant effect is the opposite
    one: a 2024 judgment can only be cited by the handful of 2024-2025 documents
    we indexed.  Reporting both the count and the per-node rate makes the shape
    of the bias visible instead of implied.
    """
    node_year = np.zeros(len(uni), dtype=np.int32)
    node_year[uni["node_id"].to_numpy()] = uni["year"].to_numpy()
    out = []
    for dec in range(1950, 2030, 10):
        mask = (node_year >= dec) & (node_year < dec + 10)
        nodes = int(mask.sum())
        if not nodes:
            continue
        cites = int(indeg[mask].sum())
        out.append({
            "decade": dec, "nodes": nodes, "citations_received": cites,
            "per_node": round(cites / nodes, 4),
            "share_ever_cited": round(float((indeg[mask] > 0).mean()), 4),
        })
    return out


def _rank_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of the top-k set shared by the two rankings."""
    ta, tb = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
    return round(len(ta & tb) / k, 4)


def write_authority(authority: Dict[int, Authority], stats: Dict[str, Any],
                    out: Optional[Path] = None) -> Path:
    """Persist one authority store.

    Uniform and treatment weightings go to *different* files, so running the S4
    build never destroys the S3 one and ``search(mode=...)`` can switch between
    them without a rebuild.
    """
    config.INDEX.mkdir(parents=True, exist_ok=True)
    weighting = stats.get("weighting", "uniform")
    target = Path(out) if out else authority_path(weighting)
    blob = {
        "header": {k: v for k, v in stats.items()
                   if k not in ("top", "in_degree_histogram", "recency_bias")},
        "nodes": {str(n): asdict(a) for n, a in authority.items()},
    }
    target.write_text(json.dumps(blob), encoding="utf-8")
    stats_path(weighting).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return target


def load_authority(path: Path = None) -> Dict[int, Authority]:
    p = Path(path or AUTHORITY_JSON)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run `python -m nyaya.authority.pagerank build`"
        )
    blob = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): Authority(**v) for k, v in blob["nodes"].items()}


def compare_weightings(uniform: Optional[Path] = None,
                       treatment: Optional[Path] = None,
                       universe=None,
                       top: int = 15) -> Dict[str, Any]:
    """Diff the S3 and S4 stores: which judgments lose authority, and by how much.

    This is the novelty claim in two numbers.  Plain PageRank (S3) rewards a
    judgment for *being cited*, with no regard for what the citing court said
    about it — so a much-cited judgment that has since been overruled keeps its
    high authority and keeps being recommended as live law.  Treatment-weighted
    PageRank (S4) gives an overruling citation weight 0.0
    (``config.TREATMENT_WEIGHTS``) and multiplies the overruled judgment's g(d)
    by ``config.TREATMENT_PENALTY['overruled']``, so it drops.

    Returns the judgments whose status is not ``ok`` under S4, ordered by how
    much g(d) fell, with their titles.
    """
    s3 = load_authority(uniform or AUTHORITY_JSON)
    s4_path = Path(treatment or AUTHORITY_TREATMENT_JSON)
    if not s4_path.exists():
        raise FileNotFoundError(
            f"{s4_path} missing — run "
            "`python -m nyaya.authority.pagerank build --weighting treatment` "
            "after M1's treatment classifier has labelled index/citations.jsonl")
    s4 = load_authority(s4_path)

    if universe is None:
        universe = load_universe()
    title = dict(zip(universe["node_id"], universe["title"]))
    year = dict(zip(universe["node_id"], universe["year"]))

    # How many in-edges actually say "overruled", so the report can show the
    # strength of the evidence behind each status rather than just the verdict.
    # This matters: a judgment flagged on ONE edge out of 181 is far more likely a
    # classifier error than overruled law, and 71% of the flagged judgments rest
    # on exactly one edge.  A share threshold cannot separate the two, because
    # genuinely overruled cases also often have one explicit overruling (Sankari
    # Prasad 1/73, Sabhajit Tewary 1/14), so the honest move is to publish the
    # evidence count next to the verdict instead of silently filtering.
    overruled_edges: Counter = Counter()
    in_edges: Counter = Counter()
    try:
        from nyaya.authority.citations import load_edges
        for e in load_edges(resolved_only=True):
            in_edges[int(e.dst)] += 1
            if (e.treatment or "").strip().lower() == "overruled":
                overruled_edges[int(e.dst)] += 1
    except FileNotFoundError:
        pass

    changed: List[Dict[str, Any]] = []
    status_counts: Counter = Counter()
    for node, a4 in s4.items():
        status_counts[a4.status] += 1
        a3 = s3.get(node)
        if a3 is None:
            continue
        if a4.status == "ok" and abs(a4.g - a3.g) < 1e-12:
            continue
        changed.append({
            "node_id": node,
            "year": int(year.get(node, 0)),
            "title": str(title.get(node, "")),
            "status": a4.status,
            "g_s3": round(a3.g, 4),
            "g_s4": round(a4.g, 4),
            "delta": round(a4.g - a3.g, 4),
            "penalty": a4.penalty,
            "in_degree": a4.in_degree,
            "overruled_edges": int(overruled_edges.get(node, 0)),
            "cited_by": int(in_edges.get(node, 0)),
            "single_edge_evidence": bool(overruled_edges.get(node, 0) == 1),
            "pagerank_s3": a3.pagerank,
            "pagerank_s4": a4.pagerank,
        })
    changed.sort(key=lambda r: r["delta"])
    return {
        "nodes": len(s4),
        "status_counts": dict(status_counts),
        "changed": len(changed),
        "biggest_drops": changed[:top],
        "mean_drop": (sum(r["delta"] for r in changed) / len(changed)
                      if changed else 0.0),
        "overruled_on_one_edge": sum(
            1 for r in changed
            if r["status"] == "overruled" and r["single_edge_evidence"]),
        "overruled_total": sum(1 for r in changed if r["status"] == "overruled"),
    }


def print_comparison(c: Dict[str, Any]) -> None:
    p = print
    p("=" * 78)
    p("S3 (plain PageRank) vs S4 (treatment-weighted) — the g(d) that changed")
    p("=" * 78)
    p(f"nodes                  : {c['nodes']:,}")
    p(f"status under S4        : {c['status_counts']}")
    p(f"judgments whose g moved: {c['changed']:,}  "
      f"(mean change {c['mean_drop']:+.4f})")
    if not c["biggest_drops"]:
        p("")
        p("  nothing moved — every citation is still unlabelled, so S4 and S3")
        p("  are the same graph.  This is expected until M1's classifier runs.")
        p("=" * 78)
        return
    p("")
    p(f"  {'year':>4}  {'cited':>6}{'ovr':>5}  {'status':<14}{'g S3':>7}"
      f"{'g S4':>7}{'delta':>8}  title")
    for r in c["biggest_drops"]:
        flag = " (!)" if r.get("single_edge_evidence") else "    "
        p(f"  {r['year']:>4}  {r.get('cited_by', 0):>6}"
          f"{r.get('overruled_edges', 0):>5}{flag} {r['status']:<10}"
          f"{r['g_s3']:>7.3f}{r['g_s4']:>7.3f}{r['delta']:>+8.3f}  "
          f"{r['title'][:34]}")
    p("")
    p(f"  (!) = flagged on a SINGLE overruling citation: "
      f"{c.get('overruled_on_one_edge', 0)} of {c.get('overruled_total', 0)} "
      f"overruled judgments.")
    p("      One edge against a hundred that say 'followed' is more likely a")
    p("      classifier error than overruled law, so those rows are marked rather")
    p("      than filtered - a share threshold would also discard the genuine")
    p("      single-overruling cases (Sankari Prasad 1/73, Sabhajit Tewary 1/14).")
    p("=" * 78)


# ==========================================================================
# 5. Report / CLI
# ==========================================================================


def print_stats(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 78)
    p(f"S4 part 2 — PageRank and g(d)   [{s['system']}, weighting={s['weighting']}]")
    p("=" * 78)
    p(f"nodes                    : {s['n_nodes']:,}")
    p(f"distinct edges           : {s['distinct_edges']:,}  "
      f"(from {s['edges_loaded']:,} resolved citations; "
      f"{s['repeat_citations_folded']:,} repeats folded, "
      f"{s['self_citations_dropped']:,} self-citations dropped)")
    p(f"dangling (no out-links)  : {s['dangling_nodes']:,}  "
      f"({s['dangling_nodes'] / s['n_nodes']:.2%}) — their mass is redistributed")
    p(f"nodes ever cited         : {s['nodes_with_in_edges']:,}  "
      f"({s['nodes_with_in_edges'] / s['n_nodes']:.2%} of the universe)")
    p(f"power iteration          : {s['iterations']} iters, "
      f"final L1 delta {s['final_delta']:.3e}, converged={s['converged']}, "
      f"mass={s['mass']:.12f}")
    if "networkx_max_abs_diff" in s:
        p(f"networkx cross-check     : max |diff| = {s['networkx_max_abs_diff']:.3e}"
          f"   top-100 set overlap = {s.get('networkx_spearman_top100')}")
    elif "networkx_error" in s:
        p(f"networkx cross-check     : FAILED {s['networkx_error']}")
    bp = s["bench_provenance"]
    p(f"bench for g(d)           : {bp['pdf']:,} nodes from the PDF CORAM line, "
      f"{bp['prior']:,} universe-only nodes at the prior ({bp['prior_value']})")
    sk = s["pagerank_skew"]
    p(f"pagerank skew            : min {sk['min']:.3e}  median {sk['median']:.3e}  "
      f"p99 {sk['p99']:.3e}  max {sk['max']:.3e}  (max/median = {sk['max_over_median']:,.0f}x)")
    p("                           -> min-max on a LOG scale, or the term is a constant")
    p("")
    p("in-degree distribution (citations received)")
    for k, v in s["in_degree_histogram"].items():
        p(f"  {k:>6} : {v:>8,}")
    p("")
    p("top 20 judgments by PageRank")
    p(f"  {'#':>2}  {'year':>4}  {'in':>4}  {'pagerank':>10}  {'g':>5}  title")
    for r in s["top"]:
        p(f"  {r['rank']:>2}  {r['year']:>4}  {r['in_degree']:>4}  "
          f"{r['pagerank']:>10.3e}  {r['g']:>5.3f}  {r['title'][:52]}")
    p("")
    p("citations received by decade (recency / antiquity bias)")
    p(f"  {'decade':>7}  {'nodes':>7}  {'cites':>7}  {'per node':>9}  {'% ever cited':>12}")
    for r in s["recency_bias"]:
        p(f"  {r['decade']:>6}s  {r['nodes']:>7,}  {r['citations_received']:>7,}  "
          f"{r['per_node']:>9.3f}  {r['share_ever_cited']:>11.2%}")
    p("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.authority.pagerank",
                                 description="PageRank and g(d) over the citation graph.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--weighting", choices=list(WEIGHTINGS), default="uniform")
    b.add_argument("--no-validate", action="store_true")
    b.add_argument("--quiet", action="store_true")
    t = sub.add_parser("top")
    t.add_argument("--n", type=int, default=20)
    sub.add_parser("validate")
    st = sub.add_parser("stats")
    st.add_argument("--weighting", choices=list(WEIGHTINGS), default="uniform")
    cp = sub.add_parser("compare",
                        help="diff the S3 and S4 stores (the g(d) deltas)")
    cp.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    if args.command == "build":
        authority, stats = build(weighting=args.weighting,
                                 validate=not args.no_validate,
                                 verbose=not args.quiet)
        out = write_authority(authority, stats)
        print(f"wrote {len(authority):,} authority records -> {out}")
        print_stats(stats)
    elif args.command == "compare":
        print_comparison(compare_weightings(top=args.top))
    elif args.command in ("stats", "top"):
        sp = stats_path(getattr(args, "weighting", "uniform"))
        if not sp.exists():
            raise SystemExit(f"no {sp.name} — run `build` first")
        s = json.loads(sp.read_text(encoding="utf-8"))
        if args.command == "stats":
            print_stats(s)
        else:
            for r in s["top"][:args.n]:
                print(f"{r['rank']:>3}  {r['year']}  in={r['in_degree']:>4}  "
                      f"pr={r['pagerank']:.3e}  g={r['g']:.3f}  {r['title'][:70]}")
    else:
        _, stats = build(validate=True, verbose=False)
        print(f"networkx max |diff| = {stats.get('networkx_max_abs_diff')}")
        print(f"top-100 overlap     = {stats.get('networkx_spearman_top100')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
