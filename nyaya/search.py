"""`search()` — the single entry point everything is measured through.

Why one function
----------------
The evaluation harness and the user interface call **this**.  If the harness
measured a different code path from the one the demo shows, every number in the
report would describe a system nobody can use.  So the six modes are not six
pipelines; they are one pipeline with four switches:

====  ===========================  =======  ======  ==========  =========
mode  text scorer                  index    zones   authority   prox
====  ===========================  =======  ======  ==========  =========
B0    lnc.ltc tf-idf               baseline flat    no          no
S1    uniform-zone cosine          full     flat    no          no
S2    weighted-zone cosine         full     tuned   no          no
S2p   BM25F                        full     tuned   no          no
S3    BM25F                        full     tuned   plain PR    no
S4    BM25F                        full     tuned   treatment   yes
====  ===========================  =======  ======  ==========  =========

``B0`` is the only one that reads a different index on disk: it is the
no-canonicalisation, no-legal-stop-list, unzoned build in ``index/baseline/``.
``S1`` is the full index scored with flat zone weights — a third index would
have been a waste of 700 MB to express a scoring choice.

The two paths through a query
-----------------------------
* **Free text** (no operators, no filters): the candidate set is the union of the
  query terms' postings, scored and heaped.  This is the overwhelming majority of
  real queries and it never touches the Boolean machinery.
* **Operators or filters**: M2's planner executes the tree rarest-first, hands
  back a candidate array, and the same scorer runs over it.

A **filters-only** query (``act:NDPS type:"criminal appeal"``) has no text to
score at all.  Ranking it by a text score of zero would order it arbitrarily, so
it ranks by authority alone and the explanation says so.

Author: Sumedha (M3)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import FULL_MODE, MODES, Result, ScoreExplanation
from nyaya.query.planner import SearchContext

__all__ = ["search", "SearchContext", "load_context", "MODE_SETTINGS",
           "SearchReport"]


# ==========================================================================
# Mode table — one row per system, so an ablation is a lookup, not an if-tree
# ==========================================================================


@dataclass(frozen=True)
class ModeSettings:
    name: str
    scorer: str                  # 'lnc.ltc' | 'uniform-zone' | 'weighted-zone' | 'bm25f'
    baseline_index: bool = False
    authority: bool = False
    treatment: bool = False      # treatment-aware g(d) rather than plain PageRank
    proximity: bool = False
    description: str = ""

    @property
    def alpha(self) -> float:
        """With no authority, all the weight is on text."""
        return 1.0 if not self.authority else config.ALPHA

    @property
    def beta(self) -> float:
        return config.BETA if self.proximity else 0.0


MODE_SETTINGS: Dict[str, ModeSettings] = {
    "B0": ModeSettings("B0", "lnc.ltc", baseline_index=True,
                       description="flat tf-idf lnc.ltc over the plain-tokenizer "
                                   "index: no statute canonicalisation, no legal "
                                   "stop list, no zones, no authority"),
    "S1": ModeSettings("S1", "uniform-zone",
                       description="B0 + statute canonicalisation + the measured "
                                   "legal stop list, still unweighted"),
    "S2": ModeSettings("S2", "weighted-zone",
                       description="S1 + weighted zone scoring"),
    "S2p": ModeSettings("S2p", "bm25f",
                        description="S1 + BM25F instead of weighted-zone cosine"),
    "S3": ModeSettings("S3", "bm25f", authority=True,
                       description="S2p + static authority from plain PageRank"),
    "S4": ModeSettings("S4", "bm25f", authority=True, treatment=True,
                       proximity=True,
                       description="S3 + treatment-aware g(d) + proximity "
                                   "(the full system)"),
}

_SCORERS = {}


def _scorer(name: str):
    if not _SCORERS:
        from nyaya.rank import bm25f, tfidf, zonescore

        _SCORERS.update({
            "lnc.ltc": tfidf.score,
            "uniform-zone": zonescore.uniform,
            "weighted-zone": zonescore.score,
            "bm25f": bm25f.score,
        })
    return _SCORERS[name]


# ==========================================================================
# Context
# ==========================================================================


@dataclass
class SearchReport:
    """What the query did — for the CLI, the UI and the efficiency table."""

    mode: str = "S4"
    path: str = "free-text"           # 'free-text' | 'boolean' | 'filters-only'
    candidates: int = 0
    scored: int = 0
    returned: int = 0
    elapsed_ms: float = 0.0
    parse_ms: float = 0.0
    plan_ms: float = 0.0
    score_ms: float = 0.0
    scorer: str = ""
    warnings: List[str] = field(default_factory=list)
    plan: Any = None
    plan_stats: Any = None
    rank_report: Any = None

    def as_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("plan", "plan_stats", "rank_report")}
        d["plan_stats"] = self.plan_stats.as_json() if self.plan_stats else None
        d["rank_report"] = self.rank_report.as_json() if self.rank_report else None
        return d


_CONTEXTS: Dict[str, SearchContext] = {}


def load_context(mode: str = "S4", reload: bool = False) -> SearchContext:
    """Load S1-S5 once and cache per index path.

    Cached because the UI and the harness both call `search()` in a loop and the
    stores are hundreds of megabytes; the postings are memory-mapped, so holding
    the context open costs almost no resident memory.
    """
    from nyaya.authority.pagerank import authority_path

    settings = MODE_SETTINGS.get(mode, MODE_SETTINGS["S4"])
    # S3 and S4 differ in which authority store they read: plain PageRank versus
    # the treatment-weighted one.  That is the S3 -> S4 ablation, so the cache
    # key has to carry it or the second mode would silently reuse the first's g.
    weighting = "treatment" if settings.treatment else "uniform"
    if weighting == "treatment" and not authority_path("treatment").exists():
        weighting = "uniform"
    key = ("baseline" if settings.baseline_index else "full", weighting)
    if not reload and key in _CONTEXTS:
        ctx = _CONTEXTS[key]
        ctx.mode = mode
        return ctx

    from nyaya.index.positional import PositionalIndex

    path = (config.INDEX / "baseline") if settings.baseline_index else config.INDEX
    ctx = SearchContext(index=PositionalIndex.load(path), mode=mode)

    try:
        from nyaya.index.parametric import ParametricIndex

        ctx.params = ParametricIndex.load()
    except Exception:                                        # pragma: no cover
        ctx.params = None
    try:
        from nyaya.index.paragraphs import ParagraphStore

        ctx.paras = ParagraphStore.load() if ParagraphStore.exists() else None
    except Exception:                                        # pragma: no cover
        ctx.paras = None

    from nyaya.authority.gscore import AuthorityStore
    from nyaya.rank.net import DocNodeMap

    ctx.docnode = DocNodeMap.from_manifest()
    # Two views of S4, deliberately both present:
    #   ctx.authority  the raw {node_id: Authority} dict, which is what M2's
    #                  net.g_of_docs reads - his convention, not changed here;
    #   ctx.gscore     the doc-id-addressed AuthorityStore, which is what the
    #                  badges and the "why this result" panel read.
    apath = authority_path(weighting)
    try:
        ctx.gscore = (AuthorityStore.load(apath, mapping=ctx.docnode)
                      if apath.exists() else AuthorityStore.empty())
    except Exception:                                        # pragma: no cover
        ctx.gscore = AuthorityStore.empty()
    ctx.authority = ctx.gscore.nodes
    ctx.authority_weighting = weighting
    if settings.treatment and weighting != "treatment":
        ctx.authority_note = (
            "S4 wants the treatment-weighted authority store, but "
            f"{authority_path('treatment').name} does not exist yet, so it is "
            "reading the plain-PageRank one - S3 and S4 will rank identically "
            "until M1's treatment classifier and a `--weighting treatment` "
            "PageRank run land")
    else:
        ctx.authority_note = ""
    try:
        from nyaya.metadata import load_manifest

        ctx.docs = load_manifest()
    except Exception:                                        # pragma: no cover
        ctx.docs = None
    ctx.tiers = None                      # S5 champion lists, when M2 lands them
    _CONTEXTS[key] = ctx
    return ctx


# ==========================================================================
# Candidate selection
# ==========================================================================


def _free_text_candidates(terms: Sequence[str], ctx: SearchContext) -> np.ndarray:
    """Union of the query terms' postings — the ranked-retrieval candidate set.

    Union, not intersection: ranked retrieval must not require every term, or a
    five-word question returns nothing.  `net.rank`'s index elimination then
    trims the union to documents matching at least half the terms when the query
    is long, which is where the efficiency comes back.
    """
    index = ctx.index
    lists = []
    for t in terms:
        p = index.postings(t)
        if p is not None:
            lists.append(np.asarray(p.doc_ids, dtype=np.int32))
    if not lists:
        return np.empty(0, dtype=np.int32)
    if len(lists) == 1:
        return lists[0]
    return np.unique(np.concatenate(lists))


# ==========================================================================
# The entry point
# ==========================================================================


def search(query: str, k: int = config.TOP_K, mode: str = FULL_MODE,
           explain: bool = True, ctx: Optional[SearchContext] = None,
           report: Optional[SearchReport] = None) -> List[Result]:
    """Run one query end to end and return the top ``k`` `Result` objects.

    ``report``, if given, is filled in place with timings, the plan and the
    candidate counts — the CLI and the efficiency table read it.
    """
    from nyaya.query.parser import parse
    from nyaya.rank import net

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    settings = MODE_SETTINGS[mode]
    rep = report if report is not None else SearchReport()
    rep.mode = mode
    t_start = time.perf_counter()

    ctx = ctx or load_context(mode)
    ctx.mode = mode

    t0 = time.perf_counter()
    q = parse(query)
    rep.parse_ms = (time.perf_counter() - t0) * 1000.0
    rep.warnings = list(getattr(q, "warnings", []))
    note = getattr(ctx, "authority_note", "")
    if note and settings.treatment:
        rep.warnings.append(note)

    terms = list(q.free_terms)

    # ---- candidates -----------------------------------------------------
    t0 = time.perf_counter()
    plan = stats = None
    if q.is_free_text and not q.filters:
        rep.path = "free-text"
        cands = _free_text_candidates(terms, ctx)
    else:
        from nyaya.query.planner import plan_query

        rep.path = "filters-only" if not terms else "boolean"
        cands, plan, stats = plan_query(q, ctx)
    rep.plan_ms = (time.perf_counter() - t0) * 1000.0
    rep.plan, rep.plan_stats = plan, stats
    rep.candidates = int(np.asarray(cands).size)

    if rep.candidates == 0:
        rep.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return []

    # ---- score ----------------------------------------------------------
    t0 = time.perf_counter()
    score_fn = _scorer(settings.scorer)
    rep.scorer = settings.scorer
    gscore = ctx.gscore if settings.authority else None

    prox = None
    if settings.proximity and q.root is not None:
        prox = _proximity_signal(q, cands, ctx)

    if not terms:
        # Filters only: nothing to score.  Rank by authority alone and say so.
        rep.path = "filters-only"
        g = (gscore.g_array(cands) if gscore is not None
             else np.zeros(np.asarray(cands).size))
        ranked = net.top_k(cands, g, k)
        rank_report = None
        text_raw = {int(d): 0.0 for d, _ in ranked}
        text_norm = dict(text_raw)
        alpha_used, beta_used = 0.0, 0.0
    else:
        # net.rank reads ctx.authority (the node dict); the mode table decides
        # whether authority counts at all through alpha, which is 1.0 when it
        # does not - so g is computed and then weighted to zero, rather than
        # the code path forking.
        ranked, rank_report = net.rank(
            cands, terms, ctx, k=k, score_fn=score_fn,
            mapping=getattr(ctx, "docnode", None), prox=prox,
            alpha=settings.alpha, beta=settings.beta)
        alpha_used, beta_used = settings.alpha, settings.beta
        ids = np.asarray([d for d, _ in ranked], dtype=np.int32)
        raw = np.asarray(score_fn(ids, terms, ctx), dtype=np.float64) if ids.size \
            else np.empty(0)
        # the same max-normalisation net.rank applied, recomputed on the top-k
        # so the panel's text_norm matches the net score it printed
        allraw = np.asarray(score_fn(np.asarray(cands, dtype=np.int32), terms, ctx),
                            dtype=np.float64)
        top = float(allraw.max()) if allraw.size and allraw.max() > 0 else 1.0
        text_raw = {int(d): float(r) for d, r in zip(ids.tolist(), raw.tolist())}
        text_norm = {d: r / top for d, r in text_raw.items()}
    rep.score_ms = (time.perf_counter() - t0) * 1000.0
    rep.rank_report = rank_report
    rep.scored = int(getattr(rank_report, "scored", rep.candidates))

    # ---- results --------------------------------------------------------
    out: List[Result] = []
    for doc_id, net_value in ranked:
        r = _result(int(doc_id), float(net_value), ctx, settings)
        if explain:
            r.explanation = _explain_one(
                int(doc_id), np.asarray(cands, dtype=np.int32), terms, ctx,
                settings, text_raw.get(int(doc_id), 0.0),
                text_norm.get(int(doc_id), 0.0), prox, q,
                alpha_used, beta_used)
            if r.explanation is not None:
                r.score = float(r.explanation.net)
        out.append(r)

    rep.returned = len(out)
    rep.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return out


def _proximity_signal(q, cands: np.ndarray, ctx: SearchContext
                      ) -> Optional[np.ndarray]:
    """1.0 for candidates that satisfy the query's proximity node, else 0.

    A blunt signal on purpose: `config.BETA` is 0.05, so proximity is a tie-break
    between documents the text scorer already likes, not a ranking of its own.
    """
    from nyaya.contracts import ProxNode

    nodes = _find_prox(q.root)
    if not nodes:
        return None
    ids = np.asarray(cands, dtype=np.int32)
    hit = np.zeros(ids.size, dtype=np.float64)
    from nyaya.query import planner

    for node in nodes:
        try:
            matched = planner.execute(node, ctx)
        except Exception:                                    # pragma: no cover
            continue
        if matched is None or np.asarray(matched).size == 0:
            continue
        mask = np.isin(ids, np.asarray(matched, dtype=np.int32))
        hit = np.maximum(hit, mask.astype(np.float64))
    return hit


def _find_prox(node) -> List[Any]:
    from nyaya.contracts import ProxNode

    if node is None:
        return []
    if isinstance(node, ProxNode):
        return [node]
    out: List[Any] = []
    for c in node.children():
        out.extend(_find_prox(c))
    return out


def _prox_window(q) -> int:
    nodes = _find_prox(q.root)
    return int(nodes[0].k) if nodes else 0


def _prox_kind(q) -> str:
    nodes = _find_prox(q.root)
    return str(nodes[0].kind) if nodes else ""


def _explain_one(doc_id: int, cands: np.ndarray, terms: Sequence[str],
                 ctx: SearchContext, settings: ModeSettings,
                 text_raw: float, text_norm: float,
                 prox: Optional[np.ndarray], q, alpha: float, beta: float
                 ) -> Optional[ScoreExplanation]:
    from nyaya.rank.explain import ExplainInputs, build

    p = 0.0
    if prox is not None:
        where = np.nonzero(np.asarray(cands, dtype=np.int32) == doc_id)[0]
        if where.size:
            p = float(prox[int(where[0])])
    return build(ExplainInputs(
        doc_id=doc_id, doc_ids=cands, terms=terms, ctx=ctx,
        scorer=settings.scorer, text_raw=text_raw, text_norm=text_norm,
        alpha=alpha, beta=beta, proximity=p, prox_window=_prox_window(q),
        prox_kind=_prox_kind(q),
        authority=ctx.gscore if settings.authority else None,
        use_authority=settings.authority,
        expansions=dict(getattr(q, "expansions", {}) or {}),
    ))


def _result(doc_id: int, score: float, ctx: SearchContext,
            settings: ModeSettings) -> Result:
    docs = getattr(ctx, "docs", None)
    row = None
    if docs is not None and 0 <= doc_id < len(docs):
        row = docs.iloc[doc_id]
    auth = getattr(ctx, "gscore", None) if settings.authority else None
    judges = []
    if row is not None:
        j = row.get("judges")
        judges = [str(x) for x in (j if j is not None else [])][:5]
    # bench size: prefer M2's parametric column, which is M1's PDF CORAM count.
    # The manifest's `bench_size_meta` is the metadata one, and metadata carries
    # a single judge name for 87% of rows - taking it would print "bench 1" for
    # a nine-judge Constitution bench.
    bench = 0
    params = getattr(ctx, "params", None)
    if params is not None:
        try:
            col = params.column("bench")
            if col is not None and 0 <= doc_id < len(col):
                bench = int(col[doc_id])
        except Exception:                                    # pragma: no cover
            bench = 0
    if not bench and row is not None:
        bench = int(row.get("bench_size_meta") or 0)

    return Result(
        doc_id=doc_id,
        title=str(row["title"]) if row is not None else f"doc {doc_id}",
        year=int(row["year"]) if row is not None else 0,
        score=float(score),
        bench_size=bench,
        judges=judges,
        neutral_cite=str(row.get("neutral_cite") or "") if row is not None else "",
        scr_cite=str(row.get("scr_cite") or "") if row is not None else "",
        case_no="",
        disposal=str(row.get("disposal") or "") if row is not None else "",
        status=auth.status(doc_id) if auth is not None else "ok",
        badge=auth.badge(doc_id) if auth is not None else "",
    )


def snippets_for(result: Result, terms: Sequence[str], ctx: SearchContext,
                 limit: int = 2) -> List[str]:
    """Evidence passages from S1, with an honest paragraph citation.

    M1 owns the highlighting; this is the plain version the CLI uses so the demo
    has evidence on screen today.  The citation comes from `ParaRef`, so a
    judgment with no printed numbering says "window 14 of 32" and never "¶ 14".
    """
    store = getattr(ctx, "paras", None)
    if store is None or not store.has(result.doc_id):
        return []
    wanted = {t for t in terms}
    best: List[Tuple[int, str]] = []
    for para in store.paras_of(result.doc_id):
        from nyaya.analyze import analyze_terms

        got = set(analyze_terms(para.text))
        overlap = len(wanted & got)
        if overlap:
            ref = store.ref_for(result.doc_id, para)
            best.append((overlap, f"[{ref.cite()}] {para.text[:260].strip()}"))
    best.sort(key=lambda kv: -kv[0])
    return [s for _, s in best[:limit]]
