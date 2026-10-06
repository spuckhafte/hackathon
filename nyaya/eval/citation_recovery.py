"""Track B — citation recovery: automatic, large-scale, no hand judging.

Member 2 (Suvrat).  The idea: a judgment's own citations are a free relevance
judgment.  When the Supreme Court decided case X in 2017 and cited Y, Z and W,
then given the *facts* of X a good precedent-search engine ought to return Y, Z
and W.  Nobody has to label anything, so this scales to hundreds of queries
instead of the handful Track A can pool by hand.

Everything goes through ``nyaya.search.search()`` — the same function the CLI and
the UI call — so these numbers describe the shipped system, not a parallel
pipeline built for the evaluation.

**Three ways this evaluation could lie, and what is done about each.**

1. *Leakage.*  The query is built from the judgment's own text, which contains the
   citations we are trying to recover and the names of the cited parties.  Left
   in, every system would score absurdly well for the wrong reason.  So every
   reporter citation and every ``X v. Y`` case name is masked out, **and the
   masking is verified**: :func:`audit_leakage` re-checks each finished query
   against each of its gold judgments' discriminative title tokens and reports
   how many survived.  A number, not a hope.
2. *Unretrievable gold.*  A cited precedent that was never sampled into the
   13,942-document index cannot be returned by any system, so counting it as a
   miss measures the sampling policy rather than the ranking.  Gold is therefore
   restricted to precedents that are in the index, and the excluded fraction is
   reported — it is **46.4%** of all resolved citations, which is a large and
   honest limitation of the corpus policy, not of the retrieval.
3. *The tautology.*  Authority-based systems (S3, S4) rank by how often a
   judgment is cited, and the gold set is *defined* by citations.  Popular cases
   are cited more often, so S3/S4 beating B0 is partly circular.  This is
   quantified rather than hand-waved: :data:`POPULARITY_BASELINE` ("G0") ranks by
   ``g(d)`` **alone, ignoring the query entirely**.  Whatever recall G0 reaches is
   the share obtainable with no relevance signal at all, and it is the floor that
   S3/S4's numbers have to be read against.

Also: being cited is not the same as being the only relevant precedent.  A system
can return an excellent case the judgment did not happen to cite and be counted
wrong.  Recall here is therefore a *lower bound* on real recall, which is stated
in the output, not just here.

CLI::

    python -m nyaya.eval.citation_recovery sample [--n 200]
    python -m nyaya.eval.citation_recovery run [--n 200] [--modes B0,S4]
    python -m nyaya.eval.citation_recovery report
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from nyaya import config
from nyaya.utils import ensure_utf8_stdout, timed

RUNS = config.RUNS
QUERIES_JSON = RUNS / "citation_recovery_queries.json"
RESULTS_JSON = RUNS / "citation_recovery.json"
RESULTS_CSV = RUNS / "citation_recovery.csv"
PER_QUERY_CSV = RUNS / "citation_recovery_per_query.csv"

#: Judgments decided before this are excluded as queries: the point is to search
#: for old precedent from a modern judgment, and a 1955 query has almost no
#: earlier indexed corpus to search.
MIN_QUERY_YEAR = 2001

#: A query judgment must cite at least this many *indexed* precedents, or
#: recall@k is measured on one or two documents and the variance swamps the
#: signal.
MIN_GOLD = 3

#: Fraction of the judgment's paragraphs used to build the query.  The first 20%
#: is roughly the facts and the issue — what a lawyer would have in hand *before*
#: finding the precedent, which is the situation we are simulating.
FACTS_FRACTION = 0.20

#: Query length cap.  A 3,000-word query is not a query, and the long tail is all
#: low-idf words that index elimination would drop anyway.
MAX_QUERY_TERMS = 60

#: Retrieve this deep, then drop the judgment itself and anything decided on or
#: after its date, then cut to 10 / 50.  Deep enough that the post-filter cannot
#: starve the top 50.
RETRIEVE_DEPTH = 300

#: The non-query baseline that quantifies the authority tautology.
POPULARITY_BASELINE = "G0"

SYSTEMS = ("B0", "S1", "S2", "S2p", "S3", "S4")

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = config.SAMPLE_SEED


# ==========================================================================
# 1. Metrics — Rakshit owns eval/metrics.py; use his when it exists
# ==========================================================================
# TODO(M1): `nyaya/eval/metrics.py` is yours.  Recall@k and MRR are not in the
# P@k / MAP / nDCG set you listed, so they are defined here.  If you add them,
# delete these two and import yours instead — two implementations that disagree
# is exactly the trap we are avoiding.  `_import_metrics()` already prefers
# yours for anything whose name matches.


def _import_metrics():
    try:
        from nyaya.eval import metrics as m
        return m
    except Exception:
        return None


def recall_at_k(ranked: Sequence[int], gold: Set[int], k: int) -> float:
    """Share of the gold set found in the first ``k`` results.

    Gold is per-query and small (3-20 documents), so this is recall in the plain
    sense; it is not averaged precision and must not be read as one.
    """
    if not gold:
        return 0.0
    m = _import_metrics()
    if m is not None and hasattr(m, "recall_at_k"):
        return float(m.recall_at_k(ranked, gold, k))
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: Sequence[int], gold: Set[int]) -> float:
    """1/rank of the first gold document, 0 if none is retrieved."""
    m = _import_metrics()
    if m is not None and hasattr(m, "reciprocal_rank"):
        return float(m.reciprocal_rank(ranked, gold))
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES,
                 alpha: float = 0.05,
                 seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Percentile bootstrap CI for a mean over queries.

    Per-query scores are not normally distributed — recall@10 over a 3-document
    gold set takes values in {0, 1/3, 2/3, 1} — so a t-interval would be wrong.
    Resampling queries makes no distributional assumption.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (0.0, 0.0)
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(samples, v.size))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ==========================================================================
# 2. Masking — the part that decides whether the whole track is honest
# ==========================================================================

MASK = " MASKED "

#: Words that mark the sentence as *about* a precedent rather than about the
#: facts.  A sentence containing one of these is dropped wholesale, because the
#: citation masker only removes the citation string and the party names, and the
#: surrounding words ("relied on the decision in", "this Court held in") can still
#: carry distinctive phrasing from the cited case.
_CITATION_CUES = re.compile(
    r"(?i)\b(relied\s+(?:up)?on|reliance|cited|citing|referred\s+to|"
    r"following\s+the\s+decision|in\s+the\s+case\s+of|this\s+court\s+held\s+in|"
    r"laid\s+down\s+in|reported\s+in|see\s+also|approved\s+in|"
    r"overrul\w+|distinguish\w+|affirm\w+\s+in)\b"
)


def mask_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Remove every citation and case name from a block of judgment text.

    Order matters: citations are masked first (they are unambiguous), then
    ``X v. Y`` names, then any sentence that still carries a citation cue.
    """
    from nyaya.analyze import sentences
    from nyaya.authority.citations import _NAME_RE, find_citations

    counts = {"citations": 0, "names": 0, "cue_sentences": 0}

    # --- 1. reporter citations ------------------------------------------
    masked = text
    spans = [(m.start, m.end) for m in find_citations(masked)]
    for a, b in reversed(spans):
        masked = masked[:a] + MASK + masked[b:]
    counts["citations"] = len(spans)

    # --- 2. "X v. Y" case names -----------------------------------------
    name_spans = [(m.start(), m.end()) for m in _NAME_RE.finditer(masked)]
    for a, b in reversed(name_spans):
        masked = masked[:a] + MASK + masked[b:]
    counts["names"] = len(name_spans)

    # --- 3. whole sentences that discuss a precedent ---------------------
    kept: List[str] = []
    for s in sentences(masked):
        if _CITATION_CUES.search(s) or MASK.strip() in s:
            counts["cue_sentences"] += 1
            continue
        kept.append(s)
    return " ".join(kept), counts


def build_query(paras: Sequence[Dict[str, Any]],
                fraction: float = FACTS_FRACTION,
                max_terms: int = MAX_QUERY_TERMS) -> Tuple[str, Dict[str, int]]:
    """Turn the first ``fraction`` of a judgment's paragraphs into a masked query."""
    body = [p for p in paras if p.get("zone") in ("body", "order")]
    if not body:
        return "", {"citations": 0, "names": 0, "cue_sentences": 0}
    n = max(1, int(round(len(body) * fraction)))
    text = " ".join((p.get("text") or "") for p in body[:n])
    masked, counts = mask_text(text)

    # Reduce to the highest-idf terms: a query is a bag of discriminative words,
    # and keeping 3,000 of them would just measure the stop list.
    from nyaya.query.normalise import terms_of
    from nyaya.stoplist import filter_scoring_terms

    terms = filter_scoring_terms(terms_of(masked))
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in terms:
        if t not in seen and t != "masked":
            seen.add(t)
            uniq.append(t)
    counts["terms_before_cap"] = len(uniq)
    return " ".join(uniq[:max_terms]), counts


# ==========================================================================
# 3. Sampling the queries
# ==========================================================================


@dataclass
class RecoveryQuery:
    doc_id: int
    node_id: int
    year: int
    date: str
    title: str
    query: str
    gold: List[int] = field(default_factory=list)          # doc_ids, in index
    gold_nodes_all: List[int] = field(default_factory=list)  # resolved targets
    gold_excluded_not_indexed: int = 0
    gold_excluded_later: int = 0
    mask_counts: Dict[str, int] = field(default_factory=dict)
    leaked_tokens: List[str] = field(default_factory=list)

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


def _resolved_targets() -> Dict[int, Set[int]]:
    """``{citing doc_id: {cited node_id}}`` from the resolved edges."""
    from nyaya.authority.citations import load_edges

    per: Dict[int, Set[int]] = defaultdict(set)
    for e in load_edges(resolved_only=True):
        if e.dst >= 0:
            per[int(e.src)].add(int(e.dst))
    return per


def sample_queries(n: int = 200, seed: int = config.SAMPLE_SEED,
                   verbose: bool = True) -> List[RecoveryQuery]:
    """Pick ``n`` judgments and build a masked query for each."""
    from nyaya.metadata import load_manifest
    from nyaya.rank.net import DocNodeMap

    man = load_manifest()
    mapping = DocNodeMap.from_manifest(man)
    year = dict(zip(man["doc_id"].tolist(), man["year"].tolist()))
    date = dict(zip(man["doc_id"].tolist(), man["decision_date"].tolist()))
    title = dict(zip(man["doc_id"].tolist(), man["title"].tolist()))

    per = _resolved_targets()
    eligible: List[int] = []
    for src, dsts in per.items():
        if int(year.get(src, 0)) < MIN_QUERY_YEAR:
            continue
        indexed = [d for d in dsts if mapping.doc_of(d) >= 0]
        if len(indexed) >= MIN_GOLD:
            eligible.append(src)
    eligible.sort()
    rng = random.Random(seed)
    chosen = set(rng.sample(eligible, min(n, len(eligible))))
    if verbose:
        print(f"{len(eligible)} eligible judgments; sampling {len(chosen)}",
              file=sys.stderr)

    # One streaming pass over the 1 GB corpus, keeping only the chosen records.
    out: List[RecoveryQuery] = []
    for rec in _iter_selected(chosen):
        doc_id = int(rec["doc_id"])
        q_date = str(date.get(doc_id) or "")
        query, counts = build_query(rec.get("paras", ()))
        if not query:
            continue
        gold: List[int] = []
        excluded_later = 0
        excluded_unindexed = 0
        for node in sorted(per[doc_id]):
            cited_doc = mapping.doc_of(node)
            if cited_doc < 0:
                excluded_unindexed += 1
                continue
            cited_date = str(date.get(cited_doc) or "")
            # A precedent must predate the citing judgment.  A later date means
            # the citation was resolved to the wrong judgment, or the metadata
            # date is wrong; either way it cannot be gold.
            if cited_date and q_date and cited_date >= q_date:
                excluded_later += 1
                continue
            gold.append(cited_doc)
        if len(gold) < MIN_GOLD:
            continue
        out.append(RecoveryQuery(
            doc_id=doc_id, node_id=mapping.node_of(doc_id),
            year=int(year.get(doc_id, 0)), date=q_date,
            title=str(title.get(doc_id, "")), query=query,
            gold=gold, gold_nodes_all=sorted(per[doc_id]),
            gold_excluded_not_indexed=excluded_unindexed,
            gold_excluded_later=excluded_later,
            mask_counts=counts,
        ))
    out.sort(key=lambda q: q.doc_id)
    if verbose:
        print(f"built {len(out)} masked queries", file=sys.stderr)
    return out


def _iter_selected(doc_ids: Set[int]) -> Iterator[Dict[str, Any]]:
    """Stream ``judgments.jsonl``, yielding only the selected documents."""
    path = Path(config.JUDGMENTS_JSONL)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    remaining = set(doc_ids)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not remaining:
                return
            line = line.strip()
            if not line:
                continue
            # Cheap pre-filter: avoid json.loads on 13,700 unwanted records.
            head = line[:40]
            if '"doc_id":' not in head:
                rec = json.loads(line)
            else:
                try:
                    did = int(head.split('"doc_id":')[1].split(",")[0].strip())
                except (IndexError, ValueError):
                    rec = json.loads(line)
                else:
                    if did not in remaining:
                        continue
                    rec = json.loads(line)
            did = int(rec.get("doc_id", -1))
            if did in remaining:
                remaining.discard(did)
                yield rec


# ==========================================================================
# 4. Leakage audit — does the masking actually work?
# ==========================================================================


#: A token shared between a query and a gold title only *identifies* that gold
#: judgment if it is rare.  Measured on the real sample: of 83 distinct shared
#: tokens, 80 were ordinary topical vocabulary — "income", "gujarat", "labour",
#: "high", "supreme", "singh" — words any lawyer describing the facts would use,
#: and which no more leak the answer than the word "contract" does.  Only 3
#: ("asian", "avadhuta", "sudeer") were distinctive party names.  So raw token
#: overlap is the wrong statistic; overlap on a token this rare is the right one.
DISTINCTIVE_DF = 100


def _gold_title_stems(doc_id: int, title_key: Dict[int, str]) -> Set[str]:
    from nyaya.analyze import stem_term

    return {stem_term(t) for t in str(title_key.get(doc_id, "")).split()
            if len(t) > 3}


def audit_leakage(queries: Sequence[RecoveryQuery], index=None,
                  verbose: bool = True) -> Dict[str, Any]:
    """Check each query against its gold judgments' discriminative title tokens.

    If a gold judgment's party name survives in the query, that query is
    answerable by lexical match on the answer itself and every system's score is
    inflated.  So this counts the leaks rather than assuming there are none —
    and it separates the two very different things that look alike:

    * **topical overlap** — the query and a gold title share a common word.
      Expected, harmless, and not removable without destroying the query: a tax
      case's facts are *about* income.
    * **distinctive overlap** — they share a token rarer than
      :data:`DISTINCTIVE_DF` documents, which is effectively a party name.  That
      is a real leak, and :func:`strip_distinctive_leaks` removes it.
    """
    from nyaya.metadata import load_manifest

    if index is None:
        from nyaya.index.positional import PositionalIndex
        index = PositionalIndex.load()

    man = load_manifest()
    title_key = dict(zip(man["doc_id"].tolist(), man["title_key"].tolist()))

    leaked_queries = leaked_pairs = total_pairs = 0
    dist_queries = dist_pairs = 0
    token_df: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    for q in queries:
        qterms = set(q.query.split())
        hits: List[str] = []
        dist_hits: List[str] = []
        for g in q.gold:
            total_pairs += 1
            overlap = qterms & _gold_title_stems(g, title_key)
            if not overlap:
                continue
            leaked_pairs += 1
            hits.extend(sorted(overlap))
            rare = {t for t in overlap
                    if token_df.setdefault(t, index.df(t)) <= DISTINCTIVE_DF}
            if rare:
                dist_pairs += 1
                dist_hits.extend(sorted(rare))
        if hits:
            leaked_queries += 1
            q.leaked_tokens = sorted(set(dist_hits))
        if dist_hits:
            dist_queries += 1
            if len(examples) < 10:
                examples.append({"doc_id": q.doc_id,
                                 "distinctive_tokens": sorted(set(dist_hits))})
    out = {
        "queries": len(queries),
        "gold_pairs": total_pairs,
        # topical (any shared token) — reported for transparency, not a problem
        "queries_with_topical_overlap": leaked_queries,
        "topical_overlap_rate_queries": leaked_queries / max(len(queries), 1),
        "gold_pairs_with_topical_overlap": leaked_pairs,
        "topical_overlap_rate_pairs": leaked_pairs / max(total_pairs, 1),
        # distinctive (rare shared token) — this is the real leak
        "distinctive_df_threshold": DISTINCTIVE_DF,
        "queries_with_distinctive_leak": dist_queries,
        "distinctive_leak_rate_queries": dist_queries / max(len(queries), 1),
        "gold_pairs_with_distinctive_leak": dist_pairs,
        "distinctive_leak_rate_pairs": dist_pairs / max(total_pairs, 1),
        "examples": examples,
        "masked_citations": sum(q.mask_counts.get("citations", 0) for q in queries),
        "masked_names": sum(q.mask_counts.get("names", 0) for q in queries),
        "dropped_cue_sentences": sum(q.mask_counts.get("cue_sentences", 0)
                                     for q in queries),
    }
    if verbose:
        print(f"leakage: topical {leaked_pairs}/{total_pairs} pairs, "
              f"DISTINCTIVE {dist_pairs}/{total_pairs}", file=sys.stderr)
    return out


def strip_distinctive_leaks(queries: Sequence[RecoveryQuery], index=None,
                            verbose: bool = True) -> Dict[str, Any]:
    """Delete rare gold-title tokens from the queries that contain them.

    This is the one place the evaluation edits its own queries, so it is
    deliberately narrow: only tokens that (a) appear in a gold judgment's title
    and (b) occur in at most :data:`DISTINCTIVE_DF` documents.  Common topical
    words are left alone — stripping "income" from a tax query would not remove a
    leak, it would remove the query.
    """
    from nyaya.index.positional import PositionalIndex
    from nyaya.metadata import load_manifest

    index = index or PositionalIndex.load()
    man = load_manifest()
    title_key = dict(zip(man["doc_id"].tolist(), man["title_key"].tolist()))

    stripped_tokens = 0
    touched = 0
    df_cache: Dict[str, int] = {}
    for q in queries:
        qterms = q.query.split()
        rare: Set[str] = set()
        for g in q.gold:
            for t in set(qterms) & _gold_title_stems(g, title_key):
                if df_cache.setdefault(t, index.df(t)) <= DISTINCTIVE_DF:
                    rare.add(t)
        if not rare:
            continue
        kept = [t for t in qterms if t not in rare]
        if kept:                       # never empty a query
            q.query = " ".join(kept)
            stripped_tokens += len(qterms) - len(kept)
            touched += 1
    out = {"queries_edited": touched, "tokens_removed": stripped_tokens}
    if verbose:
        print(f"stripped {stripped_tokens} distinctive tokens from {touched} "
              f"queries", file=sys.stderr)
    return out


# ==========================================================================
# 5. Running the systems
# ==========================================================================


def _eligible_filter(doc_dates: Dict[int, str], q: RecoveryQuery):
    """Keep only results decided strictly before the query judgment, not itself."""
    def ok(doc_id: int) -> bool:
        if doc_id == q.doc_id:
            return False
        d = doc_dates.get(doc_id, "")
        if d and q.date:
            return d < q.date
        return True
    return ok


def run_system(queries: Sequence[RecoveryQuery], mode: str,
               doc_dates: Dict[int, str],
               depth: int = RETRIEVE_DEPTH,
               verbose: bool = True) -> List[Dict[str, Any]]:
    """Run every query through ``search()`` in one mode and score it."""
    from nyaya.search import search

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(queries):
        if verbose and i and i % 25 == 0:
            print(f"  {mode}: {i}/{len(queries)}", file=sys.stderr)
        gold = set(q.gold)
        t0 = time.perf_counter()
        try:
            results = search(q.query, k=depth, mode=mode, explain=False)
        except Exception as exc:                       # never lose the whole run
            rows.append({"doc_id": q.doc_id, "mode": mode, "error": repr(exc),
                         "recall@10": 0.0, "recall@50": 0.0, "mrr": 0.0,
                         "gold": len(gold), "retrieved": 0, "ms": 0.0})
            continue
        ms = (time.perf_counter() - t0) * 1000.0
        keep = _eligible_filter(doc_dates, q)
        ranked = [r.doc_id for r in results if keep(r.doc_id)]
        rows.append({
            "doc_id": q.doc_id, "mode": mode,
            "recall@10": recall_at_k(ranked, gold, 10),
            "recall@50": recall_at_k(ranked, gold, 50),
            "mrr": reciprocal_rank(ranked, gold),
            "gold": len(gold),
            "retrieved": len(ranked),
            "dropped_ineligible": len(results) - len(ranked),
            "ms": ms,
        })
    return rows


def run_popularity_baseline(queries: Sequence[RecoveryQuery],
                            doc_dates: Dict[int, str],
                            depth: int = RETRIEVE_DEPTH) -> List[Dict[str, Any]]:
    """Rank by ``g(d)`` alone, ignoring the query — the tautology floor.

    This is deliberately **not** a retrieval system: it never reads the query.
    Whatever recall it reaches is obtainable with no relevance signal at all, so
    it is the number S3 and S4 have to be read against.  If G0 already recovers a
    third of the gold, then a third of S3's recall is popularity, not matching.
    """
    from nyaya.authority.pagerank import load_authority
    from nyaya.metadata import load_manifest
    from nyaya.rank.net import DocNodeMap

    man = load_manifest()
    mapping = DocNodeMap.from_manifest(man)
    authority = load_authority()
    docs = man["doc_id"].to_numpy()
    g = np.array([
        (authority[mapping.node_of(int(d))].g
         if mapping.node_of(int(d)) in authority else 0.0)
        for d in docs
    ])
    order = docs[np.argsort(-g)]

    rows: List[Dict[str, Any]] = []
    for q in queries:
        gold = set(q.gold)
        keep = _eligible_filter(doc_dates, q)
        ranked = [int(d) for d in order.tolist() if keep(int(d))][:depth]
        rows.append({
            "doc_id": q.doc_id, "mode": POPULARITY_BASELINE,
            "recall@10": recall_at_k(ranked, gold, 10),
            "recall@50": recall_at_k(ranked, gold, 50),
            "mrr": reciprocal_rank(ranked, gold),
            "gold": len(gold), "retrieved": len(ranked),
            "dropped_ineligible": 0, "ms": 0.0,
        })
    return rows


# ==========================================================================
# 6. Aggregation
# ==========================================================================

METRICS = ("recall@10", "recall@50", "mrr")


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    out: Dict[str, Any] = {}
    for mode, rs in by_mode.items():
        entry: Dict[str, Any] = {"queries": len(rs),
                                 "errors": sum(1 for r in rs if r.get("error"))}
        for metric in METRICS:
            vals = [float(r[metric]) for r in rs]
            lo, hi = bootstrap_ci(vals)
            entry[metric] = float(np.mean(vals)) if vals else 0.0
            entry[f"{metric}_ci_lo"] = lo
            entry[f"{metric}_ci_hi"] = hi
        entry["mean_ms"] = float(np.mean([r["ms"] for r in rs])) if rs else 0.0
        entry["mean_gold"] = float(np.mean([r["gold"] for r in rs])) if rs else 0.0
        out[mode] = entry
    return out


def corpus_caveats() -> Dict[str, Any]:
    """The structural limits, measured rather than asserted."""
    from nyaya.rank.net import DocNodeMap

    mapping = DocNodeMap.from_manifest()
    per = _resolved_targets()
    all_targets = [d for dsts in per.values() for d in dsts]
    in_index = [d for d in all_targets if mapping.doc_of(d) >= 0]
    return {
        "resolved_citations": len(all_targets),
        "targets_in_the_index": len(in_index),
        "fraction_gold_retrievable": len(in_index) / max(len(all_targets), 1),
        "fraction_gold_excluded": 1 - len(in_index) / max(len(all_targets), 1),
        "note": (
            "A cited precedent that was never sampled into the 13,942-document "
            "index cannot be retrieved by any system, so it is excluded from gold "
            "rather than counted as a miss.  That is a limitation of the "
            "stratified corpus policy (PER_YEAR_CAP=200), not of the ranking."
        ),
    }


def run(n: int = 200, modes: Sequence[str] = SYSTEMS,
        include_baseline: bool = True,
        verbose: bool = True) -> Dict[str, Any]:
    from nyaya.metadata import load_manifest

    queries = load_queries() if QUERIES_JSON.exists() else None
    if queries is None or len(queries) < n:
        queries = sample_queries(n, verbose=verbose)
        save_queries(queries)
    queries = queries[:n]

    # Remove genuine answer leaks BEFORE running anything, then re-audit so the
    # reported figure describes the queries the systems actually saw.
    stripping = strip_distinctive_leaks(queries, verbose=verbose)
    leakage = audit_leakage(queries, verbose=verbose)
    leakage["stripping"] = stripping

    man = load_manifest()
    doc_dates = {int(d): str(x) for d, x in zip(man["doc_id"].tolist(),
                                                man["decision_date"].tolist())}

    rows: List[Dict[str, Any]] = []
    for mode in modes:
        if verbose:
            print(f"running {mode} over {len(queries)} queries", file=sys.stderr)
        rows.extend(run_system(queries, mode, doc_dates, verbose=verbose))
    if include_baseline:
        rows.extend(run_popularity_baseline(queries, doc_dates))

    summary = {
        "track": "B (citation recovery)",
        "queries": len(queries),
        "min_query_year": MIN_QUERY_YEAR,
        "min_gold": MIN_GOLD,
        "facts_fraction": FACTS_FRACTION,
        "max_query_terms": MAX_QUERY_TERMS,
        "retrieve_depth": RETRIEVE_DEPTH,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "modes": list(modes) + ([POPULARITY_BASELINE] if include_baseline else []),
        "by_mode": aggregate(rows),
        "leakage": leakage,
        "caveats": corpus_caveats(),
        "interpretation": [
            "Recall is a LOWER BOUND on real recall: being cited is not the same "
            "as being the only relevant precedent, so a system can return an "
            "excellent uncited case and be scored wrong.",
            "S3/S4 have a structural advantage because the gold set is built "
            f"from citations and authority ranks by citations.  The "
            f"{POPULARITY_BASELINE} row ranks by g(d) alone, ignoring the query, "
            "and is the floor those numbers must be read against.",
        ],
        "rows": rows,
    }
    return summary


# ==========================================================================
# 7. Persistence and reporting
# ==========================================================================


def save_queries(queries: Sequence[RecoveryQuery], path: Path = None) -> Path:
    p = Path(path or QUERIES_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([q.as_json() for q in queries], ensure_ascii=False),
                 encoding="utf-8")
    return p


def load_queries(path: Path = None) -> List[RecoveryQuery]:
    p = Path(path or QUERIES_JSON)
    blob = json.loads(p.read_text(encoding="utf-8"))
    return [RecoveryQuery(**d) for d in blob]


def write_results(summary: Dict[str, Any]) -> List[Path]:
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stable key names: the report and the video's data.ts read these directly.
    lines = ["mode,queries,recall_at_10,recall_at_10_ci_lo,recall_at_10_ci_hi,"
             "recall_at_50,recall_at_50_ci_lo,recall_at_50_ci_hi,"
             "mrr,mrr_ci_lo,mrr_ci_hi,mean_gold,mean_ms"]
    for mode in summary["modes"]:
        e = summary["by_mode"].get(mode)
        if not e:
            continue
        lines.append(",".join(str(x) for x in [
            mode, e["queries"],
            f"{e['recall@10']:.6f}", f"{e['recall@10_ci_lo']:.6f}",
            f"{e['recall@10_ci_hi']:.6f}",
            f"{e['recall@50']:.6f}", f"{e['recall@50_ci_lo']:.6f}",
            f"{e['recall@50_ci_hi']:.6f}",
            f"{e['mrr']:.6f}", f"{e['mrr_ci_lo']:.6f}", f"{e['mrr_ci_hi']:.6f}",
            f"{e['mean_gold']:.2f}", f"{e['mean_ms']:.2f}",
        ]))
    RESULTS_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pq = ["doc_id,mode,recall_at_10,recall_at_50,mrr,gold,retrieved,ms"]
    for r in summary["rows"]:
        pq.append(",".join(str(x) for x in [
            r["doc_id"], r["mode"], f"{r['recall@10']:.6f}",
            f"{r['recall@50']:.6f}", f"{r['mrr']:.6f}", r["gold"],
            r["retrieved"], f"{r['ms']:.2f}"]))
    PER_QUERY_CSV.write_text("\n".join(pq) + "\n", encoding="utf-8")
    return [RESULTS_JSON, RESULTS_CSV, PER_QUERY_CSV]


def print_report(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 82)
    p(f"Track B — citation recovery   ({s['queries']} queries, "
      f"judgments from {s['min_query_year']}+, >= {s['min_gold']} indexed gold)")
    p("=" * 82)
    p(f"query = first {s['facts_fraction']:.0%} of paragraphs, citations and case "
      f"names masked, top {s['max_query_terms']} terms")
    p("")
    p(f"  {'mode':<5}{'R@10':>7}{'95% CI':>16}{'R@50':>7}{'95% CI':>16}"
      f"{'MRR':>7}{'ms':>8}")
    for mode in s["modes"]:
        e = s["by_mode"].get(mode)
        if not e:
            continue
        ci10 = f"[{e['recall@10_ci_lo']:.3f},{e['recall@10_ci_hi']:.3f}]"
        ci50 = f"[{e['recall@50_ci_lo']:.3f},{e['recall@50_ci_hi']:.3f}]"
        star = "  <- no query signal" if mode == POPULARITY_BASELINE else ""
        p(f"  {mode:<5}{e['recall@10']:>7.3f}{ci10:>16}{e['recall@50']:>7.3f}"
          f"{ci50:>16}{e['mrr']:>7.3f}{e['mean_ms']:>8.0f}{star}")
    p("")
    lk = s["leakage"]
    p("masking audit (does the query leak the answer?)")
    p(f"  citations masked         : {lk['masked_citations']:,}")
    p(f"  case names masked        : {lk['masked_names']:,}")
    p(f"  cue sentences dropped    : {lk['dropped_cue_sentences']:,}")
    st = lk.get("stripping") or {}
    p(f"  distinctive tokens stripped: {st.get('tokens_removed', 0)} from "
      f"{st.get('queries_edited', 0)} queries")
    p(f"  topical overlap (harmless): "
      f"{lk['gold_pairs_with_topical_overlap']}/{lk['gold_pairs']} gold pairs "
      f"({lk['topical_overlap_rate_pairs']:.1%})")
    p(f"  DISTINCTIVE leak (df<={lk['distinctive_df_threshold']}) : "
      f"{lk['gold_pairs_with_distinctive_leak']}/{lk['gold_pairs']} gold pairs "
      f"({lk['distinctive_leak_rate_pairs']:.2%})")
    p("    topical overlap is a query and a precedent sharing an ordinary word")
    p("    (\"income\", \"labour\"); it is not the answer leaking and cannot be")
    p("    removed without destroying the query.  Only the distinctive row matters.")
    p("")
    cv = s["caveats"]
    p("corpus caveats")
    p(f"  resolved citations     : {cv['resolved_citations']:,}")
    p(f"  gold actually in index : {cv['targets_in_the_index']:,} "
      f"({cv['fraction_gold_retrievable']:.1%})")
    p(f"  excluded as unretrievable: {cv['fraction_gold_excluded']:.1%}")
    p("")
    for line in s["interpretation"]:
        for chunk in _wrap(line, 78):
            p(f"  {chunk}")
    p("=" * 82)


def _wrap(text: str, width: int) -> List[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.eval.citation_recovery",
                                 description="Track B: citation recovery.")
    sub = ap.add_subparsers(dest="command", required=True)
    sa = sub.add_parser("sample")
    sa.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--modes", default=",".join(SYSTEMS))
    r.add_argument("--no-baseline", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    if args.command == "sample":
        with timed("sample"):
            qs = sample_queries(args.n)
            save_queries(qs)
        lk = audit_leakage(qs)
        print(f"{len(qs)} queries -> {QUERIES_JSON}")
        print(f"leak rate: {lk['leak_rate_queries']:.1%} of queries, "
              f"{lk['leak_rate_pairs']:.1%} of gold pairs")
        for q in qs[:3]:
            print(f"\n  doc {q.doc_id} ({q.year}) gold={len(q.gold)}  "
                  f"{q.title[:56]}")
            print(f"    query: {q.query[:150]}")
    elif args.command == "run":
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        with timed("track B"):
            s = run(args.n, modes, include_baseline=not args.no_baseline)
            paths = write_results(s)
        print_report(s)
        print("wrote " + ", ".join(str(p) for p in paths))
    else:
        if not RESULTS_JSON.exists():
            raise SystemExit("no citation_recovery.json — run `run` first")
        print_report(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
