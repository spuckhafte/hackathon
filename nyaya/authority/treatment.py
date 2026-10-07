"""Citation treatment classifier  (Member 1, wave 3).

Labels every edge of ``index/citations.jsonl`` as one of
``followed | distinguished | overruled | mentioned``, which is what turns plain
PageRank (system S3) into treatment-weighted authority (system S4) through
``config.TREATMENT_WEIGHTS`` and ``config.TREATMENT_PENALTY``.

The evidence is the sentence window Member 2 already stored on each edge: the
sentence containing the citation **plus the following one**, split with
``analyze.sentences()`` so the boundaries are the same ones ``analyze()`` used
for ``sent_id``.  Nothing here re-reads a PDF.

Three things make this more than a keyword list
-----------------------------------------------

**1. The cue nearest the citation wins.**  A single window routinely treats two
cases differently - "*Golak Nath* was followed, but *Sajjan Singh* is clearly
distinguishable" - so the label is decided by the cue closest to this edge's own
``cite_string``, not by the first cue in the window.  Without that, every
citation in such a sentence gets the same label.

**2. Negation is scoped.**  A naive cue list inverts
"we are unable to hold that *Mathew* stands overruled" into an overruling, which
is the opposite of what the court said.  Each cue occurrence is rejected if a
negator appears in the :data:`NEGATION_WINDOW_CHARS` characters before it inside
the same sentence.  The window is characters rather than the whole sentence
because a long sentence usually contains an unrelated "not".

**3. A smaller bench cannot overrule a larger one.**  In Indian practice a
two-judge bench has no power to overrule a Constitution Bench; what it can do is
doubt or distinguish.  So when the cue says *overruled* but the citing bench is
smaller than the cited bench, the label is **downgraded to ``distinguished``**.
That is deliberately encoded in the label rather than in a side field, because
``contracts.CitationEdge`` has no room for one and because the downgrade is
exactly what should happen to the weight: 0.0 (overruled) would zero the cited
judgment's authority on the word of a bench that could not do it, whereas 0.3
(distinguished) records the doubt.  The pre-rule label is kept in the stats file
so the size of the effect is measurable.

Bench sizes come from this project's own parse of the judge line
(``zones.find_bench``), which is why wave 1 insisted on it: the metadata
``judge`` column is single-judge on 37k of 42k rows, so the rule would never fire
if it read the metadata.

    python -m nyaya.authority.treatment label          # label the edge file
    python -m nyaya.authority.treatment stats
    python -m nyaya.authority.treatment sample --label overruled --n 20
    python -m nyaya.authority.treatment explain --edge 1234
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import CitationEdge
from nyaya.utils import ensure_utf8_stdout, read_jsonl, timed

CITATIONS_JSONL = config.INDEX / "citations.jsonl"
TREATMENT_STATS = config.INDEX / "treatment_stats.json"
BENCH_CACHE = config.PROCESSED / "bench_sizes.npz"

OVERRULED = "overruled"
DISTINGUISHED = "distinguished"
FOLLOWED = "followed"
MENTIONED = "mentioned"

#: Priority when two cues are exactly as close to the citation.  Strongest
#: statement first: a court that both follows and overrules in one breath has
#: overruled.
PRIORITY = (OVERRULED, DISTINGUISHED, FOLLOWED)

# ==========================================================================
# 1. The cue tables
# ==========================================================================
# Every phrase below was taken from this corpus, not invented.  They are matched
# case-insensitively with flexible whitespace, because the scans break lines
# anywhere.

CUES: Dict[str, Tuple[str, ...]] = {
    OVERRULED: (
        "overrule", "overruled", "overruling", "overrules",
        "stands overruled", "hereby overruled", "is overruled",
        "no longer good law", "not good law", "ceases to be good law",
        "cannot be said to lay down the correct law",
        "does not lay down the correct law",
        "does not lay down the law correctly",
        "not laid down the correct law",
        "wrongly decided", "erroneously decided",
        "per incuriam",
        "cannot be regarded as laying down the correct law",
        "no longer holds the field",
        "does not represent the correct position in law",
        "we are unable to subscribe to the view",
        "stands impliedly overruled", "not followed", "dissented from",
        "held to be wrongly decided",
    ),
    DISTINGUISHED: (
        "distinguished", "distinguishable", "is distinguishable",
        "clearly distinguishable", "distinguishable on facts",
        "has no application", "have no application",
        "held inapplicable", "doubted", "explained and distinguished",
        "is not applicable", "are not applicable", "not applicable to the facts",
        "does not apply", "do not apply", "inapplicable",
        "turned on its own facts", "turns on its own facts",
        "on its own peculiar facts",
        "of no assistance", "does not assist", "is of no help",
        "cannot be applied", "cannot be pressed into service",
        "does not help the", "is of no avail",
        "stands on a different footing", "rests on a different footing",
        "confined to its own facts",
    ),
    FOLLOWED: (
        "followed", "we follow", "following the decision", "following the view",
        "is followed", "was followed", "has been followed",
        "reiterated", "reaffirmed", "affirmed", "approved",
        "relied upon", "relied on", "relying upon", "relying on", "we rely on",
        "we respectfully agree", "respectfully agree with",
        "squarely covered", "fully covered", "covered by the decision",
        "we are bound by", "bound by the decision", "binding on us",
        "as held in", "it was held in", "settled by",
        "lays down the correct law", "correctly lays down",
        "we approve", "we endorse", "in accordance with the decision",
        "consistent with the view taken",
    ),
}

#: Words that reverse a cue when they stand in front of it.  "unable" and
#: "decline" matter as much as "not": "we are unable to hold that X stands
#: overruled" is the measured failure that motivated this whole mechanism.
NEGATORS = (
    "not", "no", "nor", "never", "none", "neither",
    "cannot", "can not", "could not", "would not", "shall not", "do not",
    "does not", "did not", "is not", "are not", "was not", "were not",
    "unable", "declined", "decline", "refuse", "refused", "reject", "rejected",
    "far from", "nothing to", "without", "rather than", "instead of",
    "no question of", "need not", "cannot be said", "hardly",
)
#: How far back a negator is allowed to reach.  Characters, not tokens, and not
#: the whole sentence: the reports write very long sentences and an unrelated
#: "not" forty words earlier does not negate the cue.  80 characters is about
#: twelve words, which covers "we are unable to hold that <case name> stands
#: overruled".
NEGATION_WINDOW_CHARS = 80

_WS = re.compile(r"\s+")


def _flex(phrase: str) -> re.Pattern:
    """A cue pattern tolerant of the line breaks these PDFs are full of."""
    parts = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\w{0,3}\b", re.I)


_CUE_RES: Dict[str, Tuple[Tuple[str, re.Pattern], ...]] = {
    label: tuple((p, _flex(p)) for p in sorted(phrases, key=len, reverse=True))
    for label, phrases in CUES.items()
}
_NEG_RES = tuple(_flex(n) for n in NEGATORS)



# ==========================================================================
# 1b. How the reports themselves record treatment
# ==========================================================================
# The decisive discovery of the hand check: the Supreme Court Reports print the
# disposition of each authority *immediately after that authority's citation*,
# both in the headnote and in the "Case Law Reference" table:
#
#     Its Employees [1960] 2 S.C.R. 942 referred to.
#     Letang v. Cooper [1965] 1 Q.B. 232 distinguished.
#     2010 (10) SCR 487 relied on Para 11
#
# That is per-citation ground truth written by the reporter, and it beats any
# cue-distance heuristic: the nearest-cue rule was reading the *neighbouring*
# entry's disposition and labelling a "referred to" case as distinguished.  So
# this table is consulted first and wins outright when it fires.

DISPOSITIONS: Tuple[Tuple[str, str], ...] = (
    # (phrase, label) - longest first, because "not followed" contains "followed"
    ("held to be wrongly decided", OVERRULED),
    ("held to be per incuriam", OVERRULED),
    ("not followed", OVERRULED),
    ("dissented from", OVERRULED),
    ("overruled", OVERRULED),
    ("per incuriam", OVERRULED),
    ("held inapplicable", DISTINGUISHED),
    ("distinguished", DISTINGUISHED),
    ("explained", DISTINGUISHED),
    ("doubted", DISTINGUISHED),
    ("inapplicable", DISTINGUISHED),
    ("relied upon", FOLLOWED),
    ("relied on", FOLLOWED),
    ("followed", FOLLOWED),
    ("approved", FOLLOWED),
    ("applied", FOLLOWED),
    ("affirmed", FOLLOWED),
    ("reiterated", FOLLOWED),
    ("referred to", MENTIONED),
    ("cited", MENTIONED),
    ("considered", MENTIONED),
    ("discussed", MENTIONED),
)
_DISPOSITION_RES = tuple((phrase, label, _flex(phrase))
                         for phrase, label in DISPOSITIONS)

#: How far after the citation a reporter's disposition may sit.  Measured: the
#: longest real gap is "[1964] 3 SCR 506 at p. 514 ... is of no assistance", and
#: 60 characters covers the "held", "at p. NNN" and punctuation that intervene
#: without reaching the next entry in the table.
DISPOSITION_WINDOW_CHARS = 60

#: How far a *cue* (as opposed to a reporter's disposition) may sit from the
#: citation before it is assumed to be about some other case.  Asymmetric on
#: purpose, and the asymmetry is the measured result of the 40-edge hand check:
#:
#: * ``overruled`` carries weight 0.0 and penalty 0.25, so a false positive
#:   silently destroys a good judgment's authority.  Seven of the ten sampled
#:   overrulings were a cue 122-692 characters away belonging to a *neighbouring*
#:   citation in a headnote list.  So an overruling is accepted only from the
#:   reporter's own disposition, or from a cue in the **same sentence** as the
#:   citation and within 150 characters.
#: * ``followed`` is the opposite case: the reports write "A; B; C - relied on",
#:   one disposition for a whole list, and all ten sampled followings were
#:   correct at distances up to 423 characters.  Capping it would throw away real
#:   labels to prevent an error that does not happen, so it is uncapped.
MAX_CUE_DISTANCE = {
    OVERRULED: 150,
    DISTINGUISHED: 200,
    FOLLOWED: 0,            # 0 means no cap
}

#: "This Court overruled the contention on behalf of the respondent" - the object
#: of the verb is an argument, not a precedent, so no judgment was overruled.
_CUE_OBJECT_NOT_A_CASE = re.compile(
    r"\s+(?:the\s+|this\s+|that\s+|such\s+|his\s+|their\s+)?"
    r"(?:contention|contentions|submission|submissions|argument|arguments"
    r"|objection|objections|plea|pleas|claim|claims|preliminary\s+objection)\b",
    re.I)

#: "... Shantilal Mangal Das, [1969] 3 S.C.R. 341, the Court overruled ..." - the
#: cited judgment is the one *doing* the overruling, so it is not overruled.
_CITED_IS_THE_ACTOR = re.compile(
    r"^[\s,;.]*(?:in\s+)?(?:the\s+|this\s+)?"
    r"(?:Court|Bench|majority|Constitution\s+Bench)\b", re.I)

#: "... has been specifically overruled by the Constitution Bench of this Court in
#: Indore Development Authority v. Manoharlal, (2020) 8 SCC 129" - the citation
#: follows the cue as its *agent*, so the cited judgment is the one that did the
#: overruling, not the one that was overruled.
_CITE_IS_THE_AGENT = re.compile(r"^\s*by\b", re.I)

# ==========================================================================
# 2. One edge
# ==========================================================================

@dataclass
class Label:
    """The classifier's decision and everything behind it."""

    treatment: str = MENTIONED
    raw: str = MENTIONED            # before the bench rule
    cue: str = ""
    distance: int = -1              # characters from the citation to the cue
    negated_cues: List[str] = field(default_factory=list)
    bench_src: int = 0
    bench_dst: int = 0
    downgraded: bool = False
    sentence_index: int = -1
    rule: str = "default"           # 'disposition' | 'nearest-cue' | 'default'
    rejected: List[str] = field(default_factory=list)

    def explain(self) -> str:
        if self.treatment == MENTIONED and not self.cue:
            bits = ["no cue fired"]
            if self.negated_cues:
                bits.append(f"negated: {self.negated_cues}")
            if self.rejected:
                bits.append(f"rejected: {self.rejected}")
            return "; ".join(bits)
        if self.rule == "disposition":
            bits = [f"reporter's disposition {self.cue!r} "
                    f"{self.distance} chars after the citation"]
        else:
            bits = [f"cue {self.cue!r} at {self.distance} chars from the citation"]
        if self.rejected:
            bits.append(f"rejected: {self.rejected}")
        if self.negated_cues:
            bits.append(f"negated cues ignored: {self.negated_cues}")
        if self.downgraded:
            bits.append(f"downgraded {self.raw} -> {self.treatment}: citing bench "
                        f"{self.bench_src} < cited bench {self.bench_dst}")
        return "; ".join(bits)


def _sentence_spans(window: str) -> List[Tuple[int, int, str]]:
    """``(start, end, text)`` per sentence, using Member 3's splitter.

    Her ``sentences()`` returns the strings, so the offsets are recovered by
    walking the window - cheaper and safer than a second splitter that would
    disagree with the ``sent_id`` values stored in the index.
    """
    from nyaya.analyze import sentences

    out: List[Tuple[int, int, str]] = []
    cursor = 0
    for sent in sentences(window):
        s = sent.strip()
        if not s:
            continue
        i = window.find(s[:40], cursor)
        if i < 0:
            i = cursor
        out.append((i, i + len(s), s))
        cursor = i + len(s)
    if not out:
        out = [(0, len(window), window)]
    return out


def _squeeze_map(text: str) -> Tuple[str, List[int]]:
    """Whitespace-collapsed text plus, per output character, its input offset."""
    out: List[str] = []
    back: List[int] = []
    prev_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_ws:
                continue
            out.append(" ")
            back.append(i)
            prev_ws = True
        else:
            out.append(ch)
            back.append(i)
            prev_ws = False
    return "".join(out), back


def cite_span(window: str, cite_string: str) -> Tuple[int, int]:
    """``(start, end)`` of this edge's citation in ``window``.

    The citation is routinely broken across a line by the scan
    ("(1979) 3\nSCC 745"), so an exact search fails and a squeezed search is
    needed.  The offsets must be mapped **back** to the original window, which is
    what the earlier version got wrong: it returned the squeezed index, so the
    disposition scan started in the wrong place and the reporter's own
    "- referred to" was missed on every multi-line citation.
    """
    if not cite_string:
        return 0, 0
    i = window.lower().find(cite_string.lower())
    if i >= 0:
        return i, i + len(cite_string)
    sq_win, back = _squeeze_map(window.lower())
    sq_cite = _WS.sub(" ", cite_string.lower()).strip()
    j = sq_win.find(sq_cite)
    if j < 0:
        return 0, 0
    start = back[j]
    last = min(j + len(sq_cite) - 1, len(back) - 1)   # last matched character
    return start, back[last] + 1


def _cite_offset(window: str, cite_string: str) -> int:
    return cite_span(window, cite_string)[0]


def forward_disposition(window: str, cite_end: int
                        ) -> Optional[Tuple[str, str, int]]:
    """``(label, phrase, offset)`` of the reporter's own disposition, if any.

    Looks only *forward* from the end of this edge's citation, so each entry in a
    "Case Law Reference" table gets its own disposition rather than its
    neighbour's.
    """
    tail = window[cite_end:cite_end + DISPOSITION_WINDOW_CHARS]
    best: Optional[Tuple[str, str, int]] = None
    for phrase, label, rx in _DISPOSITION_RES:
        m = rx.search(tail)
        if not m:
            continue
        # The guards have to run against absolute offsets in the whole window,
        # not against the 60-character tail.  "We are unable to hold that the
        # decision in Mathew [1960] 1 SCR 1 stands overruled" puts the negator
        # *before the citation*, so a tail-local negation check never sees it.
        abs_at = cite_end + m.start()
        if _is_negated(window, abs_at):
            continue
        if label == OVERRULED:
            if _CITED_IS_THE_ACTOR.match(window[cite_end:abs_at]):
                continue
            after = window[abs_at + m.end() - m.start():][:40]
            if _CUE_OBJECT_NOT_A_CASE.match(after):
                continue
        if best is None or m.start() < best[2]:
            best = (label, phrase, m.start())
    return best


def _is_negated(sentence: str, cue_start: int) -> Optional[str]:
    """The negator in front of ``cue_start``, or None."""
    lo = max(0, cue_start - NEGATION_WINDOW_CHARS)
    before = sentence[lo:cue_start]
    for rx in _NEG_RES:
        m = rx.search(before)
        if m:
            return m.group(0).lower()
    return None


def classify(sentence: str, cite_string: str = "",
             bench_src: int = 0, bench_dst: int = 0) -> Label:
    """Label one citation from its sentence window."""
    window = sentence or ""
    if not window.strip():
        return Label(bench_src=bench_src, bench_dst=bench_dst)
    cite_at, cite_end = cite_span(window, cite_string)
    spans = _sentence_spans(window)
    rejected: List[str] = []

    # --- the reporter's own disposition wins -------------------------------
    disp = forward_disposition(window, cite_end)
    if disp is not None:
        label, phrase, off = disp
        out = Label(treatment=label, raw=label, cue=phrase, distance=off,
                    bench_src=bench_src, bench_dst=bench_dst,
                    rule="disposition")
        return _apply_bench_rule(out)

    best: Optional[Tuple[int, int, str, str, int]] = None   # dist, prio, label, cue, sidx
    negated: List[str] = []
    for sidx, (s_start, _s_end, sent) in enumerate(spans):
        for label, patterns in _CUE_RES.items():
            for phrase, rx in patterns:
                for m in rx.finditer(sent):
                    neg = _is_negated(sent, m.start())
                    if neg:
                        negated.append(f"{neg} ... {phrase}")
                        continue
                    abs_at = s_start + m.start()
                    if label == OVERRULED:
                        after = sent[m.end():m.end() + 40]
                        if _CUE_OBJECT_NOT_A_CASE.match(after):
                            rejected.append(f"{phrase} + {after.split()[:2]}")
                            continue
                        if abs_at > cite_end and _CITED_IS_THE_ACTOR.match(
                                window[cite_end:abs_at]):
                            rejected.append(f"{phrase} (cited case is the actor)")
                            continue
                        cue_end = s_start + m.end()
                        if cite_at > cue_end and _CITE_IS_THE_AGENT.match(
                                window[cue_end:cue_end + 80]):
                            rejected.append(f"{phrase} by <cite> (cited case is "
                                            f"the overruler)")
                            continue
                    dist = abs(abs_at - cite_at)
                    prio = PRIORITY.index(label)
                    cand = (dist, prio, label, phrase, sidx)
                    if best is None or cand[:2] < best[:2]:
                        best = cand

    if best is None:
        return Label(negated_cues=negated[:4], rejected=rejected[:4],
                     bench_src=bench_src, bench_dst=bench_dst)
    dist, _prio, label, phrase, sidx = best
    cap = MAX_CUE_DISTANCE.get(label, 0)
    if cap and dist > cap:
        rejected.append(f"{phrase} at {dist} chars > cap {cap}")
        return Label(negated_cues=negated[:4], rejected=rejected[:4],
                     bench_src=bench_src, bench_dst=bench_dst)
    if label == OVERRULED and sidx != _sentence_of(spans, cite_at):
        rejected.append(f"{phrase} is in another sentence than the citation")
        return Label(negated_cues=negated[:4], rejected=rejected[:4],
                     bench_src=bench_src, bench_dst=bench_dst)
    out = Label(treatment=label, raw=label, cue=phrase, distance=dist,
                negated_cues=negated[:4], rejected=rejected[:4],
                bench_src=bench_src, bench_dst=bench_dst,
                sentence_index=sidx, rule="nearest-cue")
    return _apply_bench_rule(out)


def _sentence_of(spans: Sequence[Tuple[int, int, str]], offset: int) -> int:
    for i, (a, b, _t) in enumerate(spans):
        if a <= offset < b:
            return i
    return -1


def _apply_bench_rule(out: Label) -> Label:
    """A smaller bench cannot overrule a larger one; it can only doubt it.

    Encoded as a downgrade of the label rather than a side field, because
    ``contracts.CitationEdge`` has no spare field and because the downgrade *is*
    the intended effect on the weight: 0.0 would zero the cited judgment's
    authority on the word of a bench that had no power to do it, while 0.3
    records the doubt.
    """
    if (out.treatment == OVERRULED and out.bench_dst > 0 and out.bench_src > 0
            and out.bench_src < out.bench_dst):
        out.treatment = DISTINGUISHED
        out.downgraded = True
    return out


# ==========================================================================
# 3. Bench sizes for the rule
# ==========================================================================

def build_bench_cache(force: bool = False) -> Dict[str, np.ndarray]:
    """``doc_id -> bench size`` and ``node_id -> bench size``, cached on disk.

    Two tables because a citation's ``src`` is a *doc_id* (always indexed, so our
    own parse of the judge line is available) while its ``dst`` is a *node_id*
    that may point outside the indexed corpus, where only Member 2's metadata
    exists.  The indexed nodes take our parse; the rest fall back to
    ``bench_size_meta``, and a zero means "unknown", which makes the rule abstain
    rather than guess.
    """
    import pandas as pd

    if BENCH_CACHE.exists() and not force:
        with np.load(BENCH_CACHE) as z:
            return {k: z[k] for k in z.files}

    man = pd.read_parquet(config.MANIFEST_PARQUET)
    uni = pd.read_parquet(config.UNIVERSE_PARQUET,
                          columns=["node_id", "bench_size_meta"])

    n_nodes = int(uni["node_id"].max()) + 1
    node_bench = np.zeros(n_nodes, dtype=np.int16)
    node_bench[uni["node_id"].to_numpy()] = uni["bench_size_meta"].to_numpy()

    n_docs = int(man["doc_id"].max()) + 1
    doc_bench = np.zeros(n_docs, dtype=np.int16)
    # our own parse, from the built corpus
    parsed = 0
    for row in read_jsonl(config.JUDGMENTS_JSONL):
        d = int(row["doc_id"])
        b = int(row.get("bench_size") or 0)
        if 0 <= d < n_docs and b > 0:
            doc_bench[d] = b
            parsed += 1
    # where we have no parse, fall back to the metadata
    doc_to_node = dict(zip(man["doc_id"].to_numpy(), man["node_id"].to_numpy()))
    for d in range(n_docs):
        if doc_bench[d] == 0:
            nd = doc_to_node.get(d)
            if nd is not None and 0 <= int(nd) < n_nodes:
                doc_bench[d] = node_bench[int(nd)]
    # and the cited side prefers our parse wherever the node is also indexed
    for d, nd in doc_to_node.items():
        if 0 <= int(nd) < n_nodes and doc_bench[int(d)] > 0:
            node_bench[int(nd)] = doc_bench[int(d)]

    BENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BENCH_CACHE, doc_bench=doc_bench, node_bench=node_bench,
             parsed=np.asarray([parsed]))
    return {"doc_bench": doc_bench, "node_bench": node_bench,
            "parsed": np.asarray([parsed])}


# ==========================================================================
# 4. The whole edge file
# ==========================================================================

def label_file(path: Optional[Path] = None, dry_run: bool = False,
               verbose: bool = True) -> Dict[str, Any]:
    """Label every edge in place, and write ``index/treatment_stats.json``."""
    import os
    from collections import Counter

    p = Path(path or CITATIONS_JSONL)
    if not p.exists():
        raise SystemExit(f"{p} missing - run python -m nyaya.authority.citations build")
    bench = build_bench_cache()
    doc_bench = bench["doc_bench"]
    node_bench = bench["node_bench"]

    counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    cue_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    n_rejected = 0
    n_neg = 0
    n_down = 0
    n_unresolved = 0
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for n, row in enumerate(read_jsonl(p), 1):
        src = int(row.get("src") or 0)
        dst = int(row.get("dst") if row.get("dst") is not None else -1)
        bs = int(doc_bench[src]) if 0 <= src < doc_bench.size else 0
        bd = int(node_bench[dst]) if 0 <= dst < node_bench.size else 0
        lab = classify(row.get("sentence") or "", row.get("cite_string") or "",
                       bench_src=bs, bench_dst=bd)
        row["treatment"] = lab.treatment
        counts[lab.treatment] += 1
        raw_counts[lab.raw] += 1
        if lab.cue:
            cue_counts[lab.cue] += 1
        rule_counts[lab.rule] += 1
        if lab.rejected:
            n_rejected += 1
        if lab.negated_cues:
            n_neg += 1
        if lab.downgraded:
            n_down += 1
        if dst < 0:
            n_unresolved += 1
        rows.append(row)
        if verbose and n % 10_000 == 0:
            print(f"  labelled {n} edges ({time.perf_counter() - t0:.0f}s)",
                  file=sys.stderr, flush=True)

    stats = {
        "edges": len(rows),
        "unresolved_edges": n_unresolved,
        "labels": dict(counts.most_common()),
        "labels_before_bench_rule": dict(raw_counts.most_common()),
        "overruled_downgraded_by_bench_rule": n_down,
        "decided_by": dict(rule_counts.most_common()),
        "edges_with_a_negated_cue": n_neg,
        "edges_with_a_rejected_overrule_cue": n_rejected,
        "top_cues": dict(cue_counts.most_common(20)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if dry_run:
        return stats

    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    os.replace(tmp, p)
    TREATMENT_STATS.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    stats["written"] = str(p)
    return stats


def iter_labelled(path: Optional[Path] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for i, row in enumerate(read_jsonl(Path(path or CITATIONS_JSONL))):
        yield i, row


# ==========================================================================
# 5. CLI
# ==========================================================================

def _cmd_label(args) -> int:
    with timed("treatment"):
        stats = label_file(dry_run=args.dry_run)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


def _cmd_stats(args) -> int:
    if not TREATMENT_STATS.exists():
        raise SystemExit("run `python -m nyaya.authority.treatment label` first")
    for k, v in json.loads(TREATMENT_STATS.read_text(encoding="utf-8")).items():
        print(f"{k}: {v}")
    return 0


def _cmd_sample(args) -> int:
    bench = build_bench_cache()
    shown = 0
    for i, row in iter_labelled():
        if args.label and row.get("treatment") != args.label:
            continue
        if shown >= args.n:
            break
        shown += 1
        src, dst = int(row["src"]), int(row["dst"])
        bs = int(bench["doc_bench"][src]) if src < bench["doc_bench"].size else 0
        bd = int(bench["node_bench"][dst]) if 0 <= dst < bench["node_bench"].size else 0
        lab = classify(row.get("sentence") or "", row.get("cite_string") or "",
                       bs, bd)
        print("=" * 78)
        print(f"edge {i}  src={src} dst={dst}  bench {bs} vs {bd}  "
              f"para={row.get('para_id')}  via={row.get('resolved_by')}")
        print(f"  cite    : {row.get('cite_string')!r}")
        print(f"  label   : {lab.treatment}   ({lab.explain()})")
        print(f"  window  : {_WS.sub(' ', row.get('sentence') or '')[:430]}")
    return 0


def _cmd_explain(args) -> int:
    for i, row in iter_labelled():
        if i != args.edge:
            continue
        bench = build_bench_cache()
        src, dst = int(row["src"]), int(row["dst"])
        bs = int(bench["doc_bench"][src]) if src < bench["doc_bench"].size else 0
        bd = int(bench["node_bench"][dst]) if 0 <= dst < bench["node_bench"].size else 0
        lab = classify(row.get("sentence") or "", row.get("cite_string") or "",
                       bs, bd)
        print(json.dumps({"edge": i, "src": src, "dst": dst,
                          "bench_src": bs, "bench_dst": bd,
                          "cite_string": row.get("cite_string"),
                          "treatment": lab.treatment, "raw": lab.raw,
                          "cue": lab.cue, "distance": lab.distance,
                          "negated": lab.negated_cues,
                          "downgraded": lab.downgraded,
                          "sentence": row.get("sentence")},
                         indent=1, ensure_ascii=False))
        return 0
    raise SystemExit(f"no edge {args.edge}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.authority.treatment",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    lb = sub.add_parser("label", help="label index/citations.jsonl in place")
    lb.add_argument("--dry-run", action="store_true")
    lb.set_defaults(func=_cmd_label)

    st = sub.add_parser("stats", help="the label distribution")
    st.set_defaults(func=_cmd_stats)

    sm = sub.add_parser("sample", help="print labelled edges for hand checking")
    sm.add_argument("--label", default=None)
    sm.add_argument("--n", type=int, default=10)
    sm.set_defaults(func=_cmd_sample)

    ex = sub.add_parser("explain", help="everything about one edge")
    ex.add_argument("--edge", type=int, required=True)
    ex.set_defaults(func=_cmd_explain)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
