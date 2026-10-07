"""Operator correctness: Boolean, phrase, /n, /s, /p  (Member 1, wave 4).

Answers every operator query twice and compares.  This is the evidence that the
Boolean engine is correct, and it is a stated beat of the demo video.

**A. index vs naive scan** - the expected answer is recomputed by re-analysing
the judgment text with ``analyze()`` and scanning the token list with a plain
nested loop: no postings, no skip pointers, no numpy merge, and for Boolean no
planner.  This is the correctness proof for the code I wrote and it must agree
**100%**.  A single mismatch is a bug, and the command exits non-zero.

**B. index vs a regular expression over the raw text** - the same queries
answered by a regex over the surface words, which knows nothing about stemming,
statute canonicalisation, running-header removal or zone segmentation.  This is
*not* expected to agree perfectly.  Every disagreement is classified and printed,
because the regex and the index are answering slightly different questions and
the difference is worth explaining rather than hiding.

Mode B is reported twice: the share of *checks* that are identical, and the
share of *documents* that agree pooled over those checks.  The second is the
honest one - a check that differs on one document out of 1,092 and a check that
differs on all of them both count as one disagreeing check in the first.

    python -m nyaya.eval.operators --brute-force
    python -m nyaya.eval.operators --brute-force --docs 2000 --stride 7
    python -m nyaya.eval.operators --brute-force --classify 20
    python -m nyaya.eval.operators --brute-force --json runs/operators.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from nyaya import config
from nyaya.analyze import analyze, analyze_terms
from nyaya.index import positional as P
from nyaya.query import proximity as prox
from nyaya.utils import ensure_utf8_stdout

OPERATORS_JSON = config.RUNS / "operators.json"

# ==========================================================================
# The query set: surface wording a lawyer would type
# ==========================================================================

PHRASES = [
    "natural justice", "anticipatory bail", "fundamental right", "high court",
    "special leave", "burden of proof", "reasonable restriction",
    "basic structure", "res judicata", "ultra vires", "mens rea",
    "legitimate expectation", "judicial review", "public interest litigation",
]

PAIRS = [
    ("privacy", "surveillance"), ("bail", "anticipatory"),
    ("conviction", "eyewitness"), ("compensation", "negligence"),
    ("arbitration", "jurisdiction"), ("sedition", "speech"),
    ("dowry", "harassment"), ("acquittal", "benefit"),
    ("custody", "welfare"), ("reservation", "merit"),
]

WINDOWS = (1, 3, 5, 10, 25)

#: Boolean queries, as (description, expression-tree-in-python) over term sets.
BOOLEANS = [
    ("bail AND anticipatory", ("AND", "bail", "anticipatory")),
    ("privacy AND surveillance", ("AND", "privacy", "surveillance")),
    ("sedition OR defamation", ("OR", "sedition", "defamation")),
    ("bail OR parole", ("OR", "bail", "parole")),
    ("conviction NOT acquittal", ("NOT", "conviction", "acquittal")),
    ("arbitration NOT insurance", ("NOT", "arbitration", "insurance")),
    ("murder AND intention", ("AND", "murder", "intention")),
    ("dowry OR cruelty", ("OR", "dowry", "cruelty")),
    ("negligence NOT contributory", ("NOT", "negligence", "contributory")),
    ("reservation AND promotion", ("AND", "reservation", "promotion")),
]

# ==========================================================================
# The independent token stream
# ==========================================================================


class Stream:
    """One document's tokens, rebuilt from the text without touching the index."""

    __slots__ = ("terms", "zones", "sents", "paras", "index_of", "vocab")

    def __init__(self, terms, zones, sents, paras):
        self.terms = terms
        self.zones = zones
        self.sents = sents
        self.paras = paras
        self.index_of: Dict[str, List[int]] = {}
        for i, t in enumerate(terms):
            self.index_of.setdefault(t, []).append(i)
        self.vocab = set(self.index_of)

    def positions_of(self, term: str) -> List[int]:
        return self.index_of.get(term, [])


def _with_statutes_zone(record: dict) -> dict:
    """A copy of the record whose synthetic ``statutes`` zone is filled in.

    ``positional.analyse_record`` synthesises that zone when it is still empty,
    so a stream rebuilt straight from ``judgments.jsonl`` would be short exactly
    those tokens on the documents ``zones.py`` left blank, and every operator
    would then "disagree" for a reason that has nothing to do with the merge
    code.  Reproduced rather than imported, so this file still owns its own
    flattening loop.
    """
    from nyaya import statutes as stat_mod
    from nyaya import zones as zone_mod
    from nyaya.contracts import JudgmentRecord

    rec = dict(record)
    if (rec.get("zones") or {}).get("statutes"):
        return rec
    body = "\n".join(p.get("text", "") for p in (rec.get("paras") or []))
    shim = JudgmentRecord(doc_id=int(rec["doc_id"]),
                          source_path=str(rec.get("source_path") or ""),
                          title=str(rec.get("title") or ""),
                          zones=dict(rec.get("zones") or {}))
    zone_mod.attach_statute_zone(shim, stat_mod.extract_statutes(body))
    rec["zones"] = shim.zones
    return rec


def token_stream(record: dict) -> Stream:
    record = _with_statutes_zone(record)
    terms: List[str] = []
    zones: List[int] = []
    sents: List[int] = []
    paras: List[int] = []
    pos = 0
    sent_no = para_no = -1
    last_s = last_p = None
    for zone, text, para_id in P._zone_texts(record, zoned=True,
                                             include_statutes=True):
        zid = config.ZONE_ID[zone]
        for t in analyze(text, zone=zone, para_id=para_id, start_pos=pos):
            skey = (zone, para_id, t.sent_id)
            if skey != last_s:
                sent_no += 1
                last_s = skey
            pkey = (zone, para_id)
            if pkey != last_p:
                para_no += 1
                last_p = pkey
            terms.append(t.term)
            zones.append(zid)
            sents.append(sent_no)
            paras.append(para_no)
            pos += 1
    return Stream(terms, zones, sents, paras)


def raw_text(record: dict) -> str:
    parts: List[str] = []
    for z in config.ZONES:
        if z in ("body", "order"):
            parts.extend(p.get("text", "") for p in (record.get("paras") or [])
                         if p.get("zone") == z)
        else:
            t = (record.get("zones") or {}).get(z) or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts)


# ==========================================================================
# Naive scans - the ground truth for mode A
# ==========================================================================

def naive_phrase(st: Stream, words: Sequence[str]) -> bool:
    m = len(words)
    for i in st.positions_of(words[0]):
        if i + m <= len(st.terms) and st.terms[i:i + m] == list(words) \
                and len(set(st.zones[i:i + m])) == 1:
            return True
    return False


def naive_window(st: Stream, a: str, b: str, k: int) -> bool:
    pa, pb = st.positions_of(a), st.positions_of(b)
    for i in pa:
        for j in pb:
            if i != j and abs(i - j) <= k and st.zones[i] == st.zones[j]:
                return True
    return False


def naive_bucket(st: Stream, a: str, b: str, kind: str) -> bool:
    pa, pb = st.positions_of(a), st.positions_of(b)
    buckets = st.sents if kind == "s" else st.paras
    for i in pa:
        for j in pb:
            if i != j and buckets[i] == buckets[j] and st.zones[i] == st.zones[j]:
                return True
    return False


def naive_boolean(st: Stream, op: str, a: str, b: str) -> bool:
    ha, hb = a in st.vocab, b in st.vocab
    if op == "AND":
        return ha and hb
    if op == "OR":
        return ha or hb
    if op == "NOT":
        return ha and not hb
    raise ValueError(op)


def index_boolean(idx, op: str, a: str, b: str) -> Set[int]:
    """The same Boolean through the postings lists and the merge primitives."""
    da = set(map(int, prox.term_docs(a, idx)))
    db = set(map(int, prox.term_docs(b, idx)))
    if op == "AND":
        return da & db
    if op == "OR":
        return da | db
    if op == "NOT":
        return da - db
    raise ValueError(op)


# ==========================================================================
# Regex over the raw text - mode B
# ==========================================================================

def regex_phrase(text: str, surface: str) -> bool:
    pat = re.compile(r"\b" + r"[\s\-]+".join(map(re.escape, surface.split()))
                     + r"\w*", re.I)
    return bool(pat.search(text))


def regex_term(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\w{{0,3}}\b", text, re.I))


def regex_window(text: str, a: str, b: str, k: int) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    pa = [i for i, w in enumerate(words)
          if w.startswith(a) and len(w) <= len(a) + 3]
    pb = [i for i, w in enumerate(words)
          if w.startswith(b) and len(w) <= len(b) + 3]
    for i in pa:
        for j in pb:
            if i != j and abs(i - j) <= k:
                return True
    return False


# ==========================================================================
# Explaining a mode-B disagreement
# ==========================================================================

STEMMING = "stemming conflation (index matches a variant the surface regex cannot)"
NORMALISE = "normalisation (index is adjacent across OCR junk the regex sees)"
TOKENCOUNT = "token count (index counts numerals/abbreviations the regex skips)"
CROSSZONE = "cross-zone (regex matched across a zone seam; the index rejects it)"
LOOSE = "regex prefix over-match (the surface pattern hit a different word)"
UNKNOWN = "unexplained"


def _surface_variants(text: str, word: str, stem: str) -> Set[str]:
    out: Set[str] = set()
    for m in re.finditer(r"[A-Za-z]{3,}", text):
        low = m.group(0).lower()
        if low.startswith(word) and len(low) <= len(word) + 3:
            continue
        if analyze_terms(m.group(0))[:1] == [stem]:
            out.add(low)
    return out


def _regex_overmatches(text: str, word: str, stem: str) -> Dict[str, str]:
    """Surface words ``regex_term`` accepts for ``word`` that do not stem to ``stem``."""
    out: Dict[str, str] = {}
    for m in re.finditer(rf"\b{re.escape(word)}\w{{0,3}}\b", text, re.I):
        surface = m.group(0).lower()
        got = analyze_terms(surface)[:1]
        if got != [stem]:
            out[surface] = got[0] if got else ""
    return out


def explain_mismatch(kind: str, words: Sequence[str], stems: Sequence[str],
                     index_only: bool, st: Stream, text: str, k: int = 0
                     ) -> Tuple[str, str]:
    if index_only:
        for w, stem in zip(words, stems):
            extra = _surface_variants(text, w, stem)
            if extra:
                return STEMMING, f"{w!r}/{stem!r} also written {sorted(extra)[:3]}"
        i = text.lower().find(words[0])
        snippet = text[max(0, i - 5):i + 40].replace("\n", " ") if i >= 0 else ""
        return NORMALISE, f"raw text reads {snippet!r}"
    # The other side: the regex answered yes and the index no.  The usual cause
    # is the regex's three-character suffix wildcard, which is a crude stemmer -
    # "bail\w{0,3}" happily matches "bailiff", a word that stems to "bailiff"
    # and has nothing to do with bail.
    for w, stem in zip(words, stems):
        if stem not in st.vocab:
            hit = _regex_overmatches(text, w, stem)
            if hit:
                shown = ", ".join(f"{s!r}->{t!r}" for s, t in
                                  sorted(hit.items())[:3])
                return LOOSE, f"{w!r} pattern matched {shown}"
            return LOOSE, f"{w!r}/{stem!r} occurs in no zone of this judgment"
    pa = st.positions_of(stems[0])
    pb = st.positions_of(stems[-1])
    if pa and pb:
        gap, i, j = min((abs(a - b), a, b) for a in pa for b in pb)
        if st.zones[i] != st.zones[j]:
            return CROSSZONE, (f"closest pair spans "
                               f"{config.ZONES[st.zones[i]]}/"
                               f"{config.ZONES[st.zones[j]]}")
        if kind == "window" and gap > k:
            span = " ".join(st.terms[min(i, j):max(i, j) + 1])
            return TOKENCOUNT, f"token gap {gap} > k={k}; span {span!r}"
        if kind == "phrase" and gap > len(stems) - 1:
            span = " ".join(st.terms[min(i, j):max(i, j) + 1][:12])
            return TOKENCOUNT, (f"nearest occurrences are {gap} tokens apart, "
                                f"not adjacent; span {span!r}")
    return UNKNOWN, "no explanation found"


# ==========================================================================
# The run
# ==========================================================================

@dataclass
class Check:
    family: str
    label: str
    index_n: int = 0
    naive_n: int = 0
    regex_n: Optional[int] = None
    agrees_naive: bool = True
    agrees_regex: Optional[bool] = None
    index_only: List[int] = field(default_factory=list)
    other_only: List[int] = field(default_factory=list)
    classes: List[Tuple[int, str, str]] = field(default_factory=list)
    #: Document-level disagreement against the regex, counted in full even when
    #: only a sample of the documents is classified below.  Without these a
    #: "17 of 44 checks disagree" headline hides whether a check differed on one
    #: document out of a thousand or on all of them.
    rx_index_only_n: int = 0
    rx_regex_only_n: int = 0
    rx_union_n: int = 0
    rx_classified: int = 0

    @property
    def rx_diff_n(self) -> int:
        return self.rx_index_only_n + self.rx_regex_only_n

    def as_json(self) -> Dict[str, Any]:
        return {
            "family": self.family, "label": self.label,
            "index": self.index_n, "naive": self.naive_n, "regex": self.regex_n,
            "agrees_naive": self.agrees_naive, "agrees_regex": self.agrees_regex,
            "index_only": self.index_only[:6], "other_only": self.other_only[:6],
            "regex_index_only_docs": self.rx_index_only_n,
            "regex_only_docs": self.rx_regex_only_n,
            "regex_union_docs": self.rx_union_n,
            "regex_docs_classified": self.rx_classified,
            "classes": [{"doc_id": d, "class": c, "evidence": e}
                        for d, c, e in self.classes[:6]],
        }


def load_records(limit: int, stride: int = 1) -> List[dict]:
    """Every ``stride``-th judgment, so the sample spans 1950 to 2025.

    The corpus is in doc_id order, which is year order, so the first N would
    verify the operators only against the 1950s scans, where most modern legal
    vocabulary does not occur and most queries return nothing.
    """
    out: List[dict] = []
    with Path(config.JUDGMENTS_JSONL).open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if n % stride:
                continue
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def classify_regex_diff(c: Check, got: Set[int], rset: Set[int], kind: str,
                        words: Sequence[str], stems: Sequence[str],
                        streams: Dict[int, Stream], raws: Dict[int, str],
                        k: int = 0, limit: int = 0) -> None:
    """Count the mode-B disagreement in full and classify ``limit`` a side (0 = all).

    The counts are always exact.  The default classifies *every* disagreeing
    document, because "the differences are tokenisation" is only a defence if
    each one has been looked at; ``--classify N`` caps it for a quick run.
    ``rx_classified`` records how many were examined, so a class histogram can
    never be read as a total by accident.
    """
    c.rx_index_only_n = len(got - rset)
    c.rx_regex_only_n = len(rset - got)
    c.rx_union_n = len(got | rset)
    for side, index_only in ((sorted(got - rset), True), (sorted(rset - got), False)):
        for d in (side if limit <= 0 else side[:limit]):
            c.classes.append((d,) + explain_mismatch(
                kind, words, stems, index_only, streams[d], raws[d], k))
            c.rx_classified += 1


def brute_force(docs: int = 1200, stride: int = 11, verbose: bool = True,
                max_show: int = 6, classify: int = 0) -> Dict[str, Any]:
    idx = P.PositionalIndex.load()
    t0 = time.perf_counter()
    records = load_records(docs, stride)
    ids = {int(r["doc_id"]) for r in records}
    streams = {int(r["doc_id"]): token_stream(r) for r in records}
    raws = {int(r["doc_id"]): raw_text(r) for r in records}
    if verbose:
        print(f"re-analysed {len(records)} judgments in "
              f"{time.perf_counter() - t0:.0f}s", file=sys.stderr)

    def restrict(a) -> Set[int]:
        return {int(d) for d in a if int(d) in ids}

    checks: List[Check] = []
    classes: Dict[str, int] = {}

    def record(c: Check) -> None:
        checks.append(c)
        for _d, cls, _e in c.classes:
            classes[cls] = classes.get(cls, 0) + 1
        if verbose:
            flag = "OK" if c.agrees_naive else "MISMATCH"
            print(f"  {c.family:<8} {c.label:<34} index={c.index_n:>5} "
                  f"naive={c.naive_n:>5} {flag}")

    # ---- Boolean -------------------------------------------------------
    for label, (op, a, b) in BOOLEANS:
        ta, tb = analyze_terms(a), analyze_terms(b)
        if not ta or not tb:
            continue
        got = restrict(index_boolean(idx, op, ta[0], tb[0]))
        want = {d for d, st in streams.items()
                if naive_boolean(st, op, ta[0], tb[0])}
        # One pass per side, then plain set algebra.  Written as a chained
        # comprehension this read as `AND-set or (OR-set if ... else NOT-set)`,
        # which silently fell through to the NOT branch whenever the AND set
        # came out empty.
        hits_a = {d for d, t in raws.items() if regex_term(t, a)}
        hits_b = {d for d, t in raws.items() if regex_term(t, b)}
        rx = ((hits_a & hits_b) if op == "AND"
              else (hits_a | hits_b) if op == "OR"
              else (hits_a - hits_b))
        c = Check("boolean", label, len(got), len(want), len(rx),
                  got == want, rx == got,
                  sorted(got - want)[:max_show], sorted(want - got)[:max_show])
        classify_regex_diff(c, got, rx, "boolean", [a, b], [ta[0], tb[0]],
                            streams, raws, limit=classify)
        record(c)

    # ---- phrases -------------------------------------------------------
    for surface in PHRASES:
        terms = analyze_terms(surface)
        if len(terms) < 2:
            continue
        got = restrict(prox.phrase_docs(terms, idx))
        want = {d for d, st in streams.items() if naive_phrase(st, terms)}
        rx = {d for d, t in raws.items() if regex_phrase(t, surface)}
        c = Check("phrase", repr(surface), len(got), len(want), len(rx),
                  got == want, rx == got,
                  sorted(got - want)[:max_show], sorted(want - got)[:max_show])
        classify_regex_diff(c, got, rx, "phrase", surface.split(), terms,
                            streams, raws, limit=classify)
        record(c)

    # ---- /n ------------------------------------------------------------
    for (a, b), k in itertools.product(PAIRS, WINDOWS):
        ta, tb = analyze_terms(a), analyze_terms(b)
        if not ta or not tb:
            continue
        got = restrict(prox.proximity_docs(None, None, ta[:1], tb[:1], "n", k, idx))
        want = {d for d, st in streams.items()
                if naive_window(st, ta[0], tb[0], k)}
        rx = None
        agrees_rx = None
        c = Check(f"/{k}", f"{a} ~ {b}", len(got), len(want), rx,
                  got == want, agrees_rx,
                  sorted(got - want)[:max_show], sorted(want - got)[:max_show])
        if k in (5, 25):
            rset = {d for d, t in raws.items() if regex_window(t, a, b, k)}
            c.regex_n = len(rset)
            c.agrees_regex = rset == got
            classify_regex_diff(c, got, rset, "window", [a, b],
                                [ta[0], tb[0]], streams, raws, k=k,
                                limit=classify)
        record(c)

    # ---- /s and /p -----------------------------------------------------
    for (a, b), kind in itertools.product(PAIRS, ("s", "p")):
        ta, tb = analyze_terms(a), analyze_terms(b)
        if not ta or not tb:
            continue
        got = restrict(prox.proximity_docs(None, None, ta[:1], tb[:1], kind, 0, idx))
        want = {d for d, st in streams.items()
                if naive_bucket(st, ta[0], tb[0], kind)}
        record(Check(f"/{kind}", f"{a} ~ {b}", len(got), len(want), None,
                     got == want, None,
                     sorted(got - want)[:max_show],
                     sorted(want - got)[:max_show]))

    n_naive = len(checks)
    ok_naive = sum(1 for c in checks if c.agrees_naive)
    rx_checks = [c for c in checks if c.agrees_regex is not None]
    ok_rx = sum(1 for c in rx_checks if c.agrees_regex)
    rx_union_docs = sum(c.rx_union_n for c in rx_checks)
    rx_diff_docs = sum(c.rx_diff_n for c in rx_checks)
    rx_same_docs = rx_union_docs - rx_diff_docs
    out = {
        "documents_verified": len(records),
        "stride": stride,
        "checks": n_naive,
        "naive_agreement": {"ok": ok_naive, "total": n_naive,
                            "rate": round(ok_naive / max(n_naive, 1), 4)},
        "regex_agreement": {"ok": ok_rx, "total": len(rx_checks),
                            "rate": round(ok_rx / max(len(rx_checks), 1), 4)},
        # Per *document* rather than per check: Jaccard over the two answer sets,
        # pooled across every mode-B check.  A check that differs on 1 document
        # out of 1,092 and one that differs on all of them both count as a single
        # disagreeing check above, and that is the figure that misleads.
        "regex_document_agreement": {
            "same": rx_same_docs,
            "differ": rx_diff_docs,
            "union": rx_union_docs,
            "rate": round(rx_same_docs / max(rx_union_docs, 1), 4),
            "classified": sum(c.rx_classified for c in rx_checks),
        },
        "by_family": {},
        "mismatch_classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
        "unexplained": classes.get(UNKNOWN, 0),
        "bugs": [c.as_json() for c in checks if not c.agrees_naive],
        "regex_differences": [c.as_json() for c in checks
                              if c.agrees_regex is False],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for fam in sorted({c.family for c in checks}):
        fam_checks = [c for c in checks if c.family == fam]
        out["by_family"][fam] = {
            "checks": len(fam_checks),
            "naive_ok": sum(1 for c in fam_checks if c.agrees_naive),
        }
    return out


def print_report(out: Dict[str, Any]) -> None:
    print("\n" + "=" * 76)
    na = out["naive_agreement"]
    rx = out["regex_agreement"]
    print(f"A. index vs naive scan : {na['ok']}/{na['total']} = {na['rate']:.2%}"
          f"   (must be 100%)")
    print(f"B. index vs raw regex  : {rx['ok']}/{rx['total']} = {rx['rate']:.2%}"
          f"   of checks identical (differences are tokenisation, below)")
    rd = out["regex_document_agreement"]
    print(f"   per document         : {rd['same']}/{rd['union']} = "
          f"{rd['rate']:.2%} of (index OR regex) answers agree; "
          f"{rd['differ']} differ")
    print(f"documents verified     : {out['documents_verified']} "
          f"(every {out['stride']}th, so the sample spans 1950-2025)")
    print(f"wall clock             : {out['seconds']}s")
    print("\nby operator family:")
    for fam, d in out["by_family"].items():
        print(f"  {fam:<10} {d['naive_ok']}/{d['checks']} agree with the naive scan")
    if out["bugs"]:
        print(f"\nBUGS - the index disagrees with the naive scan:")
        for b in out["bugs"]:
            print(f"  {b['family']} {b['label']}: index-only {b['index_only']}, "
                  f"naive-only {b['other_only']}")
    if out["mismatch_classes"]:
        rd = out["regex_document_agreement"]
        print(f"\nmode-B mismatching documents by class "
              f"({rd['classified']} of {rd['differ']} examined - "
              f"classifying one re-analyses every word in it):")
        for cls, n in out["mismatch_classes"].items():
            print(f"  {n:>3}  {cls}")
        print(f"\nunexplained: {out['unexplained']}  "
              f"(every examined document got a class)")
        print("\nmode-B differences per check (exact document counts):")
        for c in out["regex_differences"]:
            print(f"  {c['family']:<8} {c['label']:<34} "
                  f"index {c['index']:>5}  regex {c['regex']:>5}  "
                  f"index-only {c['regex_index_only_docs']:>4}  "
                  f"regex-only {c['regex_only_docs']:>4}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.eval.operators",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--brute-force", action="store_true", default=True,
                    help="run the brute-force comparison (the default)")
    ap.add_argument("--docs", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=11)
    ap.add_argument("--classify", type=int, default=0,
                    help="cap the mode-B documents explained per side per check "
                         "(0, the default, explains every one)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default=str(OPERATORS_JSON))
    args = ap.parse_args(argv)

    out = brute_force(docs=args.docs, stride=args.stride,
                      verbose=not args.quiet, classify=args.classify)
    print_report(out)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if out["bugs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
