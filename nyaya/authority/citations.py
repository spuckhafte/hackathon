"""Store S4, part 1 — citation extraction and resolution.

Member 2 (Suvrat).  Turns free text into graph edges:

    judgment text  ->  citation strings  ->  node_ids  ->  index/citations.jsonl

**Why we resolve against the 37,210-node universe, not the 13,942 indexed
documents.**  A judgment cites precedent, and precedent is old: a 2017 judgment
cites 1954.  If a cited judgment could only be an *indexed* document, roughly
two thirds of every citation would be thrown away and the graph would be far
too sparse for PageRank to say anything.  A universe node we never downloaded
still has a title, a year and two citations from the metadata, which is
everything needed to be the *target* of an edge.  So edges point into the
universe; only the citing end (``src``) has to be an indexed document, because
only an indexed document has text we could read the citation out of.

Resolution is a **cascade, most precise first**, and every edge records which
rung resolved it so the report can be honest about precision:

1. ``scr``     — exact ``SCR:year:vol:page`` key.  Unique over the universe
                 (0 collisions in 37,210 rows), so this rung cannot be wrong.
                 Includes the volume-less old form ``[1954] S.C.R. 1077`` via a
                 ``(year, page)`` key: in the eight single-volume years that is
                 still unique, and in multi-volume years it yields 2-3
                 candidates which the nearby case name disambiguates.
2. ``neutral`` — exact ``yearINSCnnnn`` key.
3. ``name``    — the ``X v. Y`` case name next to the citation, matched against
                 universe titles by Jaccard over generic-discounted title tokens
                 (``config.CITATION_NAME_THRESHOLD``) with a year check.  This is
                 the only rung available for **SCC, AIR and SCALE**, whose page
                 numbering does not exist anywhere in our metadata.
4. ``unresolved`` — recorded with ``dst = -1`` so the misses can be counted and
                 read, not silently dropped.

CLI::

    python -m nyaya.authority.citations build [--limit N]
    python -m nyaya.authority.citations stats
    python -m nyaya.authority.citations probe "...text..."
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

from nyaya import config
from nyaya.contracts import CitationEdge
from nyaya.metadata import GENERIC_TITLE_TOKENS, load_universe, title_tokens
from nyaya.utils import ensure_utf8_stdout, timed, write_jsonl

CITATIONS_JSONL = config.INDEX / "citations.jsonl"
CITATIONS_STATS = config.INDEX / "citations_stats.json"

#: How far either side of a reporter citation we look for an "X v. Y" name.
NAME_WINDOW_BEFORE = 180
NAME_WINDOW_AFTER = 40

#: A reporter year can be one off the decision year (a December judgment is
#: reported the following volume year), so name matching accepts +/- this.
NAME_YEAR_SLACK = 1

#: Two reporter citations this close together are one reference printed twice
#: ("[2024] 10 S.C.R. 108 : 2024 INSC 735"), not two citations.  The gap must
#: also be pure punctuation: ". See also A.I.R. 1963 S.C. 1295" is a *second*
#: citation even though it starts 10 characters later, and merging it would hide
#: a real edge.
MERGE_GAP = 6
_PARALLEL_GAP_RE = re.compile(r"^[\s:;,()\[\]/.=–—-]*$")

#: A name match needs at least this many discriminative tokens on each side, or
#: "State v. Ram" would match hundreds of judgments at any threshold.
MIN_NAME_TOKENS = 2

#: Candidate-set cap for name matching.  See :meth:`CitationResolver._name_candidates`
#: — token postings are unioned rarest-first and the scan stops here, which is
#: df-based index elimination applied to candidate generation.
MAX_NAME_CANDIDATES = 400


# ==========================================================================
# 1. Extraction — one master regex, scanned once per paragraph
# ==========================================================================
# Written against what these PDFs actually print, not against a style guide.
# Measured frequencies over a 1-in-40 sample of the corpus: SCR in 43.8% of
# judgments, SCC 25.9%, AIR 15.6%, SCALE 1.7%, and INSC in 0.0% — the neutral
# citation is a modern metadata artefact that the printed reports never carried.

_CITE_RE = re.compile(
    r"""(?ix)
    (?:
      # ---- Supreme Court Reports: [2017] 11 S.C.R. 1029, [1954] SCR 1077,
      #      [1959] SUPP. 1 S.C.R. 10, 1962 Supp. 3 SCR 1 ------------------
      \[?\s*(?P<scr_year>(?:1[89]\d\d|20[0-2]\d))\s*\]?\s*
      (?P<scr_supp> supp (?:l(?:ement(?:ary)?)?)? \.? )?\s*
      (?: \(\s*(?P<scr_vol_p>\d{1,2})\s*\) | (?P<scr_vol>\d{1,2}) )?\s*
      S\.?\s*C\.?\s*R\.?\s*
      (?:\(\s*\w{1,4}\s*\)\s*)?
      (?P<scr_page>\d{1,4})
      \b
    |
      # ---- neutral citation: 2023 INSC 456 ---------------------------------
      (?P<insc_year>(?:1[89]\d\d|20[0-2]\d))\s*INSC\s*(?P<insc_num>\d{1,5})\b
    |
      # ---- Supreme Court Cases: (2017) 10 SCC 1 ----------------------------
      \(\s*(?P<scc_year>(?:1[89]\d\d|20[0-2]\d))\s*\)\s*(?P<scc_vol>\d{1,2})\s*
      S\.?\s*C\.?\s*C\.?\s*(?:\(\s*\w{1,4}\s*\)\s*)?(?P<scc_page>\d{1,4})\b
    |
      # ---- All India Reporter: AIR 1963 SC 1295 ----------------------------
      A\.?\s*I\.?\s*R\.?\s*,?\s*(?P<air_year>(?:1[89]\d\d|20[0-2]\d))\s*
      S\.?\s*C\.?\s*(?P<air_page>\d{1,4})\b
    |
      # ---- SCALE: (1994) 4 SCALE 750 ---------------------------------------
      \(\s*(?P<scale_year>(?:1[89]\d\d|20[0-2]\d))\s*\)\s*(?P<scale_vol>\d{1,2})\s*
      S\.?\s*C\.?\s*A\.?\s*L\.?\s*E\.?\s*(?P<scale_page>\d{1,4})\b
    )
    """
)

REPORTERS = ("scr", "insc", "scc", "air", "scale")

#: Which reporters our metadata can address by an exact key.  SCC/AIR/SCALE page
#: numbers appear nowhere in the AWS release, so they can only resolve by name.
KEYED_REPORTERS = frozenset({"scr", "insc"})


@dataclass(frozen=True)
class CiteMatch:
    """One reporter citation found in text."""

    raw: str
    reporter: str
    year: int
    start: int
    end: int
    volume: str = ""        # "" | "3" | "S1" (supplement)
    page: int = 0
    number: int = 0         # INSC serial

    @property
    def scr_key(self) -> str:
        if self.reporter != "scr" or not self.volume:
            return ""
        return f"SCR:{self.year}:{self.volume}:{self.page}"

    @property
    def scr_year_page(self) -> Tuple[int, int]:
        return (self.year, self.page) if self.reporter == "scr" else (0, 0)

    @property
    def neutral_key(self) -> str:
        return f"{self.year}INSC{self.number}" if self.reporter == "insc" else ""


def find_citations(text: str) -> List[CiteMatch]:
    """Every reporter citation in ``text``, left to right."""
    out: List[CiteMatch] = []
    for m in _CITE_RE.finditer(text):
        g = m.groupdict()
        if g["scr_year"]:
            vol = g["scr_vol"] or g["scr_vol_p"] or ""
            if g["scr_supp"]:
                # A supplement volume lives in its own numbering space; with no
                # volume printed, "[1959] SUPP. S.C.R. 10" is supplement 1.
                vol = f"S{int(vol)}" if vol else "S1"
            out.append(CiteMatch(m.group(0).strip(), "scr", int(g["scr_year"]),
                                 m.start(), m.end(),
                                 volume=vol, page=int(g["scr_page"])))
        elif g["insc_year"]:
            out.append(CiteMatch(m.group(0).strip(), "insc", int(g["insc_year"]),
                                 m.start(), m.end(), number=int(g["insc_num"])))
        elif g["scc_year"]:
            out.append(CiteMatch(m.group(0).strip(), "scc", int(g["scc_year"]),
                                 m.start(), m.end(),
                                 volume=g["scc_vol"], page=int(g["scc_page"])))
        elif g["air_year"]:
            out.append(CiteMatch(m.group(0).strip(), "air", int(g["air_year"]),
                                 m.start(), m.end(), page=int(g["air_page"])))
        elif g["scale_year"]:
            out.append(CiteMatch(m.group(0).strip(), "scale", int(g["scale_year"]),
                                 m.start(), m.end(),
                                 volume=g["scale_vol"], page=int(g["scale_page"])))
    return out


def merge_parallel_citations(matches: Sequence[CiteMatch],
                             text: str = "") -> List[List[CiteMatch]]:
    """Group citations printed side by side — they are ONE reference.

    ``"[2024] 10 S.C.R. 108 : 2024 INSC 735"`` is a single precedent cited in
    two reporters.  Emitting two edges for it would double-count it in PageRank,
    so adjacent matches are grouped and the group resolves once, trying its most
    precise key first.

    Two citations are only merged when the text between them is short *and*
    contains nothing but punctuation.  Without that second test, "... 2024 INSC
    735. See also A.I.R. 1963 S.C. 1295" merges two unrelated precedents and the
    second edge disappears.
    """
    groups: List[List[CiteMatch]] = []
    for m in sorted(matches, key=lambda x: x.start):
        if groups:
            prev_end = groups[-1][-1].end
            gap = text[prev_end:m.start] if text else " " * (m.start - prev_end)
            if m.start - prev_end <= MERGE_GAP and _PARALLEL_GAP_RE.match(gap):
                groups[-1].append(m)
                continue
        groups.append([m])
    return groups


# ==========================================================================
# 2. Case names near a citation
# ==========================================================================
# "X v. Y" is the only handle on an SCC or AIR citation, and it also breaks ties
# when a volume-less SCR citation has two candidates.

_PARTY_WORD = r"(?:[A-Z][\w&.'()\-/]*|of|and|the|&|v\.|at)"
_NAME_RE = re.compile(
    rf"""
    \b(?P<left>[A-Z][\w&.'()\-/]*(?:\s+{_PARTY_WORD}){{0,8}}?)
    \s+(?:v|vs|versus)\.?\s+
    (?P<right>[A-Z][\w&.'()\-/]*(?:\s+{_PARTY_WORD}){{0,8}}?)
    (?=\s*[,.;:(\[]|\s+and\b|$)
    """,
    re.VERBOSE,
)


def name_near(text: str, start: int, end: int) -> str:
    """The ``X v. Y`` case name attached to the citation at ``[start, end)``.

    Reports print the name immediately before the citation
    ("*Kesavananda Bharati v. State of Kerala*, (1973) 4 SCC 225"), so the
    window before is wide and the window after is narrow; the *last* name in the
    preceding window is the nearest one.
    """
    lo = max(0, start - NAME_WINDOW_BEFORE)
    before = text[lo:start]
    hits = list(_NAME_RE.finditer(before))
    if hits:
        return hits[-1].group(0).strip()
    after = text[end:end + NAME_WINDOW_AFTER]
    m = _NAME_RE.search(after)
    return m.group(0).strip() if m else ""


def name_match_tokens(name: str) -> List[str]:
    """Discriminative tokens of a case name, for Jaccard against ``title_key``.

    Generic legal tokens are discounted exactly as ``metadata.title_key`` does,
    because otherwise "Union of India" matches every second judgment in the
    corpus and the threshold becomes meaningless.
    """
    toks = [t for t in title_tokens(name)
            if t not in GENERIC_TITLE_TOKENS and len(t) > 2 and t not in ("vs", "versus")]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ==========================================================================
# 3. The resolver
# ==========================================================================


class CitationResolver:
    """Maps a citation (and an optional case name) onto a universe ``node_id``."""

    def __init__(self, universe: Optional[pd.DataFrame] = None,
                 enable_pinpoint: bool = False):
        uni = load_universe() if universe is None else universe
        self.n_nodes = len(uni)
        #: Pinpoint page-range resolution.  Measured at ~89% wrong on this
        #: corpus and therefore off; see :meth:`_pinpoint` for the audit.
        self.enable_pinpoint = bool(enable_pinpoint)

        self.by_scr: Dict[str, int] = {}
        self.by_scr_yp: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.by_neutral: Dict[str, int] = {}
        self.node_year: Dict[int, int] = {}
        self.node_title: Dict[int, str] = {}
        self.node_tokens: Dict[int, Set[str]] = {}
        #: (year, token) -> node ids.  Candidate generation for name matching:
        #: without it a name would have to be compared against every judgment of
        #: its year (up to 1,535), which the corpus cannot afford.
        self.by_year_token: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        #: (year, volume) -> [(first_page, last_page, node_id)], for pinpoint
        #: citations; see :meth:`_pinpoint`.  Also keyed by year alone, for a
        #: citation that prints no volume.
        self.page_ranges: Dict[Any, List[Tuple[int, int, int]]] = defaultdict(list)

        for node_id, year, scr_key, neutral_key, title, key in zip(
            uni["node_id"], uni["year"], uni["scr_key"], uni["neutral_key"],
            uni["title"], uni["title_key"],
        ):
            node_id, year = int(node_id), int(year)
            self.node_year[node_id] = year
            self.node_title[node_id] = str(title)
            if scr_key:
                self.by_scr[str(scr_key)] = node_id
                parts = str(scr_key).split(":")
                self.by_scr_yp[(int(parts[1]), int(parts[3]))].append(node_id)
            if neutral_key:
                self.by_neutral.setdefault(str(neutral_key), node_id)
            toks = {t for t in str(key).split() if len(t) > 2}
            self.node_tokens[node_id] = toks
            for t in toks:
                self.by_year_token[(year, t)].append(node_id)

        self._build_page_ranges(uni)

    def _build_page_ranges(self, uni: pd.DataFrame) -> None:
        """Printed page span of every judgment, taken from its bucket key.

        The ``path`` is ``[S_]year_volume_firstpage_lastpage``
        (``1950_1_1008_1018``, ``S_1959_1_10_27``), and the volume and first page
        agree with the parsed SCR citation in 100% of the 37,210 universe rows —
        so the key also hands us the *last* page for free.  That is what makes
        pinpoint citations resolvable.
        """
        parsed = uni["path"].astype(str).str.extract(
            r"^(?P<supp>S)?_?(?P<year>\d{4})_(?P<vol>\d+)_(?P<p1>\d+)_(?P<p2>\d+)$"
        )
        bad = 0
        for (supp, year, vol, p1, p2), node_id in zip(parsed.values, uni["node_id"]):
            if year is None or pd.isna(year):
                bad += 1
                continue
            volume = f"S{int(vol)}" if supp == "S" else str(int(vol))
            span = (int(p1), int(p2), int(node_id))
            self.page_ranges[(int(year), volume)].append(span)
            self.page_ranges[int(year)].append(span)       # volume-less fallback
        for k in self.page_ranges:
            self.page_ranges[k].sort()
        self.unparsed_paths = bad

    def _pinpoint(self, m: CiteMatch) -> List[int]:
        """Judgments whose printed page span contains the cited page.

        **MEASURED AND REJECTED — off by default.  See ``enable_pinpoint``.**

        The idea was sound: a citation often points at the page a passage is on
        rather than the judgment's first page ("[1950] S.C.R. 87" for a judgment
        running 75-87), so range containment should recover citations that miss
        the exact first-page key.  It recovered 4.46% of all occurrences, which
        looked like a clear win.

        It is not.  Two properties of this corpus break it:

        1. The ``p2`` in a bucket key is *not* consistently the last page.  In
           the 1950 volume 1, 19 of 41 judgments' ranges overlap their
           neighbour's because ``p2`` is sometimes the next judgment's first page.
           Boundary pages therefore belong to two judgments at once.
        2. The OCR truncates page numbers.  "[1950] S.C.R. 869" (Chiranjit Lal,
           the Sholapur Mills case) is read as "[1950] S.C.R. 86", which falls
           inside an unrelated judgment's range and produces a confident,
           completely wrong edge.

        Audited on a 1-in-3 sample of the first 4,000 judgments, using the nearby
        case name as a proxy for ground truth: of the pinpoint resolutions where a
        name was available to check, **31 disagreed with the resolved title and
        only 4 agreed** — about 89% wrong.  A wrong edge is worse than a missing
        one because PageRank propagates it, so the rung is disabled.  It is kept
        here, behind a flag, so the negative result stays reproducible and so the
        report can show the measurement rather than assert it.
        """
        if not self.enable_pinpoint or m.reporter != "scr" or not m.page:
            return []
        key = (m.year, m.volume) if m.volume else m.year
        # Strictly interior: the endpoints are the ambiguous ones, and an exact
        # first-page citation has already been handled by rungs 1 and 3.
        return [n for (a, b, n) in self.page_ranges.get(key, ()) if a < m.page < b]

    # -- the cascade ------------------------------------------------------
    def resolve(self, group: Sequence[CiteMatch], name: str = "") -> Tuple[int, str, str]:
        """``(node_id, resolved_by, method)``; ``node_id`` is -1 when unresolved.

        ``resolved_by`` stays inside the contract's vocabulary
        (``scr`` | ``neutral`` | ``name`` | ``unresolved``); ``method`` is the
        finer label used only for the resolution-rate report.
        """
        # --- rung 1: exact SCR key (unique over the universe) -------------
        for m in group:
            if m.scr_key and m.scr_key in self.by_scr:
                return self.by_scr[m.scr_key], "scr", "scr_exact"

        # --- rung 2: exact neutral key -----------------------------------
        for m in group:
            nk = m.neutral_key
            if nk and nk in self.by_neutral:
                return self.by_neutral[nk], "neutral", "neutral_exact"

        # --- rung 3: volume-less SCR, by (year, page) --------------------
        name_toks = set(name_match_tokens(name))
        for m in group:
            if m.reporter != "scr" or m.volume:
                continue
            cands = self.by_scr_yp.get(m.scr_year_page, [])
            if len(cands) == 1:
                return cands[0], "scr", "scr_novol_unique"
            if len(cands) > 1 and name_toks:
                best, why = self._best_by_name(cands, name_toks)
                if best >= 0:
                    return best, "scr", f"scr_novol_name_{why}"

        # --- rung 3b: pinpoint page inside a judgment's printed span ------
        for m in group:
            cands = self._pinpoint(m)
            if len(cands) == 1:
                return cands[0], "scr", "scr_pinpoint_unique"
            if len(cands) > 1 and name_toks:
                best, why = self._best_by_name(cands, name_toks)
                if best >= 0:
                    return best, "scr", f"scr_pinpoint_name_{why}"

        # --- rung 4: case name + reporter year (the only road for SCC/AIR) --
        if len(name_toks) >= MIN_NAME_TOKENS:
            years = {m.year for m in group}
            cands = self._name_candidates(name_toks, years)
            best, why = self._best_by_name(cands, name_toks)
            if best >= 0:
                return best, "name", f"name_{why}"
            return -1, "unresolved", f"unresolved_name_{why}"

        return -1, "unresolved", "unresolved_no_name"

    # -- name matching ----------------------------------------------------
    def _name_candidates(self, name_toks: Set[str], years: Iterable[int]) -> List[int]:
        """Nodes in the plausible years sharing a *discriminative* title token.

        Candidate generation is where this gets expensive, and the fix is the
        ordinary IR one: process the token postings **rarest first** and stop
        once there are enough candidates.

        A name token like "singh", "kumar" or "ram" occurs in hundreds of titles
        in a single year.  Unioning its bucket adds thousands of candidates that
        a Jaccard comparison then has to score one by one — measured at 0.65 s
        per modern judgment, 50x the cost of a 1950s one — and it adds almost no
        chance of finding the right answer, because the answer, if it is in the
        corpus at all, also shares the name's *rare* tokens.  So buckets are
        taken in increasing document frequency and the scan stops at
        :data:`MAX_NAME_CANDIDATES`; the rarest token's bucket is always included
        so a name made only of common words still gets its best shot.
        """
        buckets: List[List[int]] = []
        for y in set(years):
            for yy in range(y - NAME_YEAR_SLACK, y + NAME_YEAR_SLACK + 1):
                for t in name_toks:
                    b = self.by_year_token.get((yy, t))
                    if b:
                        buckets.append(b)
        if not buckets:
            return []
        buckets.sort(key=len)
        out: Set[int] = set(buckets[0])
        for b in buckets[1:]:
            if len(out) >= MAX_NAME_CANDIDATES:
                break
            out.update(b)
        return sorted(out)

    def _best_by_name(self, cands: Sequence[int],
                      name_toks: Set[str]) -> Tuple[int, str]:
        """Best candidate for a cited case name, or ``(-1, reason)``.

        Two accept rules, because Jaccard alone is the wrong measure here and the
        corpus shows exactly why.

        **Rule A — Jaccard >= config.CITATION_NAME_THRESHOLD.**  Symmetric, and
        correct when the cited name and the cause title are about the same
        length.

        **Rule B — subset.**  Jaccard is *systematically* biased against long
        cause titles, because a judgment is cited by an abbreviated name while
        the metadata holds the full title.  "Kesavananda Bharati v. State of
        Kerala" gives tokens {kesavananda, bharati, kerala}; the real title is
        "HIS HOLINESS KESAVANANDA BHARATI SRIPADAGALAVARU versus STATE OF
        KERALA", so the union is 6, the intersection 3, and Jaccard is 0.50 —
        under the 0.60 threshold, for the single most famous judgment in Indian
        constitutional law.  Raising the threshold would be backwards; the right
        fix is to notice that an abbreviation is a *subset*, not a near-equal
        set.  So: if exactly one candidate contains **every** token of the cited
        name, accept it.  Uniqueness is what keeps this precise — "Maneka Gandhi
        v. Union of India" is a subset of exactly one 1978 title, and when a name
        is a subset of two different judgments we refuse.

        Either way a tie is a refusal: a wrong edge is worse than a missing one,
        because PageRank propagates it to every neighbour.
        """
        if not name_toks:
            return -1, "no_name_tokens"

        # --- Rule A: symmetric Jaccard ------------------------------------
        best, best_score, runner = -1, 0.0, 0.0
        for c in cands:
            s = jaccard(name_toks, self.node_tokens.get(c, set()))
            if s > best_score:
                best, runner, best_score = c, best_score, s
            elif s > runner:
                runner = s
        if best_score >= config.CITATION_NAME_THRESHOLD and best_score > runner:
            return best, "jaccard"

        # --- Rule B: the cited name is an abbreviation of the full title ---
        if len(name_toks) >= MIN_NAME_TOKENS:
            subsets = [c for c in cands
                       if name_toks <= self.node_tokens.get(c, set())]
            if len(subsets) == 1:
                return subsets[0], "subset"
            if len(subsets) > 1:
                return -1, "subset_ambiguous"
        return -1, "below_threshold"


# ==========================================================================
# 4. Sentence window — the evidence M1's treatment classifier reads
# ==========================================================================


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """``[(start, end)]`` of each sentence, walked back onto the original string.

    Cached on the paragraph text: a long modern paragraph routinely holds a dozen
    citations, and re-splitting it once per citation was a measurable part of the
    build cost.
    """
    from nyaya.analyze import sentences

    sents = sentences(text)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for s in sents:
        i = text.find(s, cursor)
        if i < 0:
            i = cursor
        spans.append((i, i + len(s)))
        cursor = i + len(s)
    return spans


_SPAN_CACHE: Dict[int, Tuple[str, List[Tuple[int, int]]]] = {}


def sentence_spans_cached(text: str) -> List[Tuple[int, int]]:
    key = hash(text)
    hit = _SPAN_CACHE.get(key)
    if hit is not None and hit[0] == text:
        return hit[1]
    spans = _sentence_spans(text)
    if len(_SPAN_CACHE) > 64:
        _SPAN_CACHE.clear()
    _SPAN_CACHE[key] = (text, spans)
    return spans


def sentence_window(text: str, pos: int, n_after: int = 1) -> str:
    """The sentence containing offset ``pos`` plus the next ``n_after``.

    The treatment of a precedent is very often in the *following* sentence
    ("... (1973) 4 SCC 225. That decision was overruled in ..."), so the window
    deliberately runs past the citation.  Boundaries come from
    ``analyze.sentences`` so they are the same ones ``analyze()`` numbered.
    """
    spans = sentence_spans_cached(text)
    if not spans:
        return text.strip()
    hit = next((k for k, (a, b) in enumerate(spans) if a <= pos < b), None)
    if hit is None:
        hit = max(0, min(len(spans) - 1,
                         next((k for k, (a, _) in enumerate(spans) if a > pos),
                              len(spans)) - 1))
    lo = spans[hit][0]
    hi = spans[min(hit + n_after, len(spans) - 1)][1]
    return text[lo:hi].strip()


# ==========================================================================
# 5. Build
# ==========================================================================

#: Paragraph zones worth scanning.  The bench/title/parties zones are names and
#: the statutes zone is canonical tokens; neither carries case citations.
SCAN_ZONES = ("body", "order")


def _iter_judgments(path: Path = None, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    p = Path(path or config.JUDGMENTS_JSONL)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — M1's ingest has not produced it yet")
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_edges(resolver: CitationResolver,
                  judgments: Iterable[Dict[str, Any]],
                  src_node: Dict[int, int],
                  progress_every: int = 2000) -> Tuple[List[CitationEdge], Dict[str, Any]]:
    """Scan every judgment's paragraphs and resolve every citation found."""
    edges: List[CitationEdge] = []
    tally: Counter = Counter()
    by_reporter: Counter = Counter()
    resolved_by_reporter: Counter = Counter()
    method: Counter = Counter()
    docs_with_cites = 0
    n_docs = 0
    unresolved_examples: List[str] = []

    for rec in judgments:
        n_docs += 1
        if progress_every and n_docs % progress_every == 0:
            print(f"  ... {n_docs:,} judgments, {len(edges):,} edges",
                  file=sys.stderr)
        doc_id = int(rec.get("doc_id", -1))
        self_node = src_node.get(doc_id, -1)
        found_any = False

        for para in rec.get("paras", ()):
            if para.get("zone") not in SCAN_ZONES:
                continue
            text = para.get("text") or ""
            if not text:
                continue
            matches = find_citations(text)
            if not matches:
                continue
            for group in merge_parallel_citations(matches, text):
                head = group[0]
                for m in group:
                    by_reporter[m.reporter] += 1
                tally["occurrences"] += 1
                found_any = True
                name = name_near(text, head.start, group[-1].end)
                dst, resolved_by, meth = resolver.resolve(group, name)
                method[meth] += 1
                if dst >= 0:
                    for m in group:
                        resolved_by_reporter[m.reporter] += 1
                if dst >= 0 and dst == self_node:
                    tally["self_citation_dropped"] += 1
                    continue
                if dst < 0:
                    tally["unresolved"] += 1
                    if len(unresolved_examples) < 40:
                        unresolved_examples.append(
                            f"{head.raw}  |name: {name[:60]!r}"
                        )
                else:
                    tally["resolved"] += 1
                edges.append(
                    CitationEdge(
                        src=doc_id,
                        dst=dst,
                        treatment="",          # M1's classifier fills this in wave 3
                        sentence=sentence_window(text, head.start, n_after=1),
                        para_id=int(para.get("id") or 0),
                        cite_string=" : ".join(m.raw for m in group),
                        resolved_by=resolved_by,
                    )
                )
        if found_any:
            docs_with_cites += 1

    stats: Dict[str, Any] = {
        "judgments_scanned": n_docs,
        "judgments_with_a_citation": docs_with_cites,
        "citation_occurrences": int(tally["occurrences"]),
        "resolved": int(tally["resolved"]),
        "unresolved": int(tally["unresolved"]),
        "self_citations_dropped": int(tally["self_citation_dropped"]),
        "resolution_rate": (tally["resolved"] / tally["occurrences"]
                            if tally["occurrences"] else 0.0),
        "by_method": dict(method),
        "by_reporter": dict(by_reporter),
        "resolved_by_reporter": dict(resolved_by_reporter),
        "unresolved_examples": unresolved_examples,
    }
    return edges, stats


def build(limit: Optional[int] = None, verbose: bool = True) -> Tuple[List[CitationEdge], Dict[str, Any]]:
    uni = load_universe()
    resolver = CitationResolver(uni)
    if verbose:
        print(f"resolver: {resolver.n_nodes:,} universe nodes | "
              f"{len(resolver.by_scr):,} SCR keys | "
              f"{len(resolver.by_neutral):,} neutral keys | "
              f"{len(resolver.by_year_token):,} (year, token) buckets",
              file=sys.stderr)

    # doc_id -> node_id, so a judgment never cites itself into the graph.
    from nyaya.metadata import load_manifest
    man = load_manifest()
    src_node = {int(d): int(n) for d, n in zip(man["doc_id"], man["node_id"])}

    edges, stats = extract_edges(resolver, _iter_judgments(limit=limit), src_node)
    stats["universe_nodes"] = resolver.n_nodes
    stats["distinct_targets"] = len({e.dst for e in edges if e.dst >= 0})
    stats["edges_written"] = len(edges)
    return edges, stats


def write_edges(edges: Sequence[CitationEdge], stats: Dict[str, Any]) -> Path:
    from dataclasses import asdict

    config.INDEX.mkdir(parents=True, exist_ok=True)
    write_jsonl(CITATIONS_JSONL, (asdict(e) for e in edges))
    CITATIONS_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    return CITATIONS_JSONL


def load_edges(path: Path = None, resolved_only: bool = True) -> List[CitationEdge]:
    """Read the edge list back.  Unresolved rows (``dst == -1``) are kept on disk
    for error analysis but excluded from the graph by default."""
    from nyaya.utils import read_jsonl

    p = Path(path or CITATIONS_JSONL)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run `python -m nyaya.authority.citations build`"
        )
    out = []
    for row in read_jsonl(p):
        e = CitationEdge(**row)
        if resolved_only and e.dst < 0:
            continue
        out.append(e)
    return out


# ==========================================================================
# 6. Stats / CLI
# ==========================================================================


def print_stats(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 68)
    p("S4 part 1 — citation extraction and resolution")
    p("=" * 68)
    p(f"judgments scanned        : {s['judgments_scanned']:,}")
    p(f"  with >=1 citation      : {s['judgments_with_a_citation']:,} "
      f"({s['judgments_with_a_citation'] / max(s['judgments_scanned'], 1):.1%})")
    p(f"citation occurrences     : {s['citation_occurrences']:,}")
    p(f"  resolved               : {s['resolved']:,}  ({s['resolution_rate']:.2%})")
    p(f"  unresolved             : {s['unresolved']:,}")
    p(f"  self-citations dropped : {s['self_citations_dropped']:,}")
    p(f"edges written            : {s.get('edges_written', 0):,} "
      f"(distinct targets {s.get('distinct_targets', 0):,} of "
      f"{s.get('universe_nodes', 0):,} universe nodes)")
    p("")
    p("by resolution rung (share of all occurrences)")
    tot = max(s["citation_occurrences"], 1)
    for meth, n in sorted(s["by_method"].items(), key=lambda t: -t[1]):
        p(f"  {meth:<20} {n:>8,}  {n / tot:6.2%}")
    p("")
    p("by reporter (occurrences / resolved / rate)")
    for rep in REPORTERS:
        n = s["by_reporter"].get(rep, 0)
        r = s["resolved_by_reporter"].get(rep, 0)
        keyed = "exact key available" if rep in KEYED_REPORTERS else "name+year only"
        p(f"  {rep.upper():<6} {n:>8,} / {r:>8,} / "
          f"{(r / n if n else 0):6.1%}   ({keyed})")
    p("")
    if s.get("unresolved_examples"):
        p("a few unresolved citations (for the error analysis):")
        for ex in s["unresolved_examples"][:8]:
            p(f"  {ex}")
    p("=" * 68)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.authority.citations",
                                 description="Extract and resolve citations (S4).")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=None,
                   help="only the first N judgments (for a quick check)")
    sub.add_parser("stats")
    pr = sub.add_parser("probe", help="show what the extractor sees in a string")
    pr.add_argument("text")
    args = ap.parse_args(argv)

    if args.command == "build":
        with timed("citations"):
            edges, stats = build(limit=args.limit)
            out = write_edges(edges, stats)
        print(f"wrote {len(edges):,} edges -> {out}")
        print_stats(stats)
    elif args.command == "stats":
        if not CITATIONS_STATS.exists():
            raise SystemExit("no citations_stats.json — run `build` first")
        print_stats(json.loads(CITATIONS_STATS.read_text(encoding="utf-8")))
    else:
        resolver = CitationResolver()
        matches = find_citations(args.text)
        print(f"{len(matches)} citation(s):")
        for group in merge_parallel_citations(matches, args.text):
            head = group[0]
            name = name_near(args.text, head.start, group[-1].end)
            dst, by, meth = resolver.resolve(group, name)
            title = resolver.node_title.get(dst, "(unresolved)")
            print(f"  {' : '.join(m.raw for m in group)}")
            print(f"    name near : {name!r}")
            print(f"    -> node {dst} via {meth}  [{by}]  {title[:70]}")
            print(f"    sentence  : {sentence_window(args.text, head.start)[:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
