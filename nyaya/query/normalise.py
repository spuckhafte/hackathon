"""Query normalisation — the query side of `analyze()`, plus query expansion.

The one rule this file exists to enforce
----------------------------------------
**A query term is normalised by exactly the same code that normalised the
documents.**  Every leaf of the parse tree goes through `nyaya.analyze`, so
``S. 302 IPC`` in a query becomes ``ipc§302`` just as it did in the index, and
``convicted`` becomes ``convict`` on both sides.  Anything written here that
bypasses `analyze()` is a recall bug waiting to happen.

Two things are legitimately *query-only*, and both are expansions rather than
rewrites — the index is never touched:

1. **Act fallback.**  A user who types ``sec. 302`` with no act means the Penal
   Code (`statutes.QUERY_FALLBACK_ACT`).  A *document* that says "section 302"
   with no act in scope gets no guess, because guessing there would fabricate a
   citation; guessing for a query only widens the candidate set, which recall
   can absorb.
2. **The IPC↔BNS crosswalk.**  ``ipc§302`` is OR-ed with ``bns§103`` at query
   time, because a 1985 judgment never mentioned the Bharatiya Nyaya Sanhita and
   writing ``bns§103`` into its postings would corrupt idf for every BNS term.

Statute canonicalisation runs on the **whole raw query string**, before lexing,
for the same reason it runs before tokenisation in `analyze()`: the patterns need
the punctuation.  A happy side effect is that the two-word phrase
``"section 302"`` collapses to the single token ``ipc§302`` — exactly what the
index holds, so the phrase match still works.

Author: Sumedha (M3)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nyaya import statutes
from nyaya.analyze import analyze_terms
from nyaya.contracts import FilterNode
from nyaya.metadata import CASE_TYPE_DOMAIN, DISPOSAL_DOMAIN
from nyaya.stoplist import is_scoring_stopword

# ==========================================================================
# Fields
# ==========================================================================
#: Parametric filter fields.  This is M2's domain exactly — `ParametricIndex`
#: consumes the `FilterNode`s we emit, so the names, ops and value types here
#: must match `metadata.PARAMETRIC_FIELDS` or the filter silently selects nothing.
FILTER_FIELDS: Tuple[str, ...] = ("year", "bench", "judge", "act", "type", "disposal")

#: Numeric fields take range operators; the rest are set/categorical lookups.
NUMERIC_FIELDS = frozenset({"year", "bench"})

#: Zone restriction fields: `title:(privacy)`.  From config.ZONES, so the two
#: never drift apart.
from nyaya import config                                     # noqa: E402

ZONE_FIELDS: Tuple[str, ...] = tuple(config.ZONES)

#: M2's `parametric._ALIASES`, mirrored so a user may type either spelling.
FIELD_ALIASES: Dict[str, str] = {
    "bench_size": "bench", "benchsize": "bench", "judges": "judge",
    "acts": "act", "case_type": "type", "casetype": "type", "cases": "type",
    "yr": "year", "decided": "year", "coram": "bench",
}

#: `bench` is both a zone (the coram line) and a filter (the bench size).  The
#: value decides: numeric or comparison -> filter, anything else -> zone.
AMBIGUOUS_FIELDS = frozenset(ZONE_FIELDS) & frozenset(FILTER_FIELDS)

OPS = ("!=", ">=", "<=", "=", ">", "<")

_RANGE_RE = re.compile(r"^(\d{3,4})\s*(?:-|\.\.|to)\s*(\d{3,4})$")


def canonical_field(name: str) -> str:
    f = str(name).strip().lower().rstrip(":")
    return FIELD_ALIASES.get(f, f)


def is_filter_field(name: str) -> bool:
    return canonical_field(name) in FILTER_FIELDS


def is_zone_field(name: str) -> bool:
    return canonical_field(name) in ZONE_FIELDS


# ==========================================================================
# Text normalisation
# ==========================================================================


def normalise_query(text: str) -> str:
    """Canonicalise statute mentions in the raw query, before lexing.

    Uses `statutes.QUERY_FALLBACK_ACT` so a bare "sec. 302" resolves; see the
    module docstring for why that is safe on the query side only.
    """
    if not text:
        return ""
    return statutes.canonicalise_statutes(
        text, default_act=statutes.QUERY_FALLBACK_ACT
    )


#: Whole-system switch for the stemming ablation.  The query side MUST match the
#: index side: running a stemmed query against an unstemmed index is not an
#: ablation, it is a broken system.  `nyaya.rank.ablate` flips this together
#: with the index path, and nothing else should touch it.
STEM_QUERIES = True


def terms_of(raw: str, *, stem: Optional[bool] = None) -> List[str]:
    """Normalised terms of one leaf, through `analyze()`.

    ``canonicalise=False`` because `normalise_query` already did pass 1 on the
    whole string; running it again would be harmless (it is idempotent) but
    pointless, and the leaf has lost the context that made it work.
    """
    stem = STEM_QUERIES if stem is None else stem
    return analyze_terms(raw, stem=stem, canonicalise=False)


def term_of(raw: str, *, stem: Optional[bool] = None) -> Optional[str]:
    """The single term a one-word leaf normalises to, or ``None`` if it vanished."""
    t = terms_of(raw, stem=stem)
    return t[0] if t else None


def phrase_terms(raw: str, *, stem: Optional[bool] = None) -> List[str]:
    """Terms of a quoted phrase, stop words **kept**.

    Keeping them is the whole point: the positional index stores every token, so
    ``"right to privacy"`` is an exact three-position match.  Dropping "to"
    here would turn it into a two-word phrase that matches "right of privacy".
    """
    return terms_of(raw, stem=stem)


#: Inflections used to find the shortest stem a typed root could produce.
#: See `prefix_root` — this is the fix for `litigat!` finding nothing.
_ROOT_INFLECTIONS = ("", "e", "es", "ed", "ing", "ion", "ions", "ity", "ies", "y")


def prefix_root(raw: str, *, stem: Optional[bool] = None) -> Optional[str]:
    """Root for ``bail!`` — the prefix for a dictionary scan.

    The subtlety that makes this more than ``raw.rstrip("!")``: the dictionary
    holds **stems**, so a prefix *longer than the stem* matches nothing.
    ``litigat!`` is the case that exposes it — Porter leaves "litigat" alone but
    turns "litigation" into "litig", so a scan for "litigat" returns an empty
    postings list while the user plainly meant litigation/litigant/litigate.

    So the prefix is the **shortest stem the typed root could produce**: stem the
    root together with a handful of inflections of it, keep only results that are
    genuine prefixes of the root (which rejects nonsense the stemmer invents for
    nonsense input), and take the shortest.  A shorter prefix is broader, and
    breadth is exactly what root expansion is for.
    """
    stem = STEM_QUERIES if stem is None else stem
    root = term_of(raw.rstrip("!*").strip(), stem=False)
    if not root or not stem:
        return root
    from nyaya.analyze import stem_term

    cands = {stem_term(root + suf) for suf in _ROOT_INFLECTIONS}
    usable = [c for c in cands if c and root.startswith(c)]
    return min(usable, key=len) if usable else stem_term(root)


def scoring_terms(terms: Iterable[str]) -> List[str]:
    """The tf-idf bag: the same terms with the scoring stop words removed."""
    from nyaya.stoplist import filter_scoring_terms

    return filter_scoring_terms(terms)


# ==========================================================================
# Query expansion  (fills ParsedQuery.expansions)
# ==========================================================================
_BARE_SECTION_RE = re.compile(r"^\d{1,3}[a-z]?$")


def build_expansions(terms: Iterable[str]) -> Dict[str, List[str]]:
    """``term -> OR alternatives``, for the planner to expand at query time.

    Two sources, both reversible and both switchable:

    * the verified IPC↔BNS/BNSS/BSA crosswalk, in whichever direction the user
      typed (``ipc§302`` -> ``bns§103`` and ``bns§103`` -> ``ipc§302``);
    * a bare number that survived canonicalisation (the user typed "302" with no
      "section" and no act) is offered ``ipc§302`` as an alternative.  That is a
      guess, so it is an *addition* to the original term, never a replacement —
      if the number meant something else, the original term still matches.
    """
    out: Dict[str, List[str]] = {}
    for t in terms:
        if statutes.is_statute_token(t):
            alts = statutes.crosswalk(t)
            if alts:
                out[t] = alts
        elif _BARE_SECTION_RE.match(t):
            guess = f"{statutes.QUERY_FALLBACK_ACT.lower()}§{t}"
            alts = [guess] + statutes.crosswalk(guess)
            out[t] = alts
    return out


def expansion_notes(expansions: Dict[str, List[str]]) -> List[str]:
    """Human-readable caveats for the "why this result" panel."""
    notes = []
    for term, alts in expansions.items():
        note = statutes.crosswalk_note(term)
        if note:
            notes.append(f"{term} -> {', '.join(alts)}: {note}")
        else:
            notes.append(f"{term} -> {', '.join(alts)}")
    return notes


# ==========================================================================
# Filter values
# ==========================================================================


def coerce_filter(field: str, op: str, value: Any) -> Tuple[Optional[FilterNode], List[str]]:
    """Turn a parsed ``field:op:value`` into a `FilterNode` M2's index accepts.

    Returns ``(node, warnings)``.  A bad value yields ``(None, [...])`` so the
    query degrades instead of failing: ``year:recent`` drops the filter and says
    why, it does not raise.
    """
    warnings: List[str] = []
    field = canonical_field(field)
    if field not in FILTER_FIELDS:
        return None, [f"unknown filter field {field!r}; "
                      f"known fields are {', '.join(FILTER_FIELDS)}"]

    values = value if isinstance(value, list) else [value]
    values = [v for v in (str(x).strip().strip("\"'") for x in values) if v]
    if not values:
        return None, [f"{field}: needs a value"]

    if field in NUMERIC_FIELDS:
        nums: List[int] = []
        for v in values:
            try:
                nums.append(int(float(v)))
            except ValueError:
                warnings.append(f"{field}:{v} ignored - {field} needs a number")
        if not nums:
            return None, warnings
        if op == "in" or len(nums) > 1:
            return FilterNode(field=field, op="in", value=nums), warnings
        if field == "year" and not (1900 <= nums[0] <= 2100):
            warnings.append(f"year:{nums[0]} looks wrong; the corpus covers "
                            f"{config.CORPUS_YEARS.start}-{config.CORPUS_YEARS.stop - 1}")
        if field == "bench" and not (1 <= nums[0] <= int(config.MAX_BENCH)):
            warnings.append(f"bench:{nums[0]} is outside 1-{int(config.MAX_BENCH)} "
                            "(the largest bench the Court has ever sat)")
        return FilterNode(field=field, op=op or "=", value=nums[0]), warnings

    # set / categorical: lower-cased, exactly as M2's _norm_value does
    vals = [v.lower() for v in values]
    if op not in ("=", "!=", "in"):
        warnings.append(f"{field} is not a numeric field, so {op!r} does not "
                        f"apply; using '='")
        op = "="
    domain = {"type": CASE_TYPE_DOMAIN, "disposal": DISPOSAL_DOMAIN}.get(field)
    if domain:
        for v in vals:
            if v not in domain:
                near = _closest(v, domain)
                warnings.append(
                    f"{field}:{v!r} is not in the closed vocabulary"
                    + (f"; did you mean {near!r}?" if near else "")
                    + f" (values: {', '.join(domain)})")
    if field == "act":
        vals = [v.upper() for v in vals]
        for v in vals:
            if v not in statutes.ACT_NAMES:
                warnings.append(f"act:{v} is not one of the "
                                f"{len(statutes.ACT_NAMES)} acts we canonicalise")
    if len(vals) > 1 or op == "in":
        return FilterNode(field=field, op="in", value=vals), warnings
    return FilterNode(field=field, op=op, value=vals[0]), warnings


def year_range(lo: str, hi: str) -> List[FilterNode]:
    """``year:2000-2010`` -> two conjunctive bounds, which is what S2 supports."""
    return [FilterNode(field="year", op=">=", value=int(lo)),
            FilterNode(field="year", op="<=", value=int(hi))]


def parse_range(value: str) -> Optional[Tuple[str, str]]:
    m = _RANGE_RE.match(str(value).strip())
    return (m.group(1), m.group(2)) if m else None


def _closest(value: str, domain: Iterable[str]) -> str:
    """Cheapest useful suggestion: longest shared prefix, then substring."""
    best, score = "", 0
    for cand in domain:
        n = len(_common_prefix(value, cand))
        if value in cand or cand in value:
            n += 3
        if n > score:
            best, score = cand, n
    return best if score >= 3 else ""


def _common_prefix(a: str, b: str) -> str:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return "".join(out)


__all__ = [
    "FILTER_FIELDS", "ZONE_FIELDS", "FIELD_ALIASES", "NUMERIC_FIELDS",
    "canonical_field", "is_filter_field", "is_zone_field",
    "normalise_query", "terms_of", "term_of", "phrase_terms", "prefix_root",
    "scoring_terms", "build_expansions", "expansion_notes", "STEM_QUERIES",
    "coerce_filter", "year_range", "parse_range", "is_scoring_stopword",
]
