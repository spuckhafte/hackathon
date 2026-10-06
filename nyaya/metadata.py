"""Metadata normalisation: the universe of graph nodes and the indexed manifest.

Member 2 (Suvrat).  This module is the single source of **parametric** facts for
NyayaTrace.  The parametric / zone distinction matters and is made explicit here
(Manning, IIR ch. 6.1):

* A **parametric field** holds a structured value drawn from a *known, finite
  domain* — a year is an integer, a disposal is one of a dozen strings, a judge
  is a member of a set of names.  Parametric fields are queried with
  *equality / range / membership* predicates (``year:>=2000``, ``bench:>=5``,
  ``judge:chandrachud``) and answered by a separate parametric index (store S2),
  never by the free-text postings.  Nothing here is stemmed, scored, or
  tf-idf weighted: a filter either admits a document or it does not.
* A **zone** holds *free text* (title, parties, bench line, statutes, order,
  body).  Zones are tokenised by ``analyze()``, inverted into the positional
  index (store S3) and scored.  Member 1 owns those.

The same raw column can feed both: ``title`` is a zone (tokenised and scored
with weight 3.0) *and* the source of the parametric ``case_type`` and of
``title_key``/``title_tokens``, which wave 2 uses for Jaccard case-name matching
when resolving a citation that carries no reporter citation.

Two frames are produced:

``data/processed/universe.parquet``
    Every English judgment in the release, de-duplicated on ``path``.  These are
    the **nodes of the citation graph** — a 1963 judgment we never download is
    still a legitimate target of a citation and still needs a ``node_id``.

``data/processed/manifest.parquet``
    The stratified sample we actually index (<= ``config.PER_YEAR_CAP`` per year,
    drawn with ``config.SAMPLE_SEED``).  ``doc_id`` is dense 0..M-1 over this
    frame; ``node_id`` is carried across so the indexed set is a strict subset of
    the graph nodes.

CLI::

    python -m nyaya.metadata universe
    python -m nyaya.metadata manifest
    python -m nyaya.metadata stats
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from nyaya import config
from nyaya.utils import ensure_utf8_stdout, timed

# --------------------------------------------------------------------------
# Raw-column handling
# --------------------------------------------------------------------------

#: Columns dropped on the way in.  ``raw_html`` is the scraped <select> blob for
#: the language dropdown (tens of KB per row, no retrieval value) and
#: ``description`` is empty in 100% of rows in all 76 partitions.
DROP_COLUMNS = ("raw_html", "description")

#: Needed to decide "is this judgment available in English at all".
ENGLISH_CODE = "ENG"


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------
# !!  WARNING — THE `judge` METADATA COLUMN IS NOT A BENCH LIST.  !!
# Of the 37,210 de-duplicated English judgments, 32,509 (87.4%) carry exactly
# ONE name in this column, and only 925 carry three or more.  The Supreme Court
# of India has never sat as a single judge: the constitutional minimum is two
# and most reported matters are two- or three-judge benches.  The column is in
# practice the *authoring* judge, or whoever the scraper happened to capture.
#
# Therefore:  `bench_size_meta` derived here is a LOWER BOUND and a FALLBACK
# ONLY.  The real bench is parsed by Rakshit (M1) from the PDF's CORAM /
# "[... , JJ.]" line, and `merge_from_record()` below prefers that value.
# Any g(d) bench component (config.G_BENCH_WEIGHT) must be computed from the
# merged `bench_size`, never from `bench_size_meta`, or every judgment in the
# corpus looks like a one-judge bench.

_HONORIFIC_RE = re.compile(
    r"""(?ix)
    \b(
        hon\s*[''`]?\s*ble | honourable | honorable |
        the\s+chief\s+justice | chief\s+justice | justice |
        mr\.? | mrs\.? | ms\.? | smt\.? | shri | sri | dr\.? |
        j{1,2}\.? |          # J.  JJ.
        cji
    )\b\.?
    """
)
_TRAILING_JUNK_RE = re.compile(r"[\*\.,;:\s]+$")
_LEADING_JUNK_RE = re.compile(r"^[\*\.,;:\s]+")
_WS_RE = re.compile(r"\s+")


def clean_judge(name: str) -> str:
    """Strip honorifics and punctuation noise from one judge name.

    ``"HON'BLE MR. JUSTICE D.Y. CHANDRACHUD*"`` -> ``"D.Y. CHANDRACHUD"``.
    Case is normalised to upper, which is what 99% of the raw column already
    uses, so display and matching agree.
    """
    if not name:
        return ""
    s = _HONORIFIC_RE.sub(" ", str(name))
    s = s.replace("*", " ")
    s = _WS_RE.sub(" ", s).strip()
    s = _TRAILING_JUNK_RE.sub("", s)
    s = _LEADING_JUNK_RE.sub("", s)
    # Collapse "D. Y. CHANDRACHUD" -> "D.Y. CHANDRACHUD" for stable keys.
    s = re.sub(r"\b([A-Za-z])\.\s+(?=[A-Za-z]\.)", r"\1.", s)
    return s.upper().strip()


def split_judges(raw: str) -> List[str]:
    """``"R.K. AGRAWAL, ABHAY MANOHAR SAPRE"`` -> two cleaned names."""
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", str(raw))
    out: List[str] = []
    for p in parts:
        c = clean_judge(p)
        # A bare initial block ("R.K.") left over from a bad split is not a name.
        if len(c) >= 3 and re.search(r"[A-Z]{3}", c) and c not in out:
            out.append(c)
    return out


def judge_keys(judges: Sequence[str]) -> List[str]:
    """Lower-cased lookup keys for the parametric set-field ``judge``.

    Both the full name and the surname are emitted so that ``judge:chandrachud``
    and ``judge:"d.y. chandrachud"`` both hit.  This is a *set* field: the
    predicate is membership, not similarity.
    """
    keys: List[str] = []
    for j in judges:
        low = j.lower().strip()
        if not low:
            continue
        if low not in keys:
            keys.append(low)
        surname = low.split()[-1] if low.split() else ""
        surname = surname.strip(".")
        if len(surname) > 2 and surname not in keys:
            keys.append(surname)
    return keys


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------
# Both citation columns are near-100% populated (neutral 99.81%, SCR 100.00%).
# That is the single biggest win available for wave 2: a citation found in a
# judgment's text that carries either form resolves by an exact dictionary
# lookup, with no name matching and no false positives.  Fuzzy Jaccard name
# matching (config.CITATION_NAME_THRESHOLD) is only the last resort.

_NEUTRAL_RE = re.compile(r"(?i)\b(\d{4})\s*INSC\s*(\d+)\b")
_SCR_RE = re.compile(
    r"""(?ix)
    \[?\s*(\d{4})\s*\]?             # year
    \s*(supp(?:l)?\.?|suppl?ement)? # 20.5% of rows are supplement volumes:
                                    #   "[1959] SUPP. 1 S.C.R. 10"
    \s*\(?\s*(\d+)\s*\)?            # volume
    \s*S\.?\s*C\.?\s*R\.?           # S.C.R.
    \s*(?:\(\s*\w+\s*\)\s*)?
    \s*(\d+)                        # first page
    """
)


def normalise_neutral(raw: str) -> str:
    """``"2017 INSC 1136"`` -> ``"2017INSC1136"``.  Exact-lookup key."""
    if not raw:
        return ""
    m = _NEUTRAL_RE.search(str(raw))
    return f"{m.group(1)}INSC{int(m.group(2))}" if m else ""


def normalise_scr(raw: str) -> str:
    """``"[2017] 11 S.C.R. 1029"`` -> ``"SCR:2017:11:1029"``.  Exact-lookup key.

    A supplement volume gets its own volume token so it cannot collide with the
    regular volume of the same number:
    ``"[1959] SUPP. 1 S.C.R. 10"`` -> ``"SCR:1959:S1:10"``.
    """
    if not raw:
        return ""
    m = _SCR_RE.search(str(raw))
    if not m:
        return ""
    vol = f"S{int(m.group(3))}" if m.group(2) else str(int(m.group(3)))
    return f"SCR:{m.group(1)}:{vol}:{int(m.group(4))}"


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------
# A Supreme Court cause title is dominated by boilerplate: thousands of cases
# are "X versus UNION OF INDIA" or "STATE OF MAHARASHTRA versus Y".  Raw Jaccard
# over such titles says every State case resembles every other State case, so
# the generic tokens below are stripped from `title_key` (the *discriminative*
# token set used for matching) while `title_tokens` keeps everything (needed to
# confirm a match once the discriminative part already agrees).

GENERIC_TITLE_TOKENS = frozenset(
    {
        "the", "of", "and", "ors", "ors.", "anr", "anr.", "another", "others",
        "versus", "vs", "v", "etc", "etc.", "through", "its", "by", "for",
        "union", "india", "state", "states", "govt", "government", "no", "nos",
        "ltd", "pvt", "co", "company", "corporation", "corpn", "smt", "shri",
        "sri", "mr", "mrs", "ms", "m/s", "dr", "late", "deceased", "appellant",
        "appellants", "respondent", "respondents", "petitioner", "petitioners",
    }
)

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def title_tokens(title: str) -> List[str]:
    """Full lower-cased, punctuation-stripped token list of a cause title."""
    if not title:
        return []
    s = _PUNCT_RE.sub(" ", str(title).lower())
    return [t for t in s.split() if t]


def title_key(title: str) -> str:
    """Discriminative part of the title, space-joined, for Jaccard matching.

    Generic legal tokens are removed so that the similarity between two titles
    reflects the *parties*, not the boilerplate.  If a title is entirely
    boilerplate (e.g. "STATE OF U.P. versus UNION OF INDIA") the full token set
    is kept rather than returning an empty key, which would match everything.
    """
    toks = title_tokens(title)
    kept = [t for t in toks if t not in GENERIC_TITLE_TOKENS and len(t) > 1]
    if not kept:
        kept = toks
    # dedupe, preserve order
    seen, out = set(), []
    for t in kept:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


# --------------------------------------------------------------------------
# Disposal and case type
# --------------------------------------------------------------------------
# `disposal_nature` is a genuine controlled vocabulary (15 distinct values over
# 37k rows) so it is a clean categorical parametric field.  We fold it to a
# small canonical domain for `disposal:allowed` style queries while keeping the
# raw string for display.

DISPOSAL_DOMAIN = (
    "allowed", "partly allowed", "dismissed", "disposed", "directions issued",
    "referred to larger bench", "reference answered", "remitted", "adjourned",
    "other",
)


def canonical_disposal(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "partly" in s:
        return "partly allowed"
    if "allowed" in s:
        return "allowed"
    if "dismiss" in s:
        return "dismissed"
    if "disposed" in s:
        return "disposed"
    if "larger bench" in s or "referred" in s:
        return "referred to larger bench"
    if "reference answered" in s:
        return "reference answered"
    if "remit" in s:
        return "remitted"
    if "adjourn" in s:
        return "adjourned"
    if "direction" in s:
        return "directions issued"
    if "leave granted" in s:
        return "allowed" if "allow" in s else "disposed"
    return "other"


CASE_TYPE_DOMAIN = (
    "civil appeal", "criminal appeal", "writ petition",
    "special leave petition", "review petition", "transfer petition",
    "contempt petition", "suo motu", "other",
)

#: Ordered: the first pattern that matches wins, so the more specific ones come
#: first ("special leave petition" before the bare "petition").
_CASE_TYPE_PATTERNS = (
    ("suo motu", re.compile(r"(?i)\bsuo\s*[- ]?\s*mot[ou]\b")),
    ("contempt petition", re.compile(r"(?i)\bcontempt\b")),
    ("transfer petition", re.compile(r"(?i)\btransfer(?:red)?\s+(?:petition|case)\b|\bt\.?p\.?\s*\(")),
    ("review petition", re.compile(r"(?i)\breview\s+(?:petition|application)\b|\br\.?p\.?\s*\(")),
    ("special leave petition", re.compile(r"(?i)\bspecial\s+leave\b|\bs\.?l\.?p\.?\b")),
    ("writ petition", re.compile(r"(?i)\bwrit\s+petition\b|\bw\.?p\.?\s*\(|\barticle\s+32\b")),
    ("criminal appeal", re.compile(r"(?i)\bcriminal\s+(?:appeal|misc)\b|\bcrl\.?\s*a\.?\b")),
    ("civil appeal", re.compile(r"(?i)\bcivil\s+appeal\b|\bc\.?a\.?\s*no\b")),
)


def derive_case_type(title: str = "", cnr: str = "", extra: str = "") -> str:
    """Best-effort case type from the metadata we have.

    HONEST NOTE: this release's cause titles are plain "A versus B" strings and
    its ``cnr`` is a uniform 16-character establishment code
    (``ESCR01`` + serial + year) with no case-type component, so **fewer than
    0.3% of universe rows yield a case type from metadata**.  The real source is
    the "CIVIL APPEAL NO. 238 OF 2012" line that Rakshit extracts from the PDF
    text; see :func:`merge_from_record`.  We expose this field anyway so the
    parametric schema is complete for judgments whose text we never index.
    """
    blob = " ".join(x for x in (title or "", cnr or "", extra or "") if x)
    if not blob.strip():
        return ""
    for label, pat in _CASE_TYPE_PATTERNS:
        if pat.search(blob):
            return label
    return ""


def merge_from_record(meta_row: Dict[str, Any], record: Any) -> Dict[str, Any]:
    """Merge PDF-derived fields (M1's ``JudgmentRecord``) over metadata fields.

    The PDF is the better witness for bench, case number and case type; the
    metadata is the better witness for the citations and the decision date.
    Rule: **a non-empty PDF-derived value wins**, otherwise the metadata value
    survives.  The metadata value is never destroyed — it stays under
    ``bench_size_meta`` / ``case_type_meta`` so an ablation can compare them.

    ``record`` may be a ``contracts.JudgmentRecord`` or a plain dict.
    """
    get = (lambda k: getattr(record, k, None)) if not isinstance(record, dict) else record.get
    out = dict(meta_row)

    out.setdefault("bench_size_meta", int(meta_row.get("bench_size_meta") or 0))
    out.setdefault("case_type_meta", meta_row.get("case_type") or "")

    pdf_bench = int(get("bench_size") or 0)
    out["bench_size"] = pdf_bench if pdf_bench > 0 else out["bench_size_meta"]
    out["bench_source"] = "pdf" if pdf_bench > 0 else "meta"

    pdf_judges = list(get("judges") or [])
    meta_judges = list(meta_row.get("judges") or [])
    out["judges"] = pdf_judges if pdf_judges else meta_judges

    pdf_type = (get("case_type") or "").strip()
    out["case_type"] = pdf_type if pdf_type else (out["case_type_meta"] or "")

    pdf_no = (get("case_no") or "").strip()
    if pdf_no:
        out["case_no"] = pdf_no
        if not out["case_type"]:
            out["case_type"] = derive_case_type(extra=pdf_no)

    for k in ("neutral_cite", "scr_cite", "decision_date", "disposal"):
        if not out.get(k):
            v = (get(k) if k != "decision_date" else get("date")) or ""
            if v:
                out[k] = v
    return out


# ==========================================================================
# The parametric field schema  (fixed here, consumed by S2 in wave 2)
# ==========================================================================


@dataclass(frozen=True)
class ParametricField:
    """One filterable field of store S2.

    ``kind`` decides the physical layout the index builder must choose:

    ``numeric``
        One dense ``np.int32``/``float32`` array of length n_docs.  Range
        predicates (``>=``, ``<=``, ``>``, ``<``, ``=``) are a vectorised
        comparison + ``np.flatnonzero`` — O(N) but with a tiny constant, and the
        result is already a sorted doc-id array, which is what the planner
        wants to intersect with postings.
    ``categorical``
        A small value domain.  Stored as a dict ``value -> sorted int32 array``
        (a postings list in all but name); ``=`` is a dict lookup, ``in`` is a
        merge of a few arrays.
    ``set``
        A document has *many* values (judges, acts).  Stored as an inverted
        dict ``value -> sorted int32 array``; the only predicates are ``=``
        (membership) and ``in``.

    None of these are tokenised or stemmed, and none participate in tf-idf:
    that is exactly what separates a parametric field from a zone.
    """

    name: str                       # the name used in queries: year, bench, ...
    kind: str                       # numeric | categorical | set
    column: str                     # source column in universe/manifest parquet
    ops: Sequence[str]              # predicates the index must support
    dtype: str = "int32"
    domain: Optional[Sequence[str]] = None   # for categorical, when closed
    source: str = "metadata"        # metadata | pdf | merged | derived
    note: str = ""


#: THE schema.  ``ParametricIndex.select(field, op, value)`` in wave 2 is driven
#: entirely by this table, so adding a filter is a row here plus an extractor,
#: not new code in the planner.
PARAMETRIC_FIELDS: Dict[str, ParametricField] = {
    "year": ParametricField(
        name="year", kind="numeric", column="year",
        ops=("=", ">=", "<=", ">", "<", "in"), dtype="int32",
        source="metadata",
        note="decision year, 1950-2025; `year:>=2000` -> flatnonzero(arr >= 2000)",
    ),
    "bench": ParametricField(
        name="bench", kind="numeric", column="bench_size",
        ops=("=", ">=", "<=", ">", "<"), dtype="int8",
        source="merged",
        note=(
            "PDF CORAM count (M1) when available, else bench_size_meta. "
            "`bench:>=5` -> Constitution-bench filter. NEVER build this from "
            "bench_size_meta alone: 87% of metadata rows carry one name."
        ),
    ),
    "judge": ParametricField(
        name="judge", kind="set", column="judge_keys",
        ops=("=", "in"), dtype="object",
        source="merged",
        note=(
            "lower-cased full names AND surnames, so `judge:chandrachud` and "
            "`judge:\"d.y. chandrachud\"` both resolve by dict lookup"
        ),
    ),
    "act": ParametricField(
        name="act", kind="set", column="acts",
        ops=("=", "in"), dtype="object",
        source="pdf",
        note=(
            "canonical act codes from M3's canonicalise_statutes (IPC, CONST, "
            "CRPC, NDPS, ...). NOT available from metadata - populated from "
            "JudgmentRecord.acts at S2 build time, so it exists for the indexed "
            "manifest only, not for universe-only nodes."
        ),
    ),
    "type": ParametricField(
        name="type", kind="categorical", column="case_type",
        ops=("=", "in"), dtype="object", domain=CASE_TYPE_DOMAIN,
        source="merged",
        note=(
            "`type:\"criminal appeal\"`. Metadata yields this for <0.3% of rows "
            "(titles are bare 'A versus B'); the PDF case-number line is the "
            "real source and wins in merge_from_record()."
        ),
    ),
    "disposal": ParametricField(
        name="disposal", kind="categorical", column="disposal",
        ops=("=", "in"), dtype="object", domain=DISPOSAL_DOMAIN,
        source="metadata",
        note="`disposal:allowed` folds 'Appeal(s) allowed', 'Case Allowed', ...",
    ),
}


def parametric_schema() -> Dict[str, Dict[str, Any]]:
    """JSON-serialisable view of :data:`PARAMETRIC_FIELDS` (for S2's header)."""
    return {
        name: {
            "kind": f.kind, "column": f.column, "ops": list(f.ops),
            "dtype": f.dtype, "domain": list(f.domain) if f.domain else None,
            "source": f.source, "note": f.note,
        }
        for name, f in PARAMETRIC_FIELDS.items()
    }


# ==========================================================================
# Building the universe
# ==========================================================================

UNIVERSE_COLUMNS = [
    "node_id", "year", "path", "title", "title_key", "title_tokens",
    "petitioner", "respondent", "neutral_cite", "neutral_key",
    "scr_cite", "scr_key", "decision_date", "disposal", "disposal_raw",
    "judges", "judge_keys", "bench_size_meta", "case_type", "cnr",
    "author_judge", "available_languages",
]

MANIFEST_COLUMNS = [
    "doc_id", "node_id", "year", "path", "title", "title_key", "title_tokens",
    "neutral_cite", "neutral_key", "scr_cite", "scr_key", "decision_date",
    "disposal", "judges", "judge_keys", "bench_size_meta", "case_type",
]

_DATE_RE = re.compile(r"^\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*$")


def to_iso_date(raw: str) -> str:
    """``"23-11-2017"`` (DD-MM-YYYY) -> ``"2017-11-23"``.  "" when unparseable."""
    if not raw:
        return ""
    m = _DATE_RE.match(str(raw))
    if not m:
        return ""
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return ""
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _raw_year_files() -> List[Path]:
    return sorted(Path(config.RAW_METADATA).glob("*.parquet"))


def read_raw(verbose: bool = False) -> pd.DataFrame:
    """Concatenate all year partitions, dropping the two dead columns."""
    frames = []
    for f in _raw_year_files():
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no year partitions under {config.RAW_METADATA}; "
            "fetch the AWS Open Data metadata first"
        )
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"read {len(out):,} raw rows from {len(frames)} year partitions",
              file=sys.stderr)
    return out


def _is_english(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.contains(ENGLISH_CODE, regex=False)


def _s(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as plain python str, '' for nulls (pyarrow strings are awkward)."""
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").map(
        lambda v: "" if v is None else str(v)
    )


def build_universe(verbose: bool = True) -> pd.DataFrame:
    """All English judgments, de-duplicated on ``path``, normalised.

    De-duplication keeps the first occurrence but *unions* the language lists,
    because the only field that ever differs between duplicate rows is
    ``available_languages`` (checked: 97 of the first 400 duplicated paths).
    """
    raw = read_raw(verbose=verbose)
    eng = raw[_is_english(raw["available_languages"])].copy()

    # --- de-duplication -------------------------------------------------
    # The release repeats rows (the same judgment is emitted once per language
    # page it was scraped from).  `path` is the bucket key stem and is the true
    # document identity, so it is the dedup key.
    langs = (
        eng.assign(_l=_s(eng, "available_languages"))
        .groupby("path")["_l"]
        .apply(lambda s: ",".join(sorted({c for v in s for c in str(v).split(",") if c})))
    )
    uni = eng.drop_duplicates(subset=["path"], keep="first").copy()
    uni["available_languages"] = uni["path"].map(langs).fillna("")

    # --- stable node ids ------------------------------------------------
    uni["year"] = pd.to_numeric(uni["year"], errors="coerce").fillna(0).astype("int32")
    uni = uni.sort_values(["year", "path"], kind="mergesort").reset_index(drop=True)
    uni["node_id"] = range(len(uni))

    # --- normalisation --------------------------------------------------
    title = _s(uni, "title")
    uni["title"] = title
    uni["title_key"] = title.map(title_key)
    uni["title_tokens"] = title.map(title_tokens)
    uni["petitioner"] = _s(uni, "petitioner")
    uni["respondent"] = _s(uni, "respondent")

    case_id = _s(uni, "case_id")
    uni["neutral_cite"] = case_id.str.strip()
    uni["neutral_key"] = case_id.map(normalise_neutral)
    # nc_display is already the normalised neutral key in this release; use it
    # to fill the handful of rows whose case_id is blank or malformed.
    nc = _s(uni, "nc_display").str.replace(r"\s+", "", regex=True).str.upper()
    uni["neutral_key"] = uni["neutral_key"].where(uni["neutral_key"] != "", nc)

    citation = _s(uni, "citation")
    uni["scr_cite"] = citation.str.strip()
    uni["scr_key"] = citation.map(normalise_scr)

    uni["decision_date"] = _s(uni, "decision_date").map(to_iso_date)

    disp = _s(uni, "disposal_nature").str.strip()
    uni["disposal_raw"] = disp
    uni["disposal"] = disp.map(canonical_disposal)

    judges = _s(uni, "judge").map(split_judges)
    uni["judges"] = judges
    uni["judge_keys"] = judges.map(judge_keys)
    # LOWER BOUND ONLY — see the warning block at the top of this module.
    uni["bench_size_meta"] = judges.map(len).astype("int16")

    uni["cnr"] = _s(uni, "cnr")
    uni["author_judge"] = _s(uni, "author_judge").map(clean_judge)
    uni["case_type"] = [
        derive_case_type(t, c) for t, c in zip(uni["title"], uni["cnr"])
    ]

    out = uni[UNIVERSE_COLUMNS].copy()
    if verbose:
        dup_rate = 1 - len(out) / max(len(eng), 1)
        print(
            f"universe: {len(out):,} rows "
            f"(english {len(eng):,}, duplicate rate {dup_rate:.2%})",
            file=sys.stderr,
        )
    return out


def write_universe(df: Optional[pd.DataFrame] = None) -> Path:
    df = build_universe() if df is None else df
    path = Path(config.UNIVERSE_PARQUET)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ==========================================================================
# Building the manifest
# ==========================================================================


def sample_manifest(universe: pd.DataFrame,
                    per_year_cap: int = config.PER_YEAR_CAP,
                    seed: int = config.SAMPLE_SEED) -> pd.DataFrame:
    """Stratified sample: <= ``per_year_cap`` judgments per year.

    Stratifying by year (rather than taking a uniform random 14k) buys equal
    temporal coverage: 1950 contributes all 41 of its judgments and 2010
    contributes 200 of its 1,535, so a query about an old doctrine can actually
    retrieve the old judgments.  The cost — stated in the README as a known
    limitation — is that in busy years only ~13% of judgments are indexed, which
    caps citation-recovery recall, because a cited precedent is retrievable only
    if it happened to be sampled.

    One ``random.Random(seed)`` is advanced across the years in ascending order,
    so the draw is reproducible given (universe, cap, seed).
    """
    rng = random.Random(seed)
    picked: List[int] = []
    for year in sorted(universe["year"].unique()):
        block = universe.index[universe["year"] == year].tolist()
        # `universe` is already sorted by (year, path), so `block` is in a
        # deterministic order before sampling.
        if len(block) <= per_year_cap:
            picked.extend(block)
        else:
            picked.extend(rng.sample(block, per_year_cap))
    man = universe.loc[sorted(picked)].copy()
    man = man.sort_values(["year", "path"], kind="mergesort").reset_index(drop=True)
    man.insert(0, "doc_id", range(len(man)))
    return man[MANIFEST_COLUMNS]


def build_manifest(universe: Optional[pd.DataFrame] = None,
                   verbose: bool = True) -> pd.DataFrame:
    if universe is None:
        p = Path(config.UNIVERSE_PARQUET)
        universe = pd.read_parquet(p) if p.exists() else build_universe(verbose=verbose)
    man = sample_manifest(universe)
    if verbose:
        print(f"manifest: {len(man):,} documents over {man['year'].nunique()} years",
              file=sys.stderr)
    return man


def write_manifest(df: Optional[pd.DataFrame] = None) -> Path:
    df = build_manifest() if df is None else df
    path = Path(config.MANIFEST_PARQUET)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ==========================================================================
# Loaders (the wave-2 parametric index builds straight off these)
# ==========================================================================


def load_universe(path: Optional[Path] = None) -> pd.DataFrame:
    """All graph nodes, indexed by ``node_id``-order.  Raises if not built."""
    p = Path(path or config.UNIVERSE_PARQUET)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m nyaya.metadata universe`")
    return pd.read_parquet(p)


def load_manifest(path: Optional[Path] = None) -> pd.DataFrame:
    """The indexed subset, in ``doc_id`` order (row i has doc_id i)."""
    p = Path(path or config.MANIFEST_PARQUET)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m nyaya.metadata manifest`")
    return pd.read_parquet(p)


def node_id_by_key(universe: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """``{neutral_key | scr_key: node_id}`` - the wave-2 exact citation resolver.

    Both key spaces live in one dict because they cannot collide: a neutral key
    is ``\\d{4}INSC\\d+`` and an SCR key starts with ``SCR:``.

    SCR keys are inserted first and are authoritative: they are 100% populated
    and perfectly unique over the universe.  Neutral keys are 99.8% populated
    but 116 of them are shared by two judgments (the same INSC number issued for
    a batch disposed together), so they are inserted with ``setdefault`` and the
    lowest ``node_id`` wins.  Wave-2 resolution therefore tries ``scr`` before
    ``neutral`` before name matching, and records which in
    ``CitationEdge.resolved_by``.
    """
    uni = load_universe() if universe is None else universe
    out: Dict[str, int] = {}
    for key_col in ("scr_key", "neutral_key"):
        for k, n in zip(uni[key_col], uni["node_id"]):
            if k:
                out.setdefault(str(k), int(n))
    return out


# ==========================================================================
# Stats
# ==========================================================================


def stats(verbose_build: bool = False) -> Dict[str, Any]:
    """Honest numbers for the report.  Rebuilds from raw so nothing is stale."""
    raw = read_raw(verbose=verbose_build)
    n_raw = len(raw)
    eng_mask = _is_english(raw["available_languages"])
    n_eng = int(eng_mask.sum())
    n_eng_paths = int(raw.loc[eng_mask, "path"].nunique())

    uni = build_universe(verbose=False)
    man = sample_manifest(uni)

    out: Dict[str, Any] = {
        "raw_rows": n_raw,
        "year_partitions": len(_raw_year_files()),
        "english_rows": n_eng,
        "english_share": n_eng / n_raw if n_raw else 0.0,
        "duplicate_rows": n_eng - n_eng_paths,
        "duplicate_rate": (n_eng - n_eng_paths) / n_eng if n_eng else 0.0,
        "universe_rows": len(uni),
        "manifest_rows": len(man),
        "sampling_rate": len(man) / len(uni) if len(uni) else 0.0,
        "per_decade": {},
        "judge_cardinality": {},
        "neutral_cite_coverage": float((uni["neutral_cite"].str.len() > 0).mean()),
        "neutral_key_coverage": float((uni["neutral_key"].str.len() > 0).mean()),
        "scr_cite_coverage": float((uni["scr_cite"].str.len() > 0).mean()),
        "scr_key_coverage": float((uni["scr_key"].str.len() > 0).mean()),
        "neutral_key_collisions": int(
            uni.loc[uni["neutral_key"] != ""].duplicated(subset=["neutral_key"]).sum()
        ),
        "scr_key_collisions": int(
            uni.loc[uni["scr_key"] != ""].duplicated(subset=["scr_key"]).sum()
        ),
        "date_parse_failures": int((uni["decision_date"].str.len() == 0).sum()),
        "disposal_coverage": float((uni["disposal"].str.len() > 0).mean()),
        "case_type_coverage_meta": float((uni["case_type"].str.len() > 0).mean()),
    }

    dec_u = (uni["year"] // 10 * 10)
    dec_m = (man["year"] // 10 * 10)
    cu, cm = dec_u.value_counts(), dec_m.value_counts()
    for d in sorted(cu.index):
        out["per_decade"][int(d)] = {
            "universe": int(cu.get(d, 0)), "manifest": int(cm.get(d, 0))
        }

    jc = uni["bench_size_meta"].value_counts().sort_index()
    for k, v in jc.items():
        out["judge_cardinality"][int(k)] = int(v)
    out["single_judge_meta_share"] = float((uni["bench_size_meta"] == 1).mean())
    return out


def print_stats(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 68)
    p("NyayaTrace metadata - universe & manifest")
    p("=" * 68)
    p(f"raw rows read            : {s['raw_rows']:,}  ({s['year_partitions']} year partitions)")
    p(f"English rows             : {s['english_rows']:,}  ({s['english_share']:.2%} of raw)")
    p(f"duplicate rows (on path) : {s['duplicate_rows']:,}  ({s['duplicate_rate']:.2%} of English)")
    p(f"UNIVERSE (graph nodes)   : {s['universe_rows']:,}")
    p(f"MANIFEST (indexed docs)  : {s['manifest_rows']:,}  "
      f"({s['sampling_rate']:.1%} of the universe, cap {config.PER_YEAR_CAP}/yr, seed {config.SAMPLE_SEED})")
    p("")
    p("per decade        universe   manifest   sampled")
    for d, v in s["per_decade"].items():
        share = v["manifest"] / v["universe"] if v["universe"] else 0.0
        p(f"  {d}s           {v['universe']:>8,}   {v['manifest']:>8,}   {share:6.1%}")
    p("")
    p("judge-field cardinality (names found in the `judge` column)")
    tot = max(s["universe_rows"], 1)
    for k, v in s["judge_cardinality"].items():
        p(f"  {k} name(s): {v:>7,}  ({v / tot:6.2%})")
    p(f"  -> {s['single_judge_meta_share']:.1%} of judgments list a SINGLE name.")
    p("     The Supreme Court never sits alone (minimum two judges), so this")
    p("     column is the AUTHORING judge, not the bench.  bench_size_meta is a")
    p("     LOWER BOUND; the real bench comes from the PDF CORAM line (M1) and")
    p("     merge_from_record() prefers it.  Never feed bench_size_meta to g(d).")
    p("")
    p("citation coverage (this is the big win for citation resolution)")
    p(f"  neutral cite (`2017 INSC 1136`) : {s['neutral_cite_coverage']:.2%}"
      f"   normalised key: {s['neutral_key_coverage']:.2%}")
    p(f"  SCR cite (`[2017] 11 S.C.R. 1029`): {s['scr_cite_coverage']:.2%}"
      f"   normalised key: {s['scr_key_coverage']:.2%}")
    p(f"  key collisions                  : scr {s['scr_key_collisions']},"
      f" neutral {s['neutral_key_collisions']}")
    p("  -> every node is addressable by an EXACT key, so a citation found in")
    p("     judgment text resolves by dictionary lookup with no false")
    p("     positives.  SCR keys are unique, so resolution order is")
    p("     scr -> neutral -> fuzzy case-name Jaccard (threshold "
      f"{config.CITATION_NAME_THRESHOLD}, the fallback only).")
    p("")
    p(f"date-parse failures      : {s['date_parse_failures']:,}  "
      f"(decision_date DD-MM-YYYY -> ISO)")
    p(f"disposal coverage        : {s['disposal_coverage']:.2%}")
    p(f"case_type from metadata  : {s['case_type_coverage_meta']:.2%} "
      "(cause titles are bare 'A versus B'; the PDF case-number line is the real source)")
    p("")
    p("parametric fields fixed for store S2:")
    for name, f in PARAMETRIC_FIELDS.items():
        p(f"  {name:<9} {f.kind:<12} ops={','.join(f.ops):<22} src={f.source}")
    p("=" * 68)


# ==========================================================================
# CLI
# ==========================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(
        prog="python -m nyaya.metadata",
        description="Build the judgment universe (graph nodes) and the "
                    "stratified manifest (indexed subset).",
    )
    ap.add_argument("command", choices=("universe", "manifest", "stats", "all"))
    ap.add_argument("--force", action="store_true",
                    help="rebuild the universe even if the parquet exists")
    args = ap.parse_args(argv)

    if args.command in ("universe", "all"):
        with timed("universe"):
            uni = build_universe()
            path = write_universe(uni)
        print(f"wrote {len(uni):,} universe rows -> {path}")

    if args.command in ("manifest", "all"):
        with timed("manifest"):
            uni = None
            if not args.force and Path(config.UNIVERSE_PARQUET).exists():
                uni = load_universe()
            man = build_manifest(uni)
            path = write_manifest(man)
        print(f"wrote {len(man):,} manifest rows -> {path}")

    if args.command == "stats":
        with timed("stats"):
            s = stats()
        print_stats(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
