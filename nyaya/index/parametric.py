"""Store S2 — the parametric index.

Member 2 (Suvrat).  This is the other half of the index, and the half that is
*not* a bag of words.  The distinction is the one in Manning, IIR 6.1:

* The **positional index** (S3, M1) answers "which documents contain this
  *term*".  Its input is free text that has been tokenised, folded and stemmed;
  its postings carry tf and positions because the answer is a matter of degree
  and gets scored.
* The **parametric index** (this file) answers "which documents have this
  *field value*".  Its input is a structured value from a known domain — an
  integer year, a bench size, a member of a set of judge names.  There is no tf,
  no idf, no stemming and no score: a predicate either admits a document or it
  does not.

What makes the two composable is that **both return a sorted array of document
ids**.  `select()` hands back exactly the `doc_ids` array that a `PostingsList`
exposes, so `FilterNode` and `TermNode` are merged by the same intersection code
in the planner; the planner never needs to know which store a candidate set came
from.  :class:`ParametricPostings` makes that equivalence explicit by satisfying
the read side of :class:`~nyaya.contracts.PostingsList`.

Physical layout, driven entirely by :data:`~nyaya.metadata.PARAMETRIC_FIELDS`:

``numeric`` (``year``, ``bench``)
    A dense value array of length n_docs *plus* the argsort permutation.  A
    range predicate is then two ``np.searchsorted`` calls — O(log N) — and a
    slice of the permutation, instead of an O(N) scan of every document.  The
    slice is sorted before it is returned, because the planner's merges need
    ascending doc ids.

``categorical`` (``type``, ``disposal``) and ``set`` (``judge``, ``act``)
    An inverted dictionary ``value -> sorted int32 doc-id array``.  This *is* a
    postings list; the only difference from S3 is that the vocabulary is a closed
    domain and the entries carry no tf.  ``=`` is one dict lookup, ``in`` is a
    union of a few arrays.

On disk (contract: ``index/parametric.json``, ``index/parametric_*.npy``)::

    index/parametric.json          schema, dictionaries, header counts
    index/parametric_year.npy      int32[n_docs]
    index/parametric_year_ord.npy  int32[n_docs]   argsort permutation
    index/parametric_bench.npy     int8[n_docs]
    index/parametric_bench_ord.npy int32[n_docs]

CLI::

    python -m nyaya.index.parametric build
    python -m nyaya.index.parametric stats
    python -m nyaya.index.parametric select year '>=' 2000
    python -m nyaya.index.parametric select judge = chandrachud
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nyaya import config
from nyaya.metadata import (
    PARAMETRIC_FIELDS,
    ParametricField,
    _as_str_list,
    load_manifest,
    merge_from_record,
)
from nyaya.utils import ensure_utf8_stdout, timed

EMPTY = np.empty(0, dtype=np.int32)

NUMERIC_OPS = ("=", "!=", ">=", "<=", ">", "<", "in")
MEMBERSHIP_OPS = ("=", "!=", "in")


# ==========================================================================
# The postings adapter — why a filter and a term are interchangeable
# ==========================================================================


class ParametricPostings:
    """A parametric selection dressed as a postings list.

    It satisfies the read side of :class:`~nyaya.contracts.PostingsList`
    (``doc_ids``, ``df``, ``skip_stride``) so the planner can intersect a
    ``FILTER(year>=2000)`` with a ``TERM(privacy)`` without a special case.  The
    methods that only make sense for text — ``positions_of``, ``zone_tf_of`` —
    raise, because a parametric field has no positions and no term frequency,
    and silently returning zeros would let a scorer treat a filter as evidence.
    """

    __slots__ = ("doc_ids", "field", "op", "value")

    def __init__(self, doc_ids: np.ndarray, field: str = "", op: str = "", value: Any = None):
        self.doc_ids = np.asarray(doc_ids, dtype=np.int32)
        self.field, self.op, self.value = field, op, value

    @property
    def df(self) -> int:
        return int(self.doc_ids.size)

    def skip_stride(self) -> int:
        return max(1, int(np.sqrt(max(self.df, 1))))

    def positions_of(self, doc_id: int):
        raise TypeError("a parametric field has no positions — it is not text")

    def zone_tf_of(self, doc_id: int):
        raise TypeError("a parametric field has no term frequency — it is a filter")

    def describe(self) -> str:
        return f"PARAM({self.field}{self.op}{self.value!r}) -> {self.df} docs"

    def __len__(self) -> int:
        return self.df

    def __repr__(self) -> str:
        return f"<ParametricPostings {self.describe()}>"


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sorted-array intersection — the same merge the planner runs on postings.

    Kept here so a reader can see that nothing about a parametric result is
    special: it is a sorted int32 doc-id array like any other.
    """
    return np.intersect1d(np.asarray(a), np.asarray(b), assume_unique=True).astype(np.int32)


def union(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return EMPTY
    out = np.concatenate([np.asarray(a, dtype=np.int32) for a in arrays]) if len(arrays) > 1 \
        else np.asarray(arrays[0], dtype=np.int32)
    return np.unique(out).astype(np.int32)


# ==========================================================================
# The index
# ==========================================================================


class ParametricIndex:
    """Store S2.  ``select(field, op, value)`` -> sorted ``int32`` doc ids."""

    def __init__(self,
                 n_docs: int,
                 numeric: Dict[str, np.ndarray],
                 numeric_order: Dict[str, np.ndarray],
                 dicts: Dict[str, Dict[str, np.ndarray]],
                 schema: Optional[Dict[str, Any]] = None,
                 header: Optional[Dict[str, Any]] = None):
        self.n_docs = int(n_docs)
        self.numeric = numeric            # field -> dense values, length n_docs
        self.numeric_order = numeric_order  # field -> argsort permutation
        self.dicts = dicts                # field -> {value: sorted int32 doc ids}
        self.schema = schema or {}
        self.header = header or {}

    # -- introspection ----------------------------------------------------
    @property
    def fields(self) -> List[str]:
        return sorted(set(self.numeric) | set(self.dicts))

    def kind(self, field: str) -> str:
        if field in self.numeric:
            return "numeric"
        f = PARAMETRIC_FIELDS.get(field)
        return f.kind if f else "categorical"

    def values(self, field: str) -> List[str]:
        """The observed domain of a categorical/set field, commonest first."""
        d = self.dicts.get(field)
        if not d:
            return []
        return sorted(d, key=lambda v: (-d[v].size, v))

    def value_counts(self, field: str, top: int = 20) -> List[Tuple[str, int]]:
        d = self.dicts.get(field, {})
        return sorted(((v, int(a.size)) for v, a in d.items()),
                      key=lambda t: (-t[1], t[0]))[:top]

    def column(self, field: str) -> np.ndarray:
        """The dense value array of a numeric field (g(d) reads ``bench``)."""
        if field not in self.numeric:
            raise KeyError(f"{field} is not a numeric parametric field")
        return self.numeric[field]

    # -- the one method the contract fixes --------------------------------
    def select(self, field: str, op: str, value: Any) -> np.ndarray:
        """Sorted ``int32`` doc ids satisfying ``field op value``.

        ``op`` is one of ``= != >= <= > < in``.  Range operators are only legal
        on a numeric field; a set/categorical field supports membership only,
        which is the whole reason the schema records ``kind``.
        """
        field = _alias(field)
        if field in self.numeric:
            return self._select_numeric(field, op, value)
        if field in self.dicts:
            return self._select_dict(field, op, value)
        raise KeyError(
            f"unknown parametric field {field!r}; have {', '.join(self.fields)}"
        )

    def postings(self, field: str, op: str, value: Any) -> ParametricPostings:
        """``select()`` wrapped so it quacks like a :class:`PostingsList`."""
        return ParametricPostings(self.select(field, op, value), field, op, value)

    # -- numeric ----------------------------------------------------------
    def _select_numeric(self, field: str, op: str, value: Any) -> np.ndarray:
        vals = self.numeric[field]
        order = self.numeric_order[field]
        sorted_vals = vals[order]

        if op == "in":
            return union([self._select_numeric(field, "=", v) for v in _as_list(value)])
        if op not in NUMERIC_OPS:
            raise ValueError(f"operator {op!r} is not defined on numeric field {field!r}")

        v = _to_number(value)
        # Binary search on the sorted view: O(log n) instead of a full scan.
        lo = int(np.searchsorted(sorted_vals, v, side="left"))
        hi = int(np.searchsorted(sorted_vals, v, side="right"))
        if op == "=":
            sl = order[lo:hi]
        elif op == ">=":
            sl = order[lo:]
        elif op == ">":
            sl = order[hi:]
        elif op == "<=":
            sl = order[:hi]
        elif op == "<":
            sl = order[:lo]
        else:                                     # "!="
            sl = np.concatenate([order[:lo], order[hi:]])
        out = np.sort(sl).astype(np.int32)        # the planner needs ascending ids
        return out

    # -- dictionary (categorical / set) -----------------------------------
    def _select_dict(self, field: str, op: str, value: Any) -> np.ndarray:
        d = self.dicts[field]
        if op == "in":
            return union([d.get(_norm_value(v), EMPTY) for v in _as_list(value)])
        if op == "=":
            return d.get(_norm_value(value), EMPTY)
        if op == "!=":
            hit = d.get(_norm_value(value), EMPTY)
            return np.setdiff1d(np.arange(self.n_docs, dtype=np.int32), hit,
                                assume_unique=True).astype(np.int32)
        raise ValueError(
            f"operator {op!r} is not defined on {self.kind(field)} field {field!r}; "
            f"only {', '.join(MEMBERSHIP_OPS)} are"
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = None) -> Path:
        base = Path(path or config.INDEX)
        base.mkdir(parents=True, exist_ok=True)
        for f, arr in self.numeric.items():
            np.save(base / f"parametric_{f}.npy", arr)
            np.save(base / f"parametric_{f}_ord.npy", self.numeric_order[f])
        blob = {
            "n_docs": self.n_docs,
            "schema": self.schema,
            "header": self.header,
            "numeric": {f: str(a.dtype) for f, a in self.numeric.items()},
            "dicts": {
                f: {v: [int(x) for x in arr] for v, arr in sorted(d.items())}
                for f, d in self.dicts.items()
            },
        }
        out = base / "parametric.json"
        out.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path = None) -> "ParametricIndex":
        base = Path(path or config.INDEX)
        meta = base / "parametric.json"
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta} missing — run `python -m nyaya.index.parametric build`"
            )
        blob = json.loads(meta.read_text(encoding="utf-8"))
        numeric, order = {}, {}
        for f in blob.get("numeric", {}):
            numeric[f] = np.load(base / f"parametric_{f}.npy")
            order[f] = np.load(base / f"parametric_{f}_ord.npy")
        dicts = {
            f: {v: np.asarray(ids, dtype=np.int32) for v, ids in d.items()}
            for f, d in blob.get("dicts", {}).items()
        }
        return cls(blob["n_docs"], numeric, order, dicts,
                   blob.get("schema"), blob.get("header"))

    def __repr__(self) -> str:
        return (f"<ParametricIndex n_docs={self.n_docs} "
                f"numeric={sorted(self.numeric)} dicts={sorted(self.dicts)}>")


# --------------------------------------------------------------------------
# Query-value plumbing
# --------------------------------------------------------------------------

#: The parser may hand us either the query word or the schema name.
_ALIASES = {
    "bench_size": "bench", "benchsize": "bench", "judges": "judge",
    "acts": "act", "case_type": "type", "casetype": "type",
}


def _alias(field: str) -> str:
    f = str(field).strip().lower()
    return _ALIASES.get(f, f)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _to_number(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().strip("\"'")
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a number; a range filter needs one") from exc


def _norm_value(value: Any) -> str:
    """Categorical/set lookups are case- and quote-insensitive."""
    return str(value).strip().strip("\"'").lower()


# ==========================================================================
# Building
# ==========================================================================


def _iter_records(path: Path = None, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream ``judgments.jsonl``, keeping only the parametric fields.

    The file is ~1 GB because it carries every zone and paragraph; we need six
    scalar fields from it, so it is streamed and discarded line by line rather
    than loaded.
    """
    p = Path(path or config.JUDGMENTS_JSONL)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — M1's ingest has not produced it yet")
    keep = ("doc_id", "bench_size", "judges", "acts", "statute_tokens",
            "case_type", "case_no", "disposal", "date", "neutral_cite", "scr_cite")
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield {k: r.get(k) for k in keep}


def build(manifest: Optional[pd.DataFrame] = None,
          records: Optional[Iterable[Dict[str, Any]]] = None,
          verbose: bool = True) -> ParametricIndex:
    """Build S2 from the manifest (metadata facts) + judgments.jsonl (PDF facts).

    Every field is taken through :func:`~nyaya.metadata.merge_from_record`, so
    the PDF-derived bench, case type and judge list win wherever M1 found them
    and the metadata is the fallback.  Nothing here reads ``bench_size_meta``
    directly: that column is 87% ones and would make every judgment look like a
    one-judge bench.
    """
    man = load_manifest() if manifest is None else manifest
    n_docs = len(man)
    doc_index = {int(d): i for i, d in enumerate(man["doc_id"])}

    year = np.zeros(n_docs, dtype=np.int32)
    bench = np.zeros(n_docs, dtype=np.int8)
    judge_sets: List[List[str]] = [[] for _ in range(n_docs)]
    act_sets: List[List[str]] = [[] for _ in range(n_docs)]
    types: List[str] = [""] * n_docs
    disposals: List[str] = [""] * n_docs

    # --- metadata facts (present for every manifest row) -----------------
    rows: Dict[int, Dict[str, Any]] = {}
    for i, row in enumerate(man.to_dict("records")):
        year[i] = int(row["year"])
        rows[int(row["doc_id"])] = row
        disposals[i] = str(row.get("disposal") or "")

    # --- PDF facts, merged over the metadata -----------------------------
    seen = 0
    missing_bench = 0
    rec_iter = _iter_records() if records is None else records
    for rec in rec_iter:
        did = int(rec.get("doc_id", -1))
        i = doc_index.get(did)
        if i is None:
            continue                       # a record outside the manifest: ignore
        merged = merge_from_record(rows.get(did, {"judges": [], "bench_size_meta": 0}), rec)
        seen += 1
        b = int(merged.get("bench_size") or 0)
        if b <= 0:
            missing_bench += 1
        bench[i] = min(b, 127)
        judge_sets[i] = _judge_lookup_keys(_as_str_list(merged.get("judges")))
        act_sets[i] = sorted({a.strip().lower() for a in _as_str_list(rec.get("acts")) if a.strip()})
        types[i] = str(merged.get("case_type") or "").strip().lower()
        if merged.get("disposal"):
            disposals[i] = str(merged["disposal"]).strip().lower()

    # Documents with no record (dropped by extraction) keep the metadata-only
    # fallback, so a filter never silently excludes them on `year`.
    for did, row in rows.items():
        i = doc_index[did]
        if bench[i] == 0:
            bench[i] = min(int(row.get("bench_size_meta") or 0), 127)
        if not judge_sets[i]:
            judge_sets[i] = _judge_lookup_keys(_as_str_list(row.get("judges")))
        if not types[i]:
            types[i] = str(row.get("case_type") or "").strip().lower()

    numeric = {"year": year, "bench": bench}
    numeric_order = {f: np.argsort(a, kind="stable").astype(np.int32)
                     for f, a in numeric.items()}
    dicts = {
        "judge": _invert_sets(judge_sets),
        "act": _invert_sets(act_sets),
        "type": _invert_single(types),
        "disposal": _invert_single(disposals),
    }

    header = {
        "n_docs": n_docs,
        "records_seen": seen,
        "records_without_pdf_bench": missing_bench,
        "docs_without_any_record": n_docs - seen,
        "distinct_judges": len(dicts["judge"]),
        "distinct_acts": len(dicts["act"]),
        "bench_from_pdf": int(seen - missing_bench),
    }
    idx = ParametricIndex(n_docs, numeric, numeric_order, dicts,
                          schema=_schema_blob(), header=header)
    if verbose:
        print(f"S2: {n_docs:,} docs | judges {len(dicts['judge']):,} | "
              f"acts {len(dicts['act']):,} | types {len(dicts['type'])} | "
              f"disposals {len(dicts['disposal'])}", file=sys.stderr)
        if not dicts["act"]:
            print("S2 WARNING: `act` is empty — M1 has not written record.acts "
                  "yet.  Rebuild this store after his statutes pass lands; the "
                  "other five fields are complete.", file=sys.stderr)
    return idx


def _judge_lookup_keys(judges: Sequence[str]) -> List[str]:
    """Full name AND surname, lower-cased, so `judge:chandrachud` resolves.

    Deliberately duplicated from metadata.judge_keys' behaviour rather than
    imported-and-mutated, because the judge list reaching here may be M1's
    PDF-parsed one, which has not been through the metadata normaliser.
    """
    from nyaya.metadata import judge_keys as _keys
    from nyaya.metadata import clean_judge

    cleaned = [clean_judge(j) for j in judges if str(j).strip()]
    return sorted(set(_keys([c for c in cleaned if c])))


def _invert_sets(per_doc: Sequence[Sequence[str]]) -> Dict[str, np.ndarray]:
    """``[doc][values]`` -> ``{value: sorted int32 doc ids}``.  A postings list."""
    acc: Dict[str, List[int]] = {}
    for doc_id, vals in enumerate(per_doc):
        for v in vals:
            if v:
                acc.setdefault(v, []).append(doc_id)
    return {v: np.asarray(ids, dtype=np.int32) for v, ids in acc.items()}


def _invert_single(per_doc: Sequence[str]) -> Dict[str, np.ndarray]:
    return _invert_sets([[v] if v else [] for v in per_doc])


def _schema_blob() -> Dict[str, Any]:
    from nyaya.metadata import parametric_schema
    return parametric_schema()


# ==========================================================================
# Stats / CLI
# ==========================================================================


def stats(idx: ParametricIndex) -> Dict[str, Any]:
    year, bench = idx.numeric["year"], idx.numeric["bench"]
    out: Dict[str, Any] = dict(idx.header)
    out["year_range"] = [int(year.min()), int(year.max())]
    out["bench_distribution"] = {
        int(k): int(v) for k, v in zip(*np.unique(bench, return_counts=True))
    }
    out["bench_ge_5"] = int(idx.select("bench", ">=", 5).size)
    out["year_ge_2000"] = int(idx.select("year", ">=", 2000).size)
    out["top_judges"] = idx.value_counts("judge", 10)
    out["top_acts"] = idx.value_counts("act", 10)
    out["types"] = idx.value_counts("type", 12)
    out["disposals"] = idx.value_counts("disposal", 12)
    out["postings_entries"] = {f: int(sum(a.size for a in d.values()))
                               for f, d in idx.dicts.items()}
    return out


def print_stats(idx: ParametricIndex) -> None:
    s = stats(idx)
    p = print
    p("=" * 68)
    p("S2 parametric index")
    p("=" * 68)
    p(f"documents                 : {s['n_docs']:,}")
    p(f"judgment records merged in: {s['records_seen']:,} "
      f"({s['docs_without_any_record']:,} manifest docs had no extracted record)")
    p(f"bench size from the PDF   : {s['bench_from_pdf']:,} "
      f"({s['records_without_pdf_bench']:,} fell back to metadata)")
    p(f"year range                : {s['year_range'][0]}-{s['year_range'][1]}")
    p("")
    p("numeric fields (dense array + argsort permutation, binary-searched)")
    p(f"  year  : year:>=2000 -> {s['year_ge_2000']:,} docs")
    p(f"  bench : bench:>=5   -> {s['bench_ge_5']:,} docs (Constitution benches)")
    p(f"  bench distribution: {s['bench_distribution']}")
    p("")
    p("dictionary fields (value -> sorted doc ids; this IS a postings list)")
    for f in ("judge", "act", "type", "disposal"):
        d = idx.dicts.get(f, {})
        p(f"  {f:<9} {len(d):>6,} distinct values, "
          f"{s['postings_entries'].get(f, 0):>8,} postings entries")
    p("")
    p(f"top judges   : {s['top_judges']}")
    p(f"case types   : {s['types']}")
    p(f"disposals    : {s['disposals']}")
    p(f"top acts     : {s['top_acts'] or '(empty — M1 has not written record.acts yet)'}")
    p("=" * 68)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.index.parametric",
                                 description="Build/inspect store S2.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("stats")
    sel = sub.add_parser("select")
    sel.add_argument("field")
    sel.add_argument("op", choices=list(NUMERIC_OPS))
    sel.add_argument("value")
    sel.add_argument("--show", type=int, default=10)
    args = ap.parse_args(argv)

    if args.command == "build":
        with timed("S2 build"):
            idx = build()
            out = idx.save()
        print(f"wrote S2 -> {out}")
        print_stats(idx)
    elif args.command == "stats":
        print_stats(ParametricIndex.load())
    else:
        idx = ParametricIndex.load()
        post = idx.postings(args.field, args.op, args.value)
        print(post.describe())
        print(f"first {args.show}: {post.doc_ids[:args.show].tolist()}")
        print(f"skip stride for intersection: {post.skip_stride()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
