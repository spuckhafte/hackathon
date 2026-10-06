"""S1 — the paragraph store: the unit of *evidence* (guide Section 4.2).

Why a separate store
--------------------
Ranking returns documents; a lawyer needs the passage.  Three features read
paragraphs and none of them wants the whole corpus in memory:

* snippets, drawn from the paragraphs a query term actually landed in (M1);
* the ``/p`` proximity operator, whose evidence is displayed per paragraph (M1);
* the Precedent Brief, which runs BM25 over the paragraphs *inside* the top-5
  documents and picks the best two from each (Section 9).

So the store is built for **seek-and-read**: ``paragraphs.jsonl`` holds one
compact JSON object per paragraph, and a companion offset array gives the byte
position of every line.  ``get(doc_id, para_id)`` is one array lookup, one
``seek`` and one ``readline`` — no scan, no 960 MB load.  The whole corpus is
437,654 paragraphs over 13,878 judgments; the offsets are 3.5 MB of int64 and
are the only thing held in RAM.

Honest paragraph numbering
--------------------------
This is the part that matters for citation integrity.  Only **28 %** of these
judgments print their own paragraph numbers; for the other 72 % M1's zoner
falls back to ~110-word sentence windows.  A window is *not* a paragraph of the
judgment, and a UI that renders it as "¶ 14" is fabricating a citation that a
reader cannot check against the official text.

So every paragraph carries its ``label`` (the court's printed number, or ``""``)
all the way through, and `ref()` returns a `ParaRef` whose ``kind`` says which
of three situations you are in:

==============  ====================================  ==========================
``kind``        meaning                               ``cite()`` renders
==============  ====================================  ==========================
``printed``     the court numbered this paragraph     ``¶ 14``
``printed-split``  numbered judgment, but this piece   ``¶ 14 (cont.)``
                is the tail of a paragraph we split
``window``      invented ~110-word window             ``window 14 of 32``
==============  ====================================  ==========================

Note that for ``printed`` the citation uses the court's **label**, not our
``para_id``: they diverge wherever a long printed paragraph was split, and the
label is the one a reader can look up.

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import Para

PARAS_FILE = "paragraphs.jsonl"
IDX_FILE = "paragraphs.idx.json"
OFF_FILE = "paragraphs.off.npz"          # offsets; too big to be readable JSON

#: Written into the header so a reader can tell a stale store from a fresh one.
FORMAT_VERSION = 1

# Compact keys: 437k lines, so "doc_id"/"para_id"/"zone"/"label"/"text" would
# cost ~12 MB of pure punctuation.
_K_DOC, _K_PARA, _K_ZONE, _K_LABEL, _K_TEXT = "d", "i", "z", "l", "t"

PRINTED = "printed"
PRINTED_SPLIT = "printed-split"
WINDOW = "window"


# ==========================================================================
# Citation reference
# ==========================================================================


@dataclass(frozen=True)
class ParaRef:
    """How to cite one paragraph without overclaiming.

    ``kind`` is the field the UI should branch on; see the module docstring.
    """

    doc_id: int
    para_id: int
    zone: str
    label: str
    kind: str
    n_paras: int

    @property
    def is_printed(self) -> bool:
        """True when the citation corresponds to the court's own numbering."""
        return self.kind in (PRINTED, PRINTED_SPLIT)

    def cite(self) -> str:
        """Long form, for the result card and the Precedent Brief."""
        if self.kind == PRINTED:
            return f"¶ {self.label}"
        if self.kind == PRINTED_SPLIT:
            return f"¶ {self.label} (cont.)" if self.label else \
                   f"para {self.para_id} (unnumbered)"
        return f"window {self.para_id} of {self.n_paras}"

    def short(self) -> str:
        """Compact form, for a snippet gutter or a dense table."""
        if self.kind == PRINTED:
            return f"¶{self.label}"
        if self.kind == PRINTED_SPLIT:
            return f"¶{self.label}+" if self.label else f"p{self.para_id}"
        return f"w{self.para_id}"

    def tooltip(self) -> str:
        """One sentence explaining the citation, shown on hover."""
        if self.is_printed:
            return ("the paragraph number printed in the judgment"
                    + (" (this is the continuation of a long paragraph we split "
                       "for retrieval)" if self.kind == PRINTED_SPLIT else ""))
        return ("this judgment prints no paragraph numbers, so the text was cut "
                f"into {self.n_paras} windows of about 110 words; this is window "
                f"{self.para_id}")


# ==========================================================================
# The store
# ==========================================================================


class ParagraphStore:
    """Random-access reader over ``index/paragraphs.jsonl``.

    Open once and keep it; the file handle is held open and seeked.  Cheap to
    construct (it loads two numpy arrays) and safe to pass into a
    ``SearchContext``.
    """

    def __init__(self, path: Path, header: Dict, row_off: np.ndarray,
                 doc_row: np.ndarray, doc_nparas: np.ndarray,
                 doc_printed: np.ndarray):
        self.path = Path(path)
        self.header = header
        self._row_off = row_off          # byte offset of every paragraph line
        self._doc_row = doc_row          # doc_id -> first row, -1 if absent
        self._doc_nparas = doc_nparas    # doc_id -> number of paragraphs
        self._doc_printed = doc_printed  # doc_id -> does it print numbers?
        self._fh = (self.path / PARAS_FILE).open("rb")
        self._label_cache: Dict[int, Dict[str, int]] = {}

    # ---- construction ---------------------------------------------------
    @classmethod
    def load(cls, path: Path | str = config.INDEX) -> "ParagraphStore":
        path = Path(path)
        header = json.loads((path / IDX_FILE).read_text(encoding="utf-8"))
        with np.load(path / OFF_FILE) as z:
            return cls(path, header, z["row_off"], z["doc_row"],
                       z["doc_nparas"], z["doc_printed"])

    @classmethod
    def exists(cls, path: Path | str = config.INDEX) -> bool:
        path = Path(path)
        return (path / PARAS_FILE).exists() and (path / OFF_FILE).exists()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:                                   # pragma: no cover
            pass

    def __enter__(self) -> "ParagraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- sizes ----------------------------------------------------------
    @property
    def doc_count(self) -> int:
        return int((self._doc_row >= 0).sum())

    @property
    def para_count(self) -> int:
        return int(self._row_off.size - 1)

    def n_paras(self, doc_id: int) -> int:
        if not 0 <= doc_id < self._doc_nparas.size:
            return 0
        return int(self._doc_nparas[doc_id])

    def has(self, doc_id: int) -> bool:
        return 0 <= doc_id < self._doc_row.size and self._doc_row[doc_id] >= 0

    # ---- reading --------------------------------------------------------
    def _read_row(self, row: int) -> Dict:
        self._fh.seek(int(self._row_off[row]))
        line = self._fh.readline()
        return json.loads(line.decode("utf-8"))

    def _row_of(self, doc_id: int, para_id: int) -> int:
        """Flat row index, or -1.  ``para_id`` is 1-based, as in `Para`."""
        if not self.has(doc_id):
            return -1
        if not 1 <= para_id <= self._doc_nparas[doc_id]:
            return -1
        return int(self._doc_row[doc_id]) + para_id - 1

    def get(self, doc_id: int, para_id: int) -> Optional[Para]:
        """One paragraph, by one seek.  ``None`` if it does not exist."""
        row = self._row_of(doc_id, para_id)
        if row < 0:
            return None
        return _to_para(self._read_row(row))

    def text(self, doc_id: int, para_id: int) -> str:
        p = self.get(doc_id, para_id)
        return p.text if p else ""

    def paras_of(self, doc_id: int) -> List[Para]:
        """Every paragraph of one judgment, in order.

        Reads the document's lines in one contiguous sweep rather than one seek
        per paragraph, because the Brief wants whole documents.
        """
        if not self.has(doc_id):
            return []
        start = int(self._doc_row[doc_id])
        n = int(self._doc_nparas[doc_id])
        self._fh.seek(int(self._row_off[start]))
        out = []
        for _ in range(n):
            out.append(_to_para(json.loads(self._fh.readline().decode("utf-8"))))
        return out

    def paras_in_zone(self, doc_id: int, zone: str) -> List[Para]:
        return [p for p in self.paras_of(doc_id) if p.zone == zone]

    def iter_paras(self, doc_ids: Optional[Iterable[int]] = None
                   ) -> Iterator[Tuple[int, Para]]:
        """Stream ``(doc_id, Para)`` — the whole store, or just some documents.

        Used by the df/length statistics and by the Brief's BM25 pass.  With
        ``doc_ids=None`` this is a single sequential read of the file, which is
        an order of magnitude faster than seeking per document.
        """
        if doc_ids is not None:
            for d in doc_ids:
                for p in self.paras_of(d):
                    yield d, p
            return
        with (self.path / PARAS_FILE).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield int(row[_K_DOC]), _to_para(row)

    # ---- citation -------------------------------------------------------
    def prints_own_numbers(self, doc_id: int) -> bool:
        """Did the court print paragraph numbers in this judgment?"""
        return bool(self.has(doc_id) and self._doc_printed[doc_id])

    def numbering(self, doc_id: int) -> str:
        """``'printed'`` or ``'window'`` — the document-level strategy."""
        return PRINTED if self.prints_own_numbers(doc_id) else WINDOW

    def ref(self, doc_id: int, para_id: int) -> Optional[ParaRef]:
        """The citation handle for one paragraph.  See `ParaRef`."""
        p = self.get(doc_id, para_id)
        if p is None:
            return None
        return self.ref_for(doc_id, p)

    def ref_for(self, doc_id: int, para: Para) -> ParaRef:
        """`ref` without a second read, when you already hold the `Para`."""
        label = (para.label or "").strip()
        if not self.prints_own_numbers(doc_id):
            kind = WINDOW
        elif label:
            kind = PRINTED
        else:
            kind = PRINTED_SPLIT
            label = self._inherited_label(doc_id, para.id)
        return ParaRef(doc_id=doc_id, para_id=para.id, zone=para.zone,
                       label=label, kind=kind, n_paras=self.n_paras(doc_id))

    def _inherited_label(self, doc_id: int, para_id: int) -> str:
        """The nearest printed label at or before ``para_id``.

        A printed paragraph that ran long was split, so the tail has no label of
        its own; citing it as "¶ 14 (cont.)" is both honest and checkable.
        """
        for pid in range(para_id - 1, 0, -1):
            p = self.get(doc_id, pid)
            if p is None:
                break
            if (p.label or "").strip():
                return p.label.strip()
        return ""

    def find_label(self, doc_id: int, label: str) -> Optional[int]:
        """``para_id`` of the paragraph the court numbered ``label``.

        Lets a user who types "¶ 14" land on the right passage even though our
        ``para_id`` numbering differs wherever we split.
        """
        want = str(label).strip().lstrip("¶").strip().lower()
        cache = self._label_cache.get(doc_id)
        if cache is None:
            cache = {}
            for p in self.paras_of(doc_id):
                lab = (p.label or "").strip().lower()
                if lab and lab not in cache:
                    cache[lab] = p.id
            self._label_cache[doc_id] = cache
        return cache.get(want)

    # ---- reporting ------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        n_printed = int(self._doc_printed.sum())
        nz = self._doc_nparas[self._doc_row >= 0]
        size = (self.path / PARAS_FILE).stat().st_size
        off = (self.path / OFF_FILE).stat().st_size
        return {
            "documents": self.doc_count,
            "paragraphs": self.para_count,
            "mean_paras_per_doc": round(float(nz.mean()), 1) if nz.size else 0.0,
            "median_paras_per_doc": int(np.median(nz)) if nz.size else 0,
            "max_paras_per_doc": int(nz.max()) if nz.size else 0,
            "docs_with_printed_numbers": n_printed,
            "printed_share": round(n_printed / max(1, self.doc_count), 3),
            "zones": self.header.get("zone_counts", {}),
            "paragraphs_jsonl": _human(size),
            "offsets_npz": _human(off),
            "resident_bytes": int(self._row_off.nbytes + self._doc_row.nbytes
                                  + self._doc_nparas.nbytes
                                  + self._doc_printed.nbytes),
            "format_version": self.header.get("format_version"),
        }

    def __repr__(self) -> str:                              # pragma: no cover
        return (f"<ParagraphStore {self.doc_count} docs, {self.para_count} paras, "
                f"{self.path}>")


def _to_para(row: Dict) -> Para:
    return Para(id=int(row[_K_PARA]), zone=row[_K_ZONE],
                text=row.get(_K_TEXT, ""), label=row.get(_K_LABEL, "") or "")


def _human(n: float) -> str:
    from nyaya.utils import human

    return human(n)


# ==========================================================================
# Build
# ==========================================================================


def build(src: Path | str = None, out: Path | str = config.INDEX,
          verbose: bool = True) -> Dict[str, object]:
    """Write ``paragraphs.jsonl`` + the offset arrays from ``judgments.jsonl``.

    One streaming pass: paragraphs are written in (doc_id, para_id) order, so a
    document's lines are contiguous on disk and `paras_of` is a sequential read.
    """
    from nyaya.utils import read_jsonl, timed

    src = Path(src or config.JUDGMENTS_JSONL)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    offsets: List[int] = [0]
    doc_rows: Dict[int, Tuple[int, int, bool]] = {}
    zone_counts: Dict[str, int] = {}
    pos = 0
    row = 0
    n_docs = 0
    label_total = 0

    with timed("paragraphs:write", quiet=not verbose):
        with (out / PARAS_FILE).open("wb") as fh:
            for rec in read_jsonl(src):
                doc_id = int(rec["doc_id"])
                paras = rec.get("paras") or []
                if not paras:
                    continue
                n_docs += 1
                first_row = row
                printed = False
                for p in paras:
                    label = str(p.get("label") or "").strip()
                    if label:
                        printed = True
                        label_total += 1
                    zone = str(p.get("zone") or "body")
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
                    blob = {
                        _K_DOC: doc_id,
                        _K_PARA: int(p["id"]),
                        _K_ZONE: zone,
                        _K_TEXT: str(p.get("text") or ""),
                    }
                    if label:
                        blob[_K_LABEL] = label
                    line = json.dumps(blob, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8") + b"\n"
                    fh.write(line)
                    pos += len(line)
                    offsets.append(pos)
                    row += 1
                doc_rows[doc_id] = (first_row, row - first_row, printed)

    max_doc = max(doc_rows) if doc_rows else -1
    doc_row = np.full(max_doc + 1, -1, dtype=np.int64)
    doc_nparas = np.zeros(max_doc + 1, dtype=np.int32)
    doc_printed = np.zeros(max_doc + 1, dtype=bool)
    for d, (r, n, printed) in doc_rows.items():
        doc_row[d] = r
        doc_nparas[d] = n
        doc_printed[d] = printed

    np.savez(out / OFF_FILE,
             row_off=np.asarray(offsets, dtype=np.int64),
             doc_row=doc_row, doc_nparas=doc_nparas, doc_printed=doc_printed)

    n_printed_docs = int(doc_printed.sum())
    header = {
        "format_version": FORMAT_VERSION,
        "source": str(src),
        "documents": n_docs,
        "paragraphs": row,
        "doc_id_slots": max_doc + 1,
        "zone_counts": zone_counts,
        "paragraphs_with_printed_label": label_total,
        "docs_with_printed_numbers": n_printed_docs,
        "printed_share": round(n_printed_docs / max(1, n_docs), 4),
        "note": (
            "Only the documents counted in docs_with_printed_numbers carry the "
            "court's own paragraph numbering; the rest were cut into ~110-word "
            "sentence windows by M1's zoner, and ParaRef.kind == 'window' says "
            "so.  Never render a window as a paragraph number."
        ),
    }
    (out / IDX_FILE).write_text(json.dumps(header, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    if verbose:
        print(f"  {row} paragraphs from {n_docs} judgments -> {out / PARAS_FILE}")
        print(f"  printed numbering: {n_printed_docs}/{n_docs} "
              f"({header['printed_share']:.1%}) documents")
    return header


# ==========================================================================
# CLI
# ==========================================================================


def _main(argv: Optional[List[str]] = None) -> int:          # pragma: no cover
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.index.paragraphs")
    ap.add_argument("--build", action="store_true", help="build S1 from judgments.jsonl")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--get", metavar="DOC:PARA", help="print one paragraph")
    ap.add_argument("--doc", type=int, metavar="DOC_ID",
                    help="print every paragraph of one judgment with its citation")
    ap.add_argument("--label", metavar="DOC:LABEL",
                    help="resolve the court's printed paragraph number")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args(argv)

    if args.build:
        build()
        return 0
    if not ParagraphStore.exists():
        print("S1 is not built; run --build first", file=sys.stderr)
        return 1

    store = ParagraphStore.load()
    if args.stats:
        for k, v in store.stats().items():
            print(f"  {k:<28}{v}")
        return 0
    if args.get:
        d, _, p = args.get.partition(":")
        ref = store.ref(int(d), int(p or 1))
        if ref is None:
            print("no such paragraph", file=sys.stderr)
            return 1
        print(f"{ref.cite()}   [{ref.kind}]  zone={ref.zone}")
        print(f"  {ref.tooltip()}")
        print()
        print(store.text(int(d), int(p or 1)))
        return 0
    if args.label:
        d, _, lab = args.label.partition(":")
        pid = store.find_label(int(d), lab)
        print(f"doc {d} label {lab!r} -> para_id {pid}")
        return 0
    if args.doc is not None:
        paras = store.paras_of(args.doc)
        print(f"doc {args.doc}: {len(paras)} paragraphs, "
              f"numbering={store.numbering(args.doc)}")
        for p in paras[: args.limit]:
            ref = store.ref_for(args.doc, p)
            print(f"\n  {ref.short():<8}{ref.cite():<22}[{ref.kind}] zone={p.zone}")
            print(f"    {p.text[:200]}...")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
