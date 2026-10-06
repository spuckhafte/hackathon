"""S3 - the positional, zoned inverted index  (Member 1, wave 2).

Built with **SPIMI** (single-pass in-memory indexing, Manning et al. ch. 4):
invert ``config.SPIMI_BLOCK_DOCS`` documents into an in-memory dictionary, flush
that block to disk *sorted by term*, and finally merge every block with a heap.
The corpus is 13,878 documents of ~5,800 tokens, so ~81 million positions: one
global dictionary of Python lists would not fit in RAM, which is the whole
reason SPIMI exists.  Each block is built by a separate process, so the
expensive part (``analyze()``, ~24 docs/s per core) runs 11-way parallel and the
merge stays single-threaded and sequential, exactly as the algorithm wants.

On-disk layout, per index directory
----------------------------------
``postings.bin``     one contiguous little-endian int32 blob, memory-mapped on
                     load.  Per term, in this order:

                         doc_ids      [df]                ascending
                         zone_tf      [df * 6]            config.ZONES order
                         pos_offsets  [df + 1]            prefix sums
                         positions    [n_positions]       ascending per doc
                         skips        [ceil(df / stride)]  doc_id values

``dictionary.json``  parallel arrays (not one object per term - that triples the
                     file size) with the sorted term list and each term's
                     ``df``, int32 offset, position count and skip stride.
                     The term list stays sorted so ``prefix_terms("bail")``
                     is two bisections, which is what the ``bail!`` root
                     operator needs.
``docmeta.npz``      ``doc_len``, ``zone_len`` (n_docs x 6), ``doc_norm``, and
                     the sentence / paragraph boundary arrays.

Document ids
------------
The index is addressed by **Member 2's manifest ``doc_id``**, not by a private
row number, so no store ever has to translate ids.  The 64 manifest rows that
failed extraction simply have ``doc_len == 0``; ``doc_count`` is the number of
documents actually indexed (the N used by idf), while the per-document arrays
are ``n_doc_slots`` long.

Positions, sentences and paragraphs
-----------------------------------
``pos`` is a running offset over the whole document, zones concatenated in
``config.ZONES`` order.  Instead of a per-position ``sent_id``/``para_id`` array
(81M x 2 x int32 = 648 MB) we persist the **start position of every sentence and
every paragraph** and answer ``sent_ids_at(doc_id, positions)`` with a single
vectorised ``np.searchsorted`` over that document's slice.  It is 40x smaller,
it is still one numpy call per postings list rather than a rescan of text, and
proximity always has an *array* of positions in hand anyway - which is the shape
searchsorted wants.  ``zone_at`` works the same way off ``cumsum(zone_len)``.

**``para_id`` is a window id for most documents.**  Only 28% of the corpus
printed its own paragraph numbers; for the other 72% ``nyaya.zones`` segments
into ~110-word sentence windows because PyMuPDF returns no blank lines for these
PDFs.  So ``/p`` means "within the same ~110-word window" on most documents and
"within the court's own numbered paragraph" on the rest.  That has to be said in
the report; it is a property of the source PDFs, not a choice.

Two indexes, not three
----------------------
``index/``           the real system: ``canonicalise=True``, ``stem=True``, all
                     six zones, the measured legal stop list.
``index/baseline/``  B0 as specified in ``contracts.MODES``: plain tokenizer
                     (``canonicalise=False``), standard English stop words only,
                     everything in a single ``body`` zone.

**S1 does not need a third index.**  S1 is "B0 + statute canonicalisation +
legal stop list", which is the *real* index scored with uniform zone weights -
a scorer flag, not a rebuild.  Likewise S2/S2p/S3/S4 all read ``index/``.

CLI
---
    python -m nyaya.index.positional build                 # the real index
    python -m nyaya.index.positional build --baseline      # B0's index
    python -m nyaya.index.positional stats
    python -m nyaya.index.positional term privaci          # a real postings list
    python -m nyaya.index.positional top --n 20
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import shutil
import sys
import time
from array import array
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import Posting, PostingsList
from nyaya.utils import ensure_utf8_stdout, human, timed

#: Workers for the inversion pass.  Fewer than EXTRACT_PROCESSES on purpose:
#: each worker holds a whole SPIMI block of postings, so this pass is bound by
#: memory rather than by CPU.  Measured on a 15 GB machine, 11 workers paged and
#: stalled; 6 runs flat out.
INDEX_PROCESSES = max(1, min(6, (os.cpu_count() or 4) - 1))

NZONES = len(config.ZONES)
ZONE_ID = config.ZONE_ID

#: Everything on disk is int32: doc ids (<= 14k), term frequencies and positions
#: (<= ~200k per document) all fit, and one dtype means the whole postings file
#: is a single mmap with no struct unpacking in the hot loop.
DT = np.int32
ITEMSIZE = 4

POSTINGS_FILE = "postings.bin"
DICTIONARY_FILE = "dictionary.json"
DOCMETA_FILE = "docmeta.npz"
BLOCK_DIR = "_spimi_blocks"


# ==========================================================================
# 1. Build profiles - the ablation is a flag, not a code change
# ==========================================================================

@dataclass(frozen=True)
class IndexProfile:
    """How one index is analysed.  Two instances exist; see the docstring."""

    name: str
    out: Path
    canonicalise: bool = True
    stem: bool = True
    zoned: bool = True
    stoplist: str = "legal"        # 'legal' (measured) | 'standard' (B0)

    def as_json(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in
             ("name", "canonicalise", "stem", "zoned", "stoplist")}
        d["out"] = str(self.out)
        return d


def full_profile(out: Optional[Path] = None) -> IndexProfile:
    return IndexProfile(name="full", out=Path(out or config.INDEX))


def baseline_profile(out: Optional[Path] = None) -> IndexProfile:
    return IndexProfile(
        name="baseline",
        out=Path(out or (config.INDEX / "baseline")),
        canonicalise=False,
        stem=True,            # B0 is "flat tf-idf lnc.ltc, plain tokenizer";
        zoned=False,          # the stemming ablation is a separate flag
        stoplist="standard",
    )


def _is_scoring_stopword(term: str, stoplist: str) -> bool:
    """Which stop list excludes a term from the tf-idf / BM25F *vector*.

    Stop words stay in the positional index either way - a phrase query needs
    "of" to be there - they are only dropped from the scoring vectors.
    """
    from nyaya import stoplist as sl

    if not term or "§" in term:      # a statute token is never a stop word
        return False
    if stoplist == "standard":
        return term in sl.STANDARD_STOPWORDS
    return sl.is_scoring_stopword(term)


# ==========================================================================
# 2. Per-document analysis
# ==========================================================================

@dataclass
class DocTokens:
    """One document's token stream, already flattened across zones."""

    doc_id: int
    terms: List[str]
    zone_ids: array                 # 'i', one per token
    sent_starts: array              # 'i', ascending positions
    para_starts: array              # 'i', ascending positions
    zone_len: List[int]
    n_tokens: int
    statute_tokens: List[str]
    acts: List[str]


def _zone_texts(record: Dict[str, Any], zoned: bool,
                include_statutes: bool = True) -> List[Tuple[str, str, int]]:
    """``[(zone, text, para_id), ...]`` in ``config.ZONES`` order.

    ``body`` and ``order`` are emitted paragraph by paragraph so that every
    token carries the paragraph it came from; the four short zones are one
    chunk each with ``para_id = 0``.

    ``include_statutes`` is False for B0.  That zone is *synthetic* - it holds
    canonical ``ipc§302`` tokens, not text from the judgment - so folding it into
    B0's single body zone would smuggle statute canonicalisation into the
    baseline and quietly destroy the "canonicalisation is worth N points of MAP"
    experiment.
    """
    zones = record.get("zones") or {}
    paras = record.get("paras") or []
    out: List[Tuple[str, str, int]] = []
    for z in config.ZONES:
        if z == "statutes" and not include_statutes:
            continue
        if z in ("body", "order"):
            for p in paras:
                if p.get("zone") == z and p.get("text"):
                    out.append((z, p["text"], int(p.get("id") or 0)))
        else:
            txt = zones.get(z) or ""
            if txt.strip():
                out.append((z, txt, 0))
    if not zoned:
        # B0 has no zones: same text, same order, all labelled 'body'.
        out = [("body", t, pid) for _z, t, pid in out]
    return out


def analyse_record(record: Dict[str, Any], profile: IndexProfile) -> DocTokens:
    """Run Member 3's ``analyze()`` over one record and flatten the result.

    The statutes zone is synthesised here if it is still empty: ``zones.py``
    leaves it blank in wave 1 and both S2 zone scoring and the ``act:`` filter
    depend on it, so the index builder is the last place it can be filled.
    """
    from nyaya.analyze import analyze
    from nyaya import statutes as stat_mod
    from nyaya import zones as zone_mod
    from nyaya.contracts import JudgmentRecord

    # --- statutes zone ---------------------------------------------------
    # Only for a canonicalising profile: B0 must not see these tokens at all.
    statute_tokens = list(record.get("statute_tokens") or [])
    acts = list(record.get("acts") or [])
    if profile.canonicalise and not (record.get("zones") or {}).get("statutes"):
        body = "\n".join(p.get("text", "") for p in (record.get("paras") or []))
        statute_tokens = stat_mod.extract_statutes(body)
        shim = JudgmentRecord(doc_id=int(record["doc_id"]),
                              source_path=str(record.get("source_path") or ""),
                              title=str(record.get("title") or ""),
                              zones=dict(record.get("zones") or {}))
        zone_mod.attach_statute_zone(shim, statute_tokens)
        record["zones"] = shim.zones
        statute_tokens = shim.statute_tokens
        acts = shim.acts

    terms: List[str] = []
    zone_ids = array("i")
    sent_starts = array("i")
    para_starts = array("i")
    zone_len = [0] * NZONES

    pos = 0
    sent_base = -1          # forces a boundary at the first token
    last_sent_key: Optional[Tuple[str, int, int]] = None
    last_para_key: Optional[Tuple[str, int]] = None

    for zone, text, para_id in _zone_texts(record, profile.zoned,
                                           include_statutes=profile.canonicalise):
        toks = analyze(text, zone=zone, para_id=para_id,
                       stem=profile.stem, canonicalise=profile.canonicalise,
                       start_pos=pos)
        if not toks:
            continue
        zid = ZONE_ID[zone]
        for t in toks:
            # Sentence ids restart at 0 inside every analyze() call, so the key
            # is (zone, para_id, local sent_id): a new key starts a sentence.
            skey = (zone, para_id, t.sent_id)
            if skey != last_sent_key:
                sent_starts.append(pos)
                last_sent_key = skey
            pkey = (zone, para_id)
            if pkey != last_para_key:
                para_starts.append(pos)
                last_para_key = pkey
            terms.append(t.term)
            zone_ids.append(zid)
            pos += 1
        zone_len[zid] += len(toks)

    return DocTokens(
        doc_id=int(record["doc_id"]), terms=terms, zone_ids=zone_ids,
        sent_starts=sent_starts, para_starts=para_starts,
        zone_len=zone_len, n_tokens=pos,
        statute_tokens=statute_tokens, acts=acts,
    )


# ==========================================================================
# 3. SPIMI: one block
# ==========================================================================
#
# A block holds, per term, a single int32 array with one run per document:
#
#     doc_id, zone_tf[0..5], n_positions, positions...
#
# One array('i') per term rather than lists of Python ints: an int object is 28
# bytes against 4, and a block of 600 documents is ~3.5M positions.

ENTRY_HEAD = 1 + NZONES + 1          # doc_id + 6 zone tfs + n_positions


def _serialise_doc(dt: DocTokens, postings: Dict[str, array]) -> None:
    """Append one document's postings to the in-memory block."""
    per_term: Dict[str, Tuple[List[int], List[int]]] = {}
    for pos, (term, zid) in enumerate(zip(dt.terms, dt.zone_ids)):
        slot = per_term.get(term)
        if slot is None:
            slot = ([0] * NZONES, [])
            per_term[term] = slot
        slot[0][zid] += 1
        slot[1].append(pos)
    for term, (ztf, positions) in per_term.items():
        buf = postings.get(term)
        if buf is None:
            buf = array("i")
            postings[term] = buf
        buf.append(dt.doc_id)
        buf.extend(ztf)
        buf.append(len(positions))
        buf.extend(positions)


def _doc_norm(dt: DocTokens, stoplist: str) -> float:
    """lnc vector length: sqrt(sum over scoring terms of (1 + ln tf)^2).

    This is the denominator of the B0 baseline's cosine, so it must be computed
    over exactly the terms the baseline scores - stop words excluded, statute
    tokens always kept.
    """
    tf: Dict[str, int] = {}
    for term in dt.terms:
        tf[term] = tf.get(term, 0) + 1
    total = 0.0
    for term, n in tf.items():
        if _is_scoring_stopword(term, stoplist):
            continue
        w = 1.0 + math.log(n)
        total += w * w
    return math.sqrt(total)


@dataclass
class BlockStats:
    """Per-document numbers a block hands back to the merge step."""

    doc_ids: List[int] = field(default_factory=list)
    doc_len: List[int] = field(default_factory=list)
    zone_len: List[List[int]] = field(default_factory=list)
    doc_norm: List[float] = field(default_factory=list)
    sent_starts: List[array] = field(default_factory=list)
    para_starts: List[array] = field(default_factory=list)
    statute_tokens: Dict[int, List[str]] = field(default_factory=dict)
    acts: Dict[int, List[str]] = field(default_factory=dict)
    n_positions: int = 0


def write_block(block_id: int, records: Iterable[Dict[str, Any]],
                profile: IndexProfile, block_dir: Path) -> Dict[str, Any]:
    """Invert ``records`` in memory and flush one SPIMI block, sorted by term.

    ``records`` is consumed as an *iterator* on purpose.  A 600-document slice of
    judgments.jsonl is ~43 MB of JSON, which becomes several hundred MB of Python
    dicts; holding all 600 at once in each of 11 workers put this 15 GB machine
    into swap and the build stopped making progress.  Parsing one line at a time
    and dropping it as soon as it is inverted keeps a worker's live set to one
    document plus the block's own postings.
    """
    postings: Dict[str, array] = {}
    stats = BlockStats()
    for rec in records:
        dt = analyse_record(rec, profile)
        rec.clear()                    # the paragraphs are the bulk; let them go
        if dt.n_tokens == 0:
            continue
        _serialise_doc(dt, postings)
        stats.doc_ids.append(dt.doc_id)
        stats.doc_len.append(dt.n_tokens)
        stats.zone_len.append(dt.zone_len)
        stats.doc_norm.append(_doc_norm(dt, profile.stoplist))
        stats.sent_starts.append(dt.sent_starts)
        stats.para_starts.append(dt.para_starts)
        stats.statute_tokens[dt.doc_id] = dt.statute_tokens
        stats.acts[dt.doc_id] = dt.acts
        stats.n_positions += dt.n_tokens

    block_dir.mkdir(parents=True, exist_ok=True)
    bin_path = block_dir / f"blk{block_id:04d}.bin"
    idx_path = block_dir / f"blk{block_id:04d}.json"
    meta_path = block_dir / f"blk{block_id:04d}.npz"

    terms = sorted(postings)                  # the "sorted by term" of SPIMI
    offsets: List[int] = []
    lengths: List[int] = []
    dfs: List[int] = []
    cursor = 0
    with bin_path.open("wb") as fh:
        for term in terms:
            buf = postings[term]
            buf.tofile(fh)
            offsets.append(cursor)
            lengths.append(len(buf))
            dfs.append(_count_entries(buf))
            cursor += len(buf)
    idx_path.write_text(
        json.dumps({"terms": terms, "off": offsets, "len": lengths, "df": dfs}),
        encoding="utf-8")

    sent_flat, sent_off = _flatten(stats.sent_starts)
    para_flat, para_off = _flatten(stats.para_starts)
    np.savez(
        meta_path,
        doc_ids=np.asarray(stats.doc_ids, dtype=DT),
        doc_len=np.asarray(stats.doc_len, dtype=DT),
        zone_len=np.asarray(stats.zone_len, dtype=DT).reshape(-1, NZONES),
        doc_norm=np.asarray(stats.doc_norm, dtype=np.float32),
        sent_flat=sent_flat, sent_off=sent_off,
        para_flat=para_flat, para_off=para_off,
    )
    statutes_path = block_dir / f"blk{block_id:04d}.statutes.json"
    statutes_path.write_text(json.dumps(
        {"statute_tokens": {str(k): v for k, v in stats.statute_tokens.items()},
         "acts": {str(k): v for k, v in stats.acts.items()}}), encoding="utf-8")
    return {
        "block_id": block_id, "n_docs": len(stats.doc_ids),
        "n_terms": len(terms), "n_positions": stats.n_positions,
        "n_ints": cursor,
    }


def _count_entries(buf: array) -> int:
    """df of one term inside a block payload: walk the variable-length runs."""
    i = 0
    n = 0
    L = len(buf)
    while i < L:
        npos = buf[i + 1 + NZONES]
        i += ENTRY_HEAD + npos
        n += 1
    return n


def _flatten(chunks: Sequence[array]) -> Tuple[np.ndarray, np.ndarray]:
    off = np.zeros(len(chunks) + 1, dtype=np.int64)
    for i, c in enumerate(chunks):
        off[i + 1] = off[i] + len(c)
    flat = np.empty(int(off[-1]), dtype=DT)
    for i, c in enumerate(chunks):
        flat[off[i]:off[i + 1]] = np.frombuffer(c, dtype=np.int32)
    return flat, off


def _block_worker(job: Tuple[int, str, int, int, Dict[str, Any], str]
                  ) -> Dict[str, Any]:
    """Process-pool entry point: read my own slice of the JSONL, invert, flush.

    Each worker reads the file itself from a byte offset instead of being handed
    pickled records - the corpus JSONL is ~960 MB and pickling it through the
    pool would cost more than the analysis.
    """
    block_id, src, byte_start, n_lines, prof, block_dir = job
    profile = IndexProfile(name=prof["name"], out=Path(prof["out"]),
                           canonicalise=prof["canonicalise"], stem=prof["stem"],
                           zoned=prof["zoned"], stoplist=prof["stoplist"])
    def _stream():
        with open(src, "r", encoding="utf-8") as fh:
            fh.seek(byte_start)
            for _ in range(n_lines):
                line = fh.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)

    return write_block(block_id, _stream(), profile, Path(block_dir))


def _line_offsets(path: Path) -> List[int]:
    """Byte offset of every line, so workers can seek instead of scanning."""
    offs: List[int] = []
    with path.open("rb") as fh:
        pos = 0
        for line in fh:
            offs.append(pos)
            pos += len(line)
    return offs


# ==========================================================================
# 4. SPIMI: the heap merge
# ==========================================================================

class _BlockReader:
    """Sequential reader over one block's terms, in sorted order."""

    def __init__(self, block_dir: Path, block_id: int):
        self.bin_path = block_dir / f"blk{block_id:04d}.bin"
        idx = json.loads((block_dir / f"blk{block_id:04d}.json")
                         .read_text(encoding="utf-8"))
        self.terms: List[str] = idx["terms"]
        self.off: List[int] = idx["off"]
        self.len: List[int] = idx["len"]
        self.df: List[int] = idx["df"]
        self.i = 0
        self._mm = np.memmap(self.bin_path, dtype=DT, mode="r") \
            if self.bin_path.stat().st_size else np.empty(0, dtype=DT)

    def peek(self) -> Optional[str]:
        return self.terms[self.i] if self.i < len(self.terms) else None

    def take(self) -> Tuple[str, np.ndarray, int]:
        t = self.terms[self.i]
        o, n, d = self.off[self.i], self.len[self.i], self.df[self.i]
        self.i += 1
        return t, np.asarray(self._mm[o:o + n]), d


def merge_blocks(block_ids: Sequence[int], block_dir: Path, out_dir: Path,
                 profile: IndexProfile, verbose: bool = True) -> Dict[str, Any]:
    """Heap-merge the blocks into ``postings.bin`` + ``dictionary.json``.

    Blocks are produced in ascending document order and each block's runs are in
    ascending doc_id, so concatenating a term's payloads in block order already
    yields ascending ``doc_ids`` - the merge never has to sort.  That is the
    property SPIMI buys by processing documents in order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    readers = [_BlockReader(block_dir, b) for b in block_ids]
    heap: List[Tuple[str, int]] = []
    for ri, r in enumerate(readers):
        t = r.peek()
        if t is not None:
            heapq.heappush(heap, (t, ri))

    terms_out: List[str] = []
    df_out: List[int] = []
    off_out: List[int] = []
    npos_out: List[int] = []
    stride_out: List[int] = []

    cursor = 0                      # in int32 units
    total_postings = 0
    total_positions = 0
    post_path = out_dir / POSTINGS_FILE
    t0 = time.perf_counter()
    with post_path.open("wb") as out:
        while heap:
            term = heap[0][0]
            payloads: List[np.ndarray] = []
            df = 0
            while heap and heap[0][0] == term:
                _t, ri = heapq.heappop(heap)
                _t2, payload, d = readers[ri].take()
                payloads.append(payload)
                df += d
                nxt = readers[ri].peek()
                if nxt is not None:
                    heapq.heappush(heap, (nxt, ri))
            blob = _finalise_term(payloads, df)
            blob.tofile(out)
            terms_out.append(term)
            df_out.append(df)
            off_out.append(cursor)
            npos_out.append(positions_in(df, int(blob.size)))
            stride_out.append(skip_stride_for(df))
            cursor += int(blob.size)
            total_postings += df
            total_positions += npos_out[-1]
            if verbose and len(terms_out) % 100_000 == 0:
                print(f"  merged {len(terms_out)} terms "
                      f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr,
                      flush=True)

    meta = {
        "profile": profile.as_json(),
        "vocab_size": len(terms_out),
        "total_postings": total_postings,
        "total_positions": total_positions,
        "n_zones": NZONES,
        "zones": list(config.ZONES),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "spimi_blocks": len(block_ids),
        "spimi_block_docs": config.SPIMI_BLOCK_DOCS,
    }
    (out_dir / DICTIONARY_FILE).write_text(json.dumps({
        "meta": meta, "terms": terms_out, "df": df_out,
        "off": off_out, "npos": npos_out, "stride": stride_out,
    }), encoding="utf-8")
    meta["postings_bytes"] = post_path.stat().st_size
    return meta


def record_ints(df: int, n_positions: int) -> int:
    """Length in int32 of one term's on-disk record.

        doc_ids[df] + zone_tf[df*NZONES] + pos_offsets[df+1]
                    + positions[n_positions] + skips[n_skips]
    """
    return df * (2 + NZONES) + 1 + n_positions + _n_skips(df)


def positions_in(df: int, record_size: int) -> int:
    """Inverse of :func:`record_ints` - how many positions a record holds."""
    return record_size - (df * (2 + NZONES) + 1 + _n_skips(df))


def skip_stride_for(df: int) -> int:
    """~sqrt(P) skip pointers over a postings list of length P (Manning 2.3)."""
    return max(1, int(math.sqrt(df))) if df > 16 else 0


def _n_skips(df: int) -> int:
    stride = skip_stride_for(df)
    return 0 if stride == 0 else (df + stride - 1) // stride


def _finalise_term(payloads: Sequence[np.ndarray], df: int) -> np.ndarray:
    """Block payload runs -> the final per-term record (see module docstring)."""
    doc_ids = np.empty(df, dtype=DT)
    ztf = np.empty(df * NZONES, dtype=DT)
    pos_offsets = np.empty(df + 1, dtype=DT)
    pos_chunks: List[np.ndarray] = []
    pos_offsets[0] = 0
    k = 0
    running = 0
    for payload in payloads:
        i = 0
        L = payload.size
        while i < L:
            doc_ids[k] = payload[i]
            ztf[k * NZONES:(k + 1) * NZONES] = payload[i + 1:i + 1 + NZONES]
            npos = int(payload[i + 1 + NZONES])
            start = i + ENTRY_HEAD
            pos_chunks.append(payload[start:start + npos])
            running += npos
            k += 1
            pos_offsets[k] = running
            i = start + npos
    positions = (np.concatenate(pos_chunks) if pos_chunks
                 else np.empty(0, dtype=DT))
    stride = skip_stride_for(df)
    skips = (doc_ids[::stride].copy() if stride
             else np.empty(0, dtype=DT))
    return np.concatenate([doc_ids, ztf, pos_offsets,
                           positions.astype(DT, copy=False), skips])


# ==========================================================================
# 5. Build
# ==========================================================================

def build(profile: Optional[IndexProfile] = None,
          src: Optional[Path] = None,
          limit: Optional[int] = None,
          processes: Optional[int] = None,
          keep_blocks: bool = False,
          persist_statutes: bool = True,
          verbose: bool = True) -> Dict[str, Any]:
    """Build one index end to end and return its metadata."""
    profile = profile or full_profile()
    src = Path(src or config.JUDGMENTS_JSONL)
    if not src.exists():
        raise SystemExit(f"{src} does not exist - run python -m nyaya.ingest build")
    processes = processes or INDEX_PROCESSES
    out_dir = Path(profile.out)
    block_dir = out_dir / BLOCK_DIR
    if block_dir.exists():
        shutil.rmtree(block_dir, ignore_errors=True)
    block_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    offs = _line_offsets(src)
    if limit:
        offs = offs[:limit]
    n_docs = len(offs)
    step = config.SPIMI_BLOCK_DOCS
    jobs = []
    for b, start in enumerate(range(0, n_docs, step)):
        jobs.append((b, str(src), offs[start], min(step, n_docs - start),
                     profile.as_json(), str(block_dir)))
    if verbose:
        print(f"[{profile.name}] {n_docs} documents -> {len(jobs)} SPIMI blocks "
              f"of {step} on {processes} processes", file=sys.stderr)

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    if processes <= 1:
        for job in jobs:
            results.append(_block_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            for i, res in enumerate(pool.map(_block_worker, jobs), 1):
                results.append(res)
                if verbose:
                    done = sum(r["n_docs"] for r in results)
                    print(f"  block {i}/{len(jobs)}  {done} docs  "
                          f"{done / max(time.perf_counter() - t0, 1e-9):.1f} doc/s",
                          file=sys.stderr, flush=True)
    invert_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    block_ids = [r["block_id"] for r in sorted(results, key=lambda r: r["block_id"])]
    meta = merge_blocks(block_ids, block_dir, out_dir, profile, verbose=verbose)
    merge_s = time.perf_counter() - t0

    docmeta = _merge_docmeta(block_ids, block_dir, out_dir)
    meta.update(docmeta)
    meta["invert_seconds"] = round(invert_s, 1)
    meta["merge_seconds"] = round(merge_s, 1)
    meta["total_seconds"] = round(time.perf_counter() - t_all, 1)

    if persist_statutes and profile.name == "full" and not limit:
        meta["statutes_written"] = _persist_statutes(block_ids, block_dir, src)

    # the dictionary already landed; rewrite it with the timings included
    dpath = out_dir / DICTIONARY_FILE
    d = json.loads(dpath.read_text(encoding="utf-8"))
    d["meta"].update({k: v for k, v in meta.items() if k != "profile"})
    dpath.write_text(json.dumps(d), encoding="utf-8")

    if not keep_blocks:
        shutil.rmtree(block_dir, ignore_errors=True)
    return meta


def _merge_docmeta(block_ids: Sequence[int], block_dir: Path,
                   out_dir: Path) -> Dict[str, Any]:
    """Collect the per-document arrays, addressed by manifest ``doc_id``."""
    parts = [np.load(block_dir / f"blk{b:04d}.npz") for b in block_ids]
    all_ids = np.concatenate([p["doc_ids"] for p in parts]) if parts \
        else np.empty(0, dtype=DT)
    n_slots = int(all_ids.max()) + 1 if all_ids.size else 0

    doc_len = np.zeros(n_slots, dtype=DT)
    zone_len = np.zeros((n_slots, NZONES), dtype=DT)
    doc_norm = np.zeros(n_slots, dtype=np.float32)
    sent_chunks: Dict[int, np.ndarray] = {}
    para_chunks: Dict[int, np.ndarray] = {}

    for p in parts:
        ids = p["doc_ids"]
        doc_len[ids] = p["doc_len"]
        zone_len[ids] = p["zone_len"]
        doc_norm[ids] = p["doc_norm"]
        so, po = p["sent_off"], p["para_off"]
        sf, pf = p["sent_flat"], p["para_flat"]
        for k, d in enumerate(ids):
            sent_chunks[int(d)] = sf[so[k]:so[k + 1]]
            para_chunks[int(d)] = pf[po[k]:po[k + 1]]

    sent_off = np.zeros(n_slots + 1, dtype=np.int64)
    para_off = np.zeros(n_slots + 1, dtype=np.int64)
    for d in range(n_slots):
        sent_off[d + 1] = sent_off[d] + len(sent_chunks.get(d, ()))
        para_off[d + 1] = para_off[d] + len(para_chunks.get(d, ()))
    sent_flat = np.empty(int(sent_off[-1]), dtype=DT)
    para_flat = np.empty(int(para_off[-1]), dtype=DT)
    for d in range(n_slots):
        s = sent_chunks.get(d)
        if s is not None and len(s):
            sent_flat[sent_off[d]:sent_off[d + 1]] = s
        q = para_chunks.get(d)
        if q is not None and len(q):
            para_flat[para_off[d]:para_off[d + 1]] = q

    np.savez(out_dir / DOCMETA_FILE,
             doc_len=doc_len, zone_len=zone_len, doc_norm=doc_norm,
             sent_flat=sent_flat, sent_off=sent_off,
             para_flat=para_flat, para_off=para_off,
             indexed=np.asarray(sorted(int(i) for i in all_ids), dtype=DT))
    for p in parts:
        p.close()
    return {
        "doc_count": int(all_ids.size),
        "n_doc_slots": n_slots,
        "n_sentences": int(sent_flat.size),
        "n_paragraphs": int(para_flat.size),
    }


def _persist_statutes(block_ids: Sequence[int], block_dir: Path,
                      src: Path) -> int:
    """Write the statute tokens back into ``judgments.jsonl``.

    The ``statutes`` zone was empty in all 13,878 records after wave 1, and both
    S2 zone scoring and Member 2's ``act:`` filter read it, so the enrichment has
    to be persisted rather than living only inside this index.
    """
    from nyaya import zones as zone_mod
    from nyaya.contracts import JudgmentRecord

    tokens: Dict[int, List[str]] = {}
    acts: Dict[int, List[str]] = {}
    for b in block_ids:
        p = block_dir / f"blk{b:04d}.statutes.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        tokens.update({int(k): v for k, v in d["statute_tokens"].items()})
        acts.update({int(k): v for k, v in d["acts"].items()})

    tmp = src.with_suffix(".jsonl.tmp")
    n = 0
    with src.open("r", encoding="utf-8") as fin, \
            tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = int(row["doc_id"])
            toks = tokens.get(did)
            if toks is not None:
                shim = JudgmentRecord(doc_id=did, source_path="", title="",
                                      zones=dict(row.get("zones") or {}))
                zone_mod.attach_statute_zone(shim, toks)
                row["statute_tokens"] = shim.statute_tokens
                row["acts"] = shim.acts
                row["zones"] = shim.zones
                n += 1 if toks else 0
            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")
    os.replace(tmp, src)
    return n


# ==========================================================================
# 6. Read side
# ==========================================================================

class MmapPostings(PostingsList):
    """One term's postings, as views into the memory-mapped ``postings.bin``."""

    __slots__ = ("term", "df", "_blob", "_npos", "_stride")

    def __init__(self, term: str, blob: np.ndarray, df: int, npos: int,
                 stride: int):
        self.term = term
        self.df = int(df)
        self._blob = blob
        self._npos = int(npos)
        self._stride = int(stride)

    # ---- the three slices of the record ------------------------------
    @property
    def doc_ids(self) -> np.ndarray:
        return self._blob[:self.df]

    @property
    def _ztf(self) -> np.ndarray:
        return self._blob[self.df:self.df + self.df * NZONES]

    @property
    def _pos_offsets(self) -> np.ndarray:
        a = self.df + self.df * NZONES
        return self._blob[a:a + self.df + 1]

    @property
    def _positions(self) -> np.ndarray:
        a = self.df + self.df * NZONES + self.df + 1
        return self._blob[a:a + self._npos]

    @property
    def skips(self) -> np.ndarray:
        a = self.df + self.df * NZONES + self.df + 1 + self._npos
        return self._blob[a:]

    # ---- the contract ------------------------------------------------
    def skip_stride(self) -> int:
        return self._stride

    def index_of(self, doc_id: int) -> int:
        """Row of ``doc_id``, or -1.  Uses the skip pointers when there are any.

        With skips the search is the textbook two-step: binary-search the
        sqrt(P)-spaced skip array to find the block, then search inside that
        block only.  Without them (short lists) it is a plain binary search.
        """
        d = self.doc_ids
        if self._stride and self.skips.size:
            b = int(np.searchsorted(self.skips, doc_id, side="right")) - 1
            if b < 0:
                return -1
            lo = b * self._stride
            hi = min(lo + self._stride, self.df)
            j = int(np.searchsorted(d[lo:hi], doc_id)) + lo
        else:
            j = int(np.searchsorted(d, doc_id))
        return j if j < self.df and int(d[j]) == doc_id else -1

    def positions_of(self, doc_id: int) -> np.ndarray:
        j = self.index_of(doc_id)
        if j < 0:
            return np.empty(0, dtype=DT)
        off = self._pos_offsets
        return self._positions[int(off[j]):int(off[j + 1])]

    def zone_tf_of(self, doc_id: int) -> Dict[str, int]:
        j = self.index_of(doc_id)
        if j < 0:
            return {}
        row = self._ztf[j * NZONES:(j + 1) * NZONES]
        return {z: int(row[i]) for i, z in enumerate(config.ZONES) if row[i]}

    def zone_tf_matrix(self) -> np.ndarray:
        """(df, 6) int32 - what a BM25F scorer wants in one shot."""
        return self._ztf.reshape(self.df, NZONES)

    def tf_array(self) -> np.ndarray:
        return self.zone_tf_matrix().sum(axis=1)

    def __iter__(self) -> Iterator[Posting]:
        d = self.doc_ids
        ztf = self.zone_tf_matrix()
        off = self._pos_offsets
        pos = self._positions
        for j in range(self.df):
            yield Posting(
                doc_id=int(d[j]),
                zone_tf={z: int(ztf[j, i]) for i, z in enumerate(config.ZONES)
                         if ztf[j, i]},
                positions=pos[int(off[j]):int(off[j + 1])],
            )

    def __len__(self) -> int:
        return self.df

    def __repr__(self) -> str:
        return (f"<MmapPostings {self.term!r} df={self.df} "
                f"positions={self._npos} skip_stride={self._stride}>")


class PositionalIndex:
    """Read side of S3.  Signatures frozen in ``docs/CONTRACTS.md`` Section 4."""

    def __init__(self, path: Path, dictionary: Dict[str, Any],
                 blob: np.ndarray, docmeta: Dict[str, np.ndarray]):
        self.path = Path(path)
        self.meta: Dict[str, Any] = dictionary["meta"]
        self._terms: List[str] = dictionary["terms"]
        self._df = np.asarray(dictionary["df"], dtype=np.int64)
        self._off = np.asarray(dictionary["off"], dtype=np.int64)
        self._npos = np.asarray(dictionary["npos"], dtype=np.int64)
        self._stride = np.asarray(dictionary["stride"], dtype=np.int64)
        self._row = {t: i for i, t in enumerate(self._terms)}
        self._blob = blob
        self.doc_len: np.ndarray = docmeta["doc_len"]
        self.zone_len: np.ndarray = docmeta["zone_len"]
        self.doc_norm: np.ndarray = docmeta["doc_norm"]
        self._sent_flat = docmeta["sent_flat"]
        self._sent_off = docmeta["sent_off"]
        self._para_flat = docmeta["para_flat"]
        self._para_off = docmeta["para_off"]
        self.indexed: np.ndarray = docmeta["indexed"]
        self._zone_cum: Optional[np.ndarray] = None

    # ---- loading -----------------------------------------------------
    @classmethod
    def load(cls, path: Path | str = config.INDEX) -> "PositionalIndex":
        path = Path(path)
        dic_path = path / DICTIONARY_FILE
        if not dic_path.exists():
            raise SystemExit(
                f"{dic_path} does not exist - run\n"
                f"    python -m nyaya.index.positional build"
                + (" --baseline" if path.name == "baseline" else ""))
        dictionary = json.loads(dic_path.read_text(encoding="utf-8"))
        post = path / POSTINGS_FILE
        blob = (np.memmap(post, dtype=DT, mode="r") if post.stat().st_size
                else np.empty(0, dtype=DT))
        with np.load(path / DOCMETA_FILE) as z:
            docmeta = {k: z[k] for k in z.files}
        return cls(path, dictionary, blob, docmeta)

    # ---- the contract ------------------------------------------------
    @property
    def doc_count(self) -> int:
        """N for idf: documents actually indexed, not the number of id slots."""
        return int(self.meta.get("doc_count") or self.indexed.size)

    @property
    def n_doc_slots(self) -> int:
        return int(self.doc_len.size)

    @property
    def vocab_size(self) -> int:
        return len(self._terms)

    def df(self, term: str) -> int:
        i = self._row.get(term)
        return int(self._df[i]) if i is not None else 0

    def idf(self, term: str) -> float:
        """ln(1 + (N - df + 0.5) / (df + 0.5)) - the BM25 probabilistic idf.

        The +0.5 smoothing keeps it positive for a term in every document, which
        a plain ln(N/df) does not, and the whole ranking would otherwise flip
        sign on words like "court".
        """
        d = self.df(term)
        n = self.doc_count
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    def postings(self, term: str) -> Optional[MmapPostings]:
        i = self._row.get(term)
        if i is None:
            return None
        df = int(self._df[i])
        npos = int(self._npos[i])
        stride = int(self._stride[i])
        start = int(self._off[i])
        length = record_ints(df, npos)
        return MmapPostings(term, self._blob[start:start + length],
                            df, npos, stride)

    def prefix_terms(self, prefix: str) -> List[str]:
        """Dictionary prefix scan for the ``bail!`` root operator.

        The term list is sorted on disk, so this is one bisection plus a walk -
        no regex over the vocabulary.
        """
        if not prefix:
            return []
        lo = bisect_left(self._terms, prefix)
        out: List[str] = []
        for i in range(lo, len(self._terms)):
            t = self._terms[i]
            if not t.startswith(prefix):
                break
            out.append(t)
        return out

    # ---- sentence / paragraph / zone lookup --------------------------
    def sent_ids_at(self, doc_id: int, positions: Sequence[int]) -> np.ndarray:
        """Sentence id of each position, vectorised (for the ``/s`` operator)."""
        a, b = int(self._sent_off[doc_id]), int(self._sent_off[doc_id + 1])
        starts = self._sent_flat[a:b]
        if starts.size == 0:
            return np.zeros(len(positions), dtype=np.int64)
        return np.searchsorted(starts, np.asarray(positions), side="right") - 1

    def para_ids_at(self, doc_id: int, positions: Sequence[int]) -> np.ndarray:
        """Paragraph id of each position (for ``/p``).

        Reminder from the module docstring: on 72% of documents a "paragraph" is
        a ~110-word window invented by ``nyaya.zones``, not the court's own
        numbered paragraph.
        """
        a, b = int(self._para_off[doc_id]), int(self._para_off[doc_id + 1])
        starts = self._para_flat[a:b]
        if starts.size == 0:
            return np.zeros(len(positions), dtype=np.int64)
        return np.searchsorted(starts, np.asarray(positions), side="right") - 1

    def zone_bounds(self, doc_id: int) -> np.ndarray:
        """Cumulative zone lengths: position ``p`` is in zone
        ``searchsorted(bounds, p, 'right')``."""
        if self._zone_cum is None:
            self._zone_cum = np.cumsum(self.zone_len.astype(np.int64), axis=1)
        return self._zone_cum[doc_id]

    def zone_at(self, doc_id: int, positions: Sequence[int]) -> np.ndarray:
        """Zone id of each position - how proximity rejects a cross-zone match."""
        return np.searchsorted(self.zone_bounds(doc_id),
                               np.asarray(positions), side="right")

    def n_sentences(self, doc_id: int) -> int:
        return int(self._sent_off[doc_id + 1] - self._sent_off[doc_id])

    def n_paragraphs(self, doc_id: int) -> int:
        return int(self._para_off[doc_id + 1] - self._para_off[doc_id])

    # ---- reporting ---------------------------------------------------
    def terms_by_df(self, n: int = 20) -> List[Tuple[str, int]]:
        order = np.argsort(-self._df, kind="stable")[:n]
        return [(self._terms[int(i)], int(self._df[int(i)])) for i in order]

    def stats(self) -> Dict[str, Any]:
        post = self.path / POSTINGS_FILE
        dic = self.path / DICTIONARY_FILE
        dm = self.path / DOCMETA_FILE
        indexed = self.doc_len > 0
        return {
            "path": str(self.path),
            "profile": self.meta.get("profile"),
            "documents": self.doc_count,
            "doc_id_slots": self.n_doc_slots,
            "vocabulary": self.vocab_size,
            "total_postings": int(self._df.sum()),
            "total_positions": int(self._npos.sum()),
            "hapax_terms": int((self._df == 1).sum()),
            "mean_doc_len": round(float(self.doc_len[indexed].mean()), 1),
            "zone_tokens": {z: int(self.zone_len[:, i].sum())
                            for i, z in enumerate(config.ZONES)},
            "sentences": int(self._sent_flat.size),
            "paragraphs": int(self._para_flat.size),
            "postings_bin": human(post.stat().st_size),
            "dictionary_json": human(dic.stat().st_size),
            "docmeta_npz": human(dm.stat().st_size),
            "on_disk_total": human(post.stat().st_size + dic.stat().st_size
                                   + dm.stat().st_size),
            "invert_seconds": self.meta.get("invert_seconds"),
            "merge_seconds": self.meta.get("merge_seconds"),
            "total_seconds": self.meta.get("total_seconds"),
            "spimi_blocks": self.meta.get("spimi_blocks"),
        }


# ==========================================================================
# 7. CLI - the printouts the demo video shows
# ==========================================================================

def _print_term(idx: PositionalIndex, term: str, max_docs: int = 8,
                max_pos: int = 12) -> None:
    pl = idx.postings(term)
    print("=" * 78)
    if pl is None:
        print(f"term {term!r}: not in the dictionary "
              f"({idx.vocab_size} terms).  Prefix matches: "
              f"{idx.prefix_terms(term)[:8]}")
        return
    print(f"term        : {term!r}")
    print(f"df          : {pl.df}   ({pl.df / idx.doc_count:.2%} of "
          f"{idx.doc_count} documents)")
    print(f"idf         : {idx.idf(term):.4f}   "
          f"= ln(1 + (N - df + 0.5)/(df + 0.5))")
    print(f"positions   : {pl._npos}")
    print(f"skip stride : {pl.skip_stride()} "
          f"({pl.skips.size} skip pointers, ~sqrt(df))")
    print(f"{'doc_id':>7}  {'tf':>4}  zone tfs"
          f"{'':<26}  first positions")
    for j, p in enumerate(pl):
        if j >= max_docs:
            print(f"{'...':>7}  ({pl.df - max_docs} more documents)")
            break
        pos = list(map(int, p.positions[:max_pos]))
        tail = ", ..." if len(p.positions) > max_pos else ""
        zt = " ".join(f"{z}={n}" for z, n in p.zone_tf.items())
        sids = idx.sent_ids_at(p.doc_id, p.positions[:3])
        pids = idx.para_ids_at(p.doc_id, p.positions[:3])
        print(f"{p.doc_id:>7}  {p.tf:>4}  {zt:<34}  {pos}{tail}")
        print(f"{'':>7}  {'':>4}  first 3 -> sent {list(map(int, sids))}, "
              f"para {list(map(int, pids))}")


def _cmd_build(args) -> int:
    profiles = []
    out = Path(args.out) if args.out else None
    if args.baseline or args.both:
        profiles.append(baseline_profile(out / "baseline" if out else None))
    if not args.baseline or args.both:
        profiles.append(full_profile(out))
    for profile in profiles:
        with timed(f"build:{profile.name}"):
            meta = build(profile, limit=args.limit, processes=args.processes,
                         keep_blocks=args.keep_blocks,
                         persist_statutes=not args.no_statutes)
        print(f"--- {profile.name} ---")
        for k, v in meta.items():
            print(f"{k}: {v}")
    return 0


def _cmd_stats(args) -> int:
    idx = PositionalIndex.load(args.path)
    for k, v in idx.stats().items():
        print(f"{k}: {v}")
    print("\ndf of a few probe terms:")
    for t in args.probe:
        pl = idx.postings(t)
        print(f"  {t:<12} df={idx.df(t):>6}  idf={idx.idf(t):.4f}"
              f"  positions={pl._npos if pl else 0}")
    return 0


def _cmd_term(args) -> int:
    idx = PositionalIndex.load(args.path)
    for t in args.terms:
        _print_term(idx, t, max_docs=args.docs)
    return 0


def _cmd_top(args) -> int:
    idx = PositionalIndex.load(args.path)
    print(f"top {args.n} terms by document frequency "
          f"(N = {idx.doc_count} documents)")
    for rank, (t, d) in enumerate(idx.terms_by_df(args.n), 1):
        print(f"{rank:>3}. {t:<16} df={d:>6}  {d / idx.doc_count:>7.2%}"
              f"  idf={idx.idf(t):.4f}")
    return 0


def _cmd_prefix(args) -> int:
    idx = PositionalIndex.load(args.path)
    hits = idx.prefix_terms(args.prefix)
    print(f"{args.prefix}! -> {len(hits)} terms")
    for t in hits[: args.n]:
        print(f"  {t:<20} df={idx.df(t)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.index.positional",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--path", default=str(config.INDEX),
                    help="index directory to read (default: index/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="SPIMI build")
    b.add_argument("--baseline", action="store_true", help="build index/baseline")
    b.add_argument("--both", action="store_true", help="build both indexes")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--processes", type=int, default=INDEX_PROCESSES)
    b.add_argument("--keep-blocks", action="store_true")
    b.add_argument("--no-statutes", action="store_true",
                   help="do not write statute tokens back to judgments.jsonl")
    b.add_argument("--out", default=None,
                   help="write to this directory instead of index/ (for tests)")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("stats", help="real numbers for the README")
    s.add_argument("--probe", nargs="*",
                   default=["privaci", "surveil", "ipc§302", "const§21",
                            "bail", "court"])
    s.set_defaults(func=_cmd_stats)

    t = sub.add_parser("term", help="pretty-print one term's postings list")
    t.add_argument("terms", nargs="+")
    t.add_argument("--docs", type=int, default=8)
    t.set_defaults(func=_cmd_term)

    tp = sub.add_parser("top", help="most frequent terms by df")
    tp.add_argument("--n", type=int, default=20)
    tp.set_defaults(func=_cmd_top)

    p = sub.add_parser("prefix", help="dictionary prefix scan, for bail!")
    p.add_argument("prefix")
    p.add_argument("--n", type=int, default=25)
    p.set_defaults(func=_cmd_prefix)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
