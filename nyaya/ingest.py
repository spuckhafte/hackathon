"""Corpus acquisition and text extraction  (Member 1, wave 1).

Three jobs, in the order the pipeline runs them:

1. **download** - pull the judgment PDFs named by ``data/processed/manifest.parquet``
   straight out of the public AWS Open Data bucket.  It is a bulk release, so
   there is no crawling and no robots.txt to respect; we still cap concurrency
   at ``config.DOWNLOAD_THREADS`` to be a good citizen and we are resumable.
2. **extract** - PyMuPDF page-by-page text, then a cleaning pass that is
   specific to Supreme Court Reports typography: repeated running headers, the
   solitary ``A``-``H`` marginal letter column, end-of-line hyphenation, and the
   U+FFFD replacement characters these PDFs are full of.
3. **build**  - extraction + zone segmentation (``nyaya.zones``) written out as
   one :class:`~nyaya.contracts.JudgmentRecord` per line of
   ``data/processed/judgments.jsonl``.

Every attempted document is logged to ``data/processed/extract_log.csv`` with a
status, so the README can report coverage honestly instead of silently
reporting only the documents that happened to work.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nyaya import config
from nyaya.contracts import JudgmentRecord, Para
from nyaya.utils import ensure_utf8_stdout, timed, write_jsonl

# ==========================================================================
# 1. Manifest
# ==========================================================================

#: ``data/processed/manifest.parquet`` is Member 2's and is the single source of
#: truth for *which* judgments we index and for their ``doc_id``.  Every other
#: store (parametric index, citation graph, authority, paragraph store) is keyed
#: on that id, so this module never invents or renumbers one.
MANIFEST_COLUMNS = ("doc_id", "year", "path", "title")


def load_manifest():
    """Member 2's ``manifest.parquet``.  There is deliberately no fallback."""
    import pandas as pd

    if not config.MANIFEST_PARQUET.exists():
        raise SystemExit(
            f"{config.MANIFEST_PARQUET} does not exist.\n"
            "It is Member 2's artefact and the only source of doc_id - run\n"
            "    python -m nyaya.metadata manifest\n"
            "first.  ingest.py deliberately has no fallback sample: two "
            "sources of truth for doc_id would silently corrupt every store."
        )
    man = pd.read_parquet(config.MANIFEST_PARQUET)
    missing = [c for c in MANIFEST_COLUMNS if c not in man.columns]
    if missing:
        raise SystemExit(f"manifest.parquet is missing columns {missing}")
    return man


# ==========================================================================
# 2. Download
# ==========================================================================

def pdf_key(year: int, path: str) -> str:
    return f"data/pdf/year={int(year)}/english/{path}_EN.pdf"


def pdf_url(year: int, path: str) -> str:
    return config.BUCKET_URL.rstrip("/") + "/" + pdf_key(year, path)


def pdf_local(year: int, path: str) -> Path:
    return config.RAW_PDF / str(int(year)) / f"{path}_EN.pdf"


_tls = threading.local()


def _session():
    import requests

    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "NyayaTrace/1.0 (CSD358 coursework; bulk open-data)"
        _tls.session = s
    return s


def _fetch_one(year: int, path: str, retries: int = 3) -> Tuple[str, int]:
    """Return ``(status, bytes)``; status in {skip, ok, missing, error}."""
    dest = pdf_local(year, path)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pdf.part")
    url = pdf_url(year, path)
    for attempt in range(retries):
        try:
            resp = _session().get(url, timeout=60, stream=True)
            if resp.status_code == 404:
                return "missing", 0
            resp.raise_for_status()
            n = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    if chunk:
                        fh.write(chunk)
                        n += len(chunk)
            if n == 0:
                raise OSError("empty body")
            os.replace(tmp, dest)
            return "ok", n
        except Exception:
            tmp.unlink(missing_ok=True)       # never leave a partial file behind
            if attempt == retries - 1:
                return "error", 0
            time.sleep(0.4 * (attempt + 1))
    return "error", 0


def download(rows: Sequence[Tuple[int, str]], threads: int | None = None) -> Dict[str, int]:
    """Resumable bulk download.  ``rows`` is a sequence of ``(year, path)``."""
    threads = threads or config.DOWNLOAD_THREADS
    tally: Counter[str] = Counter()
    nbytes = 0
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(_fetch_one, y, p): (y, p) for y, p in rows}
        for fut in as_completed(futs):
            status, n = fut.result()
            tally[status] += 1
            nbytes += n
            done += 1
            if done % 250 == 0 or done == len(futs):
                dt = time.perf_counter() - t0
                print(
                    f"  {done}/{len(futs)}  {done / max(dt, 1e-9):.1f} doc/s  "
                    f"ok={tally['ok']} skip={tally['skip']} "
                    f"missing={tally['missing']} error={tally['error']}",
                    file=sys.stderr, flush=True,
                )
    tally["bytes"] = nbytes
    return dict(tally)


# ==========================================================================
# 3. Text extraction and cleaning
# ==========================================================================
#
# These PDFs are reproductions of the printed Supreme Court Reports.  Three
# artefacts of that typography wreck a naive extraction and all three are
# cheap to undo:
#
#   * a running header on every page ("[2017] 10 S.C.R. 104" / the case title),
#   * a marginal letter column: solitary capitals A..H down the page edge,
#     which PyMuPDF reads as lines of their own and which otherwise become
#     ~2000 spurious single-letter tokens per document,
#   * end-of-line hyphenation ("convic-\ntion"), which splits real terms in
#     two and so costs recall on exactly the words the query will use.

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
    "‌": "", "‍": "", "﻿": "", "…": "...",
    "�": "",          # these PDFs are littered with the replacement char
}
_TRANS = {ord(k): v for k, v in _LIGATURES.items()}

#: A line that is nothing but one capital letter is the SCR marginal column.
_MARGIN_LETTER = re.compile(r"^[A-H]$")
_PAGE_NUMBER = re.compile(r"^\(?\[?\s*\d{1,4}\s*\]?\)?$")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WS_RUN = re.compile(r"[ \t ]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _strip_controls(text: str) -> str:
    """Remove C0/C1 control characters except newline and tab.

    The Digital SCR PDFs embed a literal U+0008 in the running head
    (``"[2024] 10 S.C.R. \\x08"``), which is enough to make two copies of the
    same header look like different strings and so survive de-duplication.
    """
    return "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANS)
    return _strip_controls(text)


def strip_marginal_letters(lines: List[str]) -> List[str]:
    """Drop the solitary A-H marginal letters of the printed SCR page.

    Guarded: we only strip when the document actually has such a column (>= 8
    of them), so a genuine one-letter line in a modern PDF survives.
    """
    hits = sum(1 for ln in lines if _MARGIN_LETTER.match(ln.strip()))
    if hits < 8:
        return lines
    return [ln for ln in lines if not _MARGIN_LETTER.match(ln.strip())]


#: Lines near the top/bottom of a page that we test for "is this a running head".
HEADER_WINDOW = 4
#: Boilerplate the Digital SCR reprint stamps on pages regardless of position.
_BOILERPLATE = frozenset({
    "digital supreme court reports",
    "supreme court reports",
    "supreme court reports [2024] 10 s.c.r.",
    "s.c.r.", "scr", "supreme court of india",
})


def repeated_lines(pages: Sequence[str], share: float = 0.30,
                   global_share: float = 0.60) -> set:
    """Running header / footer lines, detected two ways.

    *Positional* - a short line in the first or last ``HEADER_WINDOW`` lines of
    a page that also appears in that band on >= ``share`` of the other pages.
    That is exactly what a running head is, and the position test keeps us from
    deleting a body line that happens to repeat.

    *Global* - any short line repeated on >= ``global_share`` of pages,
    wherever it sits.  The pre-1990 SCR scans put a running marginal block
    (case name, judge name, date) in the middle of the text flow, so position
    alone misses them.

    Why bother: the citation and case title are reprinted on every page, so
    without this their tf grows with page count and long judgments would
    out-rank short ones on their own title words - a ranking bug, not cosmetics.
    """
    n = len(pages)
    if n < 2:
        return set()
    band: Counter[str] = Counter()
    anywhere: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        edge = set(lines[:HEADER_WINDOW]) | set(lines[-HEADER_WINDOW:])
        band.update({ln for ln in edge if len(ln) <= 90})
        anywhere.update({ln for ln in set(lines) if len(ln) <= 90})
    band_cut = max(2, int(round(n * share)))
    glob_cut = max(2, int(round(n * global_share)))
    out = {ln for ln, c in band.items() if c >= band_cut}
    out |= {ln for ln, c in anywhere.items() if c >= glob_cut}
    return out


def clean_pages(pages: Sequence[str]) -> str:
    """Turn a list of raw page texts into one cleaned document string."""
    pages = [normalise_unicode(p) for p in pages]
    headers = repeated_lines(pages)
    out: List[str] = []
    for pno, page in enumerate(pages):
        lines = page.splitlines()
        lines = strip_marginal_letters(lines)
        kept = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                kept.append("")
                continue
            # The citation / party header of page 1 *is* the document header we
            # want to keep - zones.py reads the SCR cite and the bench out of
            # it - so the first band of the first page is never stripped.
            protected = pno == 0 and i < HEADER_WINDOW
            if not protected:
                if s in headers or s.lower().strip(" .") in _BOILERPLATE:
                    continue
                if _PAGE_NUMBER.match(s):
                    continue
            kept.append(ln.rstrip())
        out.append("\n".join(kept))
    text = "\n".join(out)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)        # convic-\ntion -> conviction
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


# --- language check ------------------------------------------------------
# Cheap and defensible: a judgment in English has (a) overwhelmingly ASCII
# alphabetic tokens and (b) a high hit rate on a tiny closed-class function
# word list.  Measured on this corpus a healthy English judgment scores
# ~0.40-0.48 on the stop-word rate; a Hindi/Devanagari or badly-OCR'd
# document falls off a cliff.  No language model, no extra dependency.

_ENGLISH_FUNCTION_WORDS = frozenset("""
a an the and or but if of in on at to for with by from as is are was were be been
being that this these those it its not no nor he she they them his her their we
us our you your which who whom whose what when where while how than then there
here any all each other such shall may must can could would should will do does
did have has had upon under over before after against between into about
""".split())

_WORD = re.compile(r"[A-Za-z]+|[^\W\d_]+", re.UNICODE)


def english_score(text: str, sample_chars: int = 40_000) -> Tuple[float, float]:
    """Return ``(ascii_share, stopword_rate)``."""
    sample = text[:sample_chars]
    toks = _WORD.findall(sample)
    if not toks:
        return 0.0, 0.0
    ascii_toks = [t for t in toks if t.isascii()]
    ascii_share = len(ascii_toks) / len(toks)
    if not ascii_toks:
        return ascii_share, 0.0
    hits = sum(1 for t in ascii_toks if t.lower() in _ENGLISH_FUNCTION_WORDS)
    return ascii_share, hits / len(ascii_toks)


ASCII_SHARE_MIN = 0.90
STOPWORD_RATE_MIN = 0.18


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


@dataclass
class Extraction:
    """Result of one PDF -> text attempt.  Mirrors one extract_log.csv row."""

    doc_id: int
    path: str
    year: int
    status: str
    n_pages: int = 0
    n_words: int = 0
    reason: str = ""
    text: str = ""

    def log_row(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id, "path": self.path, "year": self.year,
            "status": self.status, "n_pages": self.n_pages,
            "n_words": self.n_words, "reason": self.reason,
        }


def extract_pdf(fp: Path) -> List[str]:
    """Page texts, in reading order."""
    import pymupdf

    with pymupdf.open(fp) as doc:
        return [page.get_text("text") for page in doc]


def extract_one(task: Tuple[int, int, str]) -> Extraction:
    """Worker entry point: ``(doc_id, year, path)`` -> :class:`Extraction`.

    Top-level (not a closure) because ProcessPoolExecutor has to pickle it.
    """
    doc_id, year, path = task
    fp = pdf_local(year, path)
    if not fp.exists() or fp.stat().st_size == 0:
        return Extraction(doc_id, path, year, "missing_file", reason="no pdf on disk")
    try:
        pages = extract_pdf(fp)
    except Exception as exc:                            # corrupt / encrypted
        return Extraction(doc_id, path, year, "drop_pdf_error",
                          reason=type(exc).__name__)
    n_pages = len(pages)
    if n_pages == 0:
        return Extraction(doc_id, path, year, "drop_pdf_error", reason="zero pages")
    text = clean_pages(pages)
    n_words = word_count(text)
    per_page = n_words / n_pages
    if per_page < config.MIN_WORDS_PER_PAGE:
        return Extraction(doc_id, path, year, "drop_scanned", n_pages, n_words,
                          f"{per_page:.0f} words/page")
    if n_words < config.MIN_WORDS_PER_DOC:
        return Extraction(doc_id, path, year, "drop_short", n_pages, n_words,
                          f"{n_words} words")
    ascii_share, stop_rate = english_score(text)
    if ascii_share < ASCII_SHARE_MIN or stop_rate < STOPWORD_RATE_MIN:
        return Extraction(doc_id, path, year, "drop_nonenglish", n_pages, n_words,
                          f"ascii={ascii_share:.2f} stop={stop_rate:.2f}")
    return Extraction(doc_id, path, year, "ok", n_pages, n_words, text=text)


LOG_FIELDS = ["doc_id", "path", "year", "status", "n_pages", "n_words", "reason"]


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
    w.writeheader()
    return fh, w


def iter_extractions(tasks: Sequence[Tuple[int, int, str]],
                     processes: int | None = None,
                     log_path: Optional[Path] = None,
                     chunksize: int = 8) -> Iterable[Extraction]:
    """Extract in parallel, logging every attempt.  Yields in completion order."""
    processes = processes or config.EXTRACT_PROCESSES
    fh = writer = None
    if log_path is not None:
        fh, writer = _open_log(Path(log_path))
    t0 = time.perf_counter()
    done = 0
    try:
        if processes <= 1:
            results = map(extract_one, tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=processes)
            results = pool.map(extract_one, tasks, chunksize=chunksize)
        for ex in results:
            done += 1
            if writer is not None:
                writer.writerow(ex.log_row())
            if done % 500 == 0:
                dt = time.perf_counter() - t0
                print(f"  extracted {done}/{len(tasks)}  "
                      f"{done / max(dt, 1e-9):.1f} doc/s", file=sys.stderr, flush=True)
            yield ex
    finally:
        if processes > 1:
            pool.shutdown()
        if fh is not None:
            fh.close()


# ==========================================================================
# 4. build -> judgments.jsonl
# ==========================================================================

def _manifest_tasks(man, limit: int | None = None) -> List[Tuple[int, int, str]]:
    rows = [
        (int(r.doc_id), int(r.year), str(r.path))
        for r in man.itertuples(index=False)
    ]
    return rows[:limit] if limit else rows


def build(limit: int | None = None, processes: int | None = None,
          out: Path | None = None) -> Dict[str, Any]:
    """Extract + zone + write ``judgments.jsonl``.

    ``doc_id`` always comes from the manifest, so ids are stable across
    rebuilds and the other members' stores stay joinable.
    """
    from nyaya import zones as zmod

    man = load_manifest()
    meta_by_id = {int(r.doc_id): r._asdict() for r in man.itertuples(index=False)}
    tasks = _manifest_tasks(man, limit)
    out = Path(out or config.JUDGMENTS_JSONL)
    out.parent.mkdir(parents=True, exist_ok=True)

    tally: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    bench_sources: Counter[str] = Counter()
    order_how: Counter[str] = Counter()
    bench_sizes: Counter[int] = Counter()
    n_words = n_paras = 0
    written = 0
    import json

    with out.open("w", encoding="utf-8") as fh:
        for ex in iter_extractions(tasks, processes=processes,
                                   log_path=config.EXTRACT_LOG):
            tally[ex.status] += 1
            if ex.status != "ok":
                continue
            meta = meta_by_id.get(ex.doc_id, {})
            rec, info = zmod.build_record(
                doc_id=ex.doc_id,
                source_path=ex.path,
                year=ex.year,
                text=ex.text,
                meta_title=str(meta.get("title") or ""),
                n_pages=ex.n_pages,
                meta=meta,
            )
            strategies[str(info["para_strategy"])] += 1
            bench_sources[str(info["bench_source"])] += 1
            order_how[str(info["order_strategy"])] += 1
            bench_sizes[rec.bench_size] += 1
            n_words += ex.n_words
            n_paras += len(rec.paras)
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False))
            fh.write("\n")
            written += 1

    summary = {
        "written": written,
        "attempted": len(tasks),
        "status": dict(tally),
        "mean_words": round(n_words / written, 1) if written else 0.0,
        "mean_paras": round(n_paras / written, 1) if written else 0.0,
        "para_strategy": dict(strategies),
        "order_strategy": dict(order_how),
        "bench_source": dict(bench_sources),
        "bench_sizes": dict(sorted(bench_sizes.items())),
        "out": str(out),
    }
    return summary


# ==========================================================================
# 5. stats
# ==========================================================================

def stats() -> Dict[str, Any]:
    from nyaya.utils import read_jsonl

    out: Dict[str, Any] = {}
    if config.EXTRACT_LOG.exists():
        statuses: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        with config.EXTRACT_LOG.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                statuses[row["status"]] += 1
                if row["status"] != "ok":
                    reasons[f"{row['status']}"] += 1
        out["attempted"] = sum(statuses.values())
        out["by_status"] = dict(statuses.most_common())
    if config.JUDGMENTS_JSONL.exists():
        n = 0
        words = paras = 0
        bench: Counter[int] = Counter()
        strat: Counter[str] = Counter()
        years: Counter[int] = Counter()
        zone_empty: Counter[str] = Counter()
        for row in read_jsonl(config.JUDGMENTS_JSONL):
            n += 1
            words += int(row.get("n_tokens") or 0)
            paras += len(row.get("paras") or [])
            bench[int(row.get("bench_size") or 0)] += 1
            labelled = sum(1 for p in row.get("paras", []) if p.get("label"))
            strat["printed" if labelled else "blank-line"] += 1
            years[int(row.get("year") or 0)] += 1
            for z, t in (row.get("zones") or {}).items():
                if not t.strip():
                    zone_empty[z] += 1
        out["documents"] = n
        out["mean_words"] = round(words / n, 1) if n else 0
        out["mean_paras"] = round(paras / n, 1) if n else 0
        out["para_strategy"] = dict(strat)
        out["bench_size_distribution"] = dict(sorted(bench.items()))
        out["empty_zone_counts"] = dict(zone_empty)
        out["years_covered"] = len(years)
    return out


# ==========================================================================
# 6. CLI
# ==========================================================================

def _cmd_download(args) -> int:
    man = load_manifest()
    rows = [(int(r.year), str(r.path)) for r in man.itertuples(index=False)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest: {len(man)} judgments; downloading {len(rows)} "
          f"with {args.threads} threads", file=sys.stderr)
    with timed("download"):
        tally = download(rows, threads=args.threads)
    print(tally, file=sys.stderr)
    on_disk = sum(1 for y, p in rows if pdf_local(y, p).exists())
    print(f"on disk: {on_disk}/{len(rows)}", file=sys.stderr)
    return 0


def _cmd_extract(args) -> int:
    """Extraction only: fills extract_log.csv so coverage can be reported."""
    man = load_manifest()
    tasks = _manifest_tasks(man, args.limit)
    tally: Counter[str] = Counter()
    words = pages = 0
    with timed("extract"):
        for ex in iter_extractions(tasks, processes=args.processes,
                                   log_path=config.EXTRACT_LOG):
            tally[ex.status] += 1
            if ex.status == "ok":
                words += ex.n_words
                pages += ex.n_pages
    ok = tally["ok"]
    print(f"attempted={sum(tally.values())} " + " ".join(
        f"{k}={v}" for k, v in tally.most_common()), file=sys.stderr)
    if ok:
        print(f"mean words/doc={words / ok:.0f}  mean pages/doc={pages / ok:.1f}",
              file=sys.stderr)
    print(f"log: {config.EXTRACT_LOG}", file=sys.stderr)
    return 0


def _cmd_build(args) -> int:
    with timed("build"):
        summary = build(limit=args.limit, processes=args.processes)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


def _cmd_stats(args) -> int:
    for k, v in stats().items():
        print(f"{k}: {v}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.ingest",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch the manifest's PDFs from the bucket")
    d.add_argument("--all", action="store_true", help="whole manifest (the default)")
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--threads", type=int, default=config.DOWNLOAD_THREADS)
    d.set_defaults(func=_cmd_download)

    e = sub.add_parser("extract", help="PDF -> cleaned text, writes extract_log.csv")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--processes", type=int, default=config.EXTRACT_PROCESSES)
    e.set_defaults(func=_cmd_extract)

    b = sub.add_parser("build", help="extract + zone -> judgments.jsonl")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--processes", type=int, default=config.EXTRACT_PROCESSES)
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("stats", help="real numbers for the README")
    s.set_defaults(func=_cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
