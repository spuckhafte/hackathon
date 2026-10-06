"""Hand-checkable audit of the zone segmenter  (Member 1).

Prints, for a stratified sample of real judgments, exactly what
``nyaya.zones.segment`` detected: title, parties, bench (and its size), case
number, case type, citations, the paragraph-numbering strategy and the first
line of the ``order`` zone.  Reading ~30 of these against the PDFs is how the
heuristics in ``zones.py`` were measured; the accuracy numbers in the README
come from this script.

    python scripts/zone_audit.py --per-year 3 --years 1950 1975 1990 2005 2017 2024
    python scripts/zone_audit.py --n 30 --csv runs/zone_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyaya import config, ingest, zones          # noqa: E402
from nyaya.utils import ensure_utf8_stdout       # noqa: E402


def sample_rows(per_year: int, years):
    import pandas as pd

    man = pd.read_parquet(config.MANIFEST_PARQUET)
    if years:
        man = man[man["year"].isin([int(y) for y in years])]
    out = []
    for _, grp in man.groupby("year", sort=True):
        out.extend(grp.head(per_year).to_dict("records"))
    return out


def main(argv=None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-year", type=int, default=3)
    ap.add_argument("--years", nargs="*", default=[1950, 1965, 1975, 1990, 2005, 2017, 2024])
    ap.add_argument("--n", type=int, default=None, help="stop after N documents")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--quiet", action="store_true", help="only the summary table")
    args = ap.parse_args(argv)

    rows = sample_rows(args.per_year, args.years)
    if args.n:
        rows = rows[: args.n]

    audit = []
    for r in rows:
        ex = ingest.extract_one((int(r["doc_id"]), int(r["year"]), str(r["path"])))
        if ex.status != "ok":
            print(f"--- {r['path']}  SKIPPED ({ex.status}: {ex.reason})")
            continue
        rec, info = zones.build_record(
            doc_id=int(r["doc_id"]), source_path=str(r["path"]), year=int(r["year"]),
            text=ex.text, meta_title=str(r.get("title") or ""), n_pages=ex.n_pages,
        )
        seg = zones.segment(ex.text, meta_title=str(r.get("title") or ""))
        order_head = " ".join(seg.zones["order"].split())[:160]
        audit.append({
            "doc_id": rec.doc_id, "year": rec.year, "path": rec.source_path,
            "title": rec.title[:70], "petitioner": rec.petitioner[:40],
            "respondent": rec.respondent[:40],
            "bench": " ".join(seg.header.bench_text.split())[:90],
            "bench_size_text": len(seg.header.judges),
            "bench_size_meta": int(r.get("bench_size_meta") or 0),
            "bench_source": seg.header.bench_source,
            "case_no": rec.case_no[:50], "case_type": rec.case_type,
            "scr_cite": rec.scr_cite, "neutral_cite": rec.neutral_cite,
            "para_strategy": info["para_strategy"], "n_paras": len(rec.paras),
            "order_strategy": info["order_strategy"],
            "n_order_paras": info["n_order_paras"],
            "order_head": order_head,
            "n_words": ex.n_words, "n_pages": ex.n_pages,
        })
        if args.quiet:
            continue
        print("=" * 100)
        print(f"[{rec.year}] doc_id={rec.doc_id}  {rec.source_path}  "
              f"{ex.n_pages}p {ex.n_words}w")
        print(f"  title      : {rec.title[:92]}")
        print(f"  parties    : {rec.petitioner[:44]!r}  V  {rec.respondent[:44]!r}")
        print(f"  bench      : {' '.join(seg.header.bench_text.split())[:92]}")
        print(f"  judges({len(seg.header.judges)}) : {seg.header.judges}"
              f"   [src={seg.header.bench_source}, meta says "
              f"{r.get('bench_size_meta')}]")
        print(f"  case_no    : {rec.case_no!r}   type={rec.case_type!r}")
        print(f"  cites      : scr={rec.scr_cite!r}  neutral={rec.neutral_cite!r}"
              f"  date={rec.date!r}")
        print(f"  paras      : {len(rec.paras)} via {info['para_strategy']} "
              f"({info['n_printed_labels']} printed labels)")
        print(f"  order({info['n_order_paras']}p via {info['order_strategy']}): {order_head}")

    # ---- summary -------------------------------------------------------
    if audit:
        n = len(audit)
        print("\n" + "=" * 100)
        print(f"SUMMARY over {n} documents")
        from collections import Counter
        for key in ("para_strategy", "order_strategy", "bench_source", "case_type"):
            print(f"  {key:15s}: {dict(Counter(a[key] for a in audit).most_common())}")
        got = lambda k: sum(1 for a in audit if str(a[k]).strip())   # noqa: E731
        for key in ("title", "petitioner", "respondent", "bench", "case_no",
                    "scr_cite", "neutral_cite"):
            print(f"  non-empty {key:13s}: {got(key)}/{n} = {got(key) / n:.0%}")
        agree = sum(1 for a in audit
                    if a["bench_size_text"] == a["bench_size_meta"])
        bigger = sum(1 for a in audit
                     if a["bench_size_text"] > a["bench_size_meta"])
        print(f"  bench size: text==meta {agree}/{n}, text>meta {bigger}/{n}")
        print(f"  mean paras {sum(a['n_paras'] for a in audit) / n:.1f}, "
              f"mean words {sum(a['n_words'] for a in audit) / n:.0f}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(audit[0].keys()))
            w.writeheader()
            w.writerows(audit)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
