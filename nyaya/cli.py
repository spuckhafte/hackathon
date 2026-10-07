"""`python -m nyaya.cli` — the command line the demo video records.

Every flag here exists because something has to be *shown*, not asserted:

===============================  =========================================
``--mode S4 "<query>"``          ranked results with year, bench and badge
``--explain-plan '<query>'``     the AST, then the df-ordered plan with the
                                 candidate count after every step
``--why N '<query>'``            the full "why this result" panel for hit N,
                                 with the arithmetic that produces the score
``--analyze "<text>"``           the token stream: term, zone, pos, sent, para
``--parse '<query>'``            the parse tree, filters and expansions
``--postings <term>``            a postings list: doc ids, per-zone tf,
                                 positions, skip stride
``--merge '<query>' --trace``    the pointer steps of the merge
``--treatment-audit``            classifier errors beside their gold labels
``--smoke``                      the end-to-end check CONTRACTS.md requires
===============================  =========================================

`nyaya.utils.ensure_utf8_stdout()` is called first, because this prints
``ipc§302`` and ``¶`` and a Windows console is cp1252 by default.

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from nyaya import config
from nyaya.contracts import MODES
from nyaya.utils import ensure_utf8_stdout

BAR = "=" * 78
DASH = "-" * 78


# ==========================================================================
# search
# ==========================================================================


def cmd_search(args) -> int:
    from nyaya.search import SearchReport, search, snippets_for

    rep = SearchReport()
    results = search(args.query, k=args.limit, mode=args.mode,
                     explain=args.why is not None or args.verbose, report=rep)

    if args.json:
        print(json.dumps({
            "query": args.query, "mode": args.mode,
            "report": rep.as_json(),
            "results": [_result_json(r) for r in results],
        }, indent=1, default=str))
        return 0

    print(f"\n{BAR}\n{args.query}\n{BAR}")
    print(f"mode {args.mode}  ·  {rep.path}  ·  {rep.candidates:,} candidates "
          f"·  {rep.scored:,} scored  ·  {rep.elapsed_ms:.0f} ms  "
          f"·  scorer {rep.scorer}")
    tier = _tier_line(rep)
    if tier:
        print(tier)
    for w in rep.warnings:
        print(f"  ! {w}")
    if not results:
        print("\nno results")
        return 0

    from nyaya.search import load_context

    ctx = load_context(args.mode)
    terms = _terms_of(args.query)
    print()
    for i, r in enumerate(results, 1):
        bench = f"bench {r.bench_size}" if r.bench_size else "bench ?"
        print(f"{i:>3}. [{r.score:.4f}] {r.title[:66]}")
        line = f"      {r.year}  ·  {bench}"
        if r.neutral_cite:
            line += f"  ·  {r.neutral_cite}"
        if r.scr_cite:
            line += f"  ·  {r.scr_cite}"
        if r.disposal:
            line += f"  ·  {r.disposal}"
        print(line)
        if r.badge:
            print(f"      !! {r.badge[:150]}")
        if args.snippets:
            for s in snippets_for(r, terms, ctx, limit=1):
                print(f"      {s[:190]}")
    print()
    return 0


def _tier_line(rep) -> str:
    """One line saying which tier answered — S5's contribution, made visible.

    Printed only when a tier decision was actually taken, i.e. the free-text path
    with S5 loaded.  A ``None`` trace means "no tier decision was taken", not
    "tier 2", so the line is omitted rather than claiming something untrue about
    a Boolean or filters-only query.
    """
    t = getattr(rep, "tier_trace", None)
    if t is None:
        return ""
    body = f"S5 tier {t.tier}  -  {t.reason}"
    if t.terms_without_champions:
        body += ("  -  no champion list for "
                 + ", ".join(t.terms_without_champions[:4])
                 + " (df below the champion threshold, so their full postings "
                   "were used)")
    return "  " + body


def _result_json(r) -> Dict[str, Any]:
    from dataclasses import asdict

    d = asdict(r)
    return d


def _terms_of(query: str) -> List[str]:
    from nyaya.query.parser import parse

    return parse(query).free_terms


# ==========================================================================
# --why
# ==========================================================================


def cmd_why(args) -> int:
    from nyaya.rank.explain import render, verify
    from nyaya.search import SearchReport, load_context, search, snippets_for

    rep = SearchReport()
    k = max(args.limit, args.why)
    results = search(args.query, k=k, mode=args.mode, explain=True, report=rep)
    if not results:
        print("no results", file=sys.stderr)
        return 1
    n = min(max(1, args.why), len(results))
    r = results[n - 1]

    print(f"\n{BAR}\nquery: {args.query}\nmode:  {args.mode}   "
          f"(rank {n} of {len(results)})\n{BAR}")
    print(f"{r.title}")
    print(f"{r.year}  ·  bench {r.bench_size}  ·  {r.neutral_cite or r.scr_cite}")
    if r.badge:
        print(f"\n  TREATMENT: {r.badge}")
        ctx = load_context(args.mode)
        gs = getattr(ctx, "gscore", None)
        if gs is not None:
            for ev in gs.evidence(r.doc_id, limit=2):
                print(f"    evidence: {ev[:200]}")
            if gs.gloss(r.doc_id):
                print(f"    meaning:  {gs.gloss(r.doc_id)}")
    print()
    if r.explanation is None:
        print("no explanation available", file=sys.stderr)
        return 1
    tier = _tier_line(rep)
    if tier:
        print("candidate selection")
        print(tier)
        print()
    print(render(r.explanation, doc_id=r.doc_id))
    v = verify(r.explanation)
    print(f"\n  arithmetic check: {'the parts reproduce the net score' if v['ok'] else 'MISMATCH'}"
          f"  ({v['sum']:.6f} vs {v['net']:.6f})")

    ctx = load_context(args.mode)
    snips = snippets_for(r, _terms_of(args.query), ctx, limit=2)
    if snips:
        print("\n  evidence passages (S1, cited honestly)")
        for s in snips:
            print(f"    {s[:300]}")
    print()
    return 0


# ==========================================================================
# --explain-plan  /  --merge --trace
# ==========================================================================


def cmd_plan(args, trace: bool = False) -> int:
    import numpy as np

    from nyaya.query.parser import describe_tree, parse
    from nyaya.query.planner import plan_query
    from nyaya.search import load_context

    q = parse(args.query)
    ctx = load_context(args.mode)

    print(f"\n{BAR}\n{args.query}\n{BAR}")
    print("\nparsed query")
    print(DASH)
    print(describe_tree(q.root,
                        "(no term tree - this query is filters only)"
                        if q.filters and not q.root else None))
    if q.filters:
        print("\nfilters")
        for f in q.filters:
            print(f"  {f.describe()}")
    if q.expansions:
        print("\nexpansions (query time only)")
        for t, alts in q.expansions.items():
            print(f"  {t} OR {' OR '.join(alts)}")
    for w in q.warnings:
        print(f"  ! {w}")

    t0 = time.perf_counter()
    docs, plan, stats = plan_query(q, ctx)
    ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nexecution plan   (rarest first; each row shows what survived)")
    print(DASH)
    _print_plan(plan, trace=trace)
    print(DASH)
    print(f"  result: {np.asarray(docs).size:,} documents in {ms:.1f} ms")
    if stats is not None:
        s = stats.as_json()
        for key in ("postings_lists_read", "postings_entries_read",
                    "merge_entries", "skip_probes", "merges",
                    "filters_applied", "terms_dropped_by_idf"):
            if key in s and s[key] not in (0, [], None):
                print(f"  {key:<26}{s[key]}")
    if trace:
        _trace_merge(q, docs, ctx)
    print()
    return 0


def _trace_merge(q, docs, ctx) -> int:
    """The actual positions the merge compared, in the first surviving document.

    `--trace` is meant to show the *pointer steps*, and these are them: the
    postings positions of each side, the pairs that satisfied the constraint and
    the smallest gap found.  Everything here is read back out of the index, not
    reconstructed.
    """
    import numpy as np

    from nyaya.contracts import PhraseNode, ProxNode
    from nyaya.query import proximity
    from nyaya.query.parser import _leaf_terms

    ids = np.asarray(docs)
    if ids.size == 0:
        print("\n  nothing survived the merge, so there is no pointer walk to show")
        return 0
    node = _first_positional(q.root)
    if node is None:
        print("\n  this query has no phrase or proximity node, so the merge is "
              "a plain sorted-array intersection; the counts above are the walk")
        return 0

    doc_id = int(ids[0])
    print(f"\npointer walk inside doc {doc_id}")
    print(DASH)
    if isinstance(node, ProxNode):
        left, right = _leaf_terms(node.left), _leaf_terms(node.right)
        for side, terms in (("left ", left), ("right", right)):
            pos = _positions(ctx, terms, doc_id)
            print(f"  {side} {str(terms):<34} positions "
                  f"{', '.join(str(p) for p in pos[:14])}"
                  f"{' ...' if len(pos) > 14 else ''}")
        ex = proximity.explain_proximity(left, right, doc_id, node.kind,
                                         node.k, ctx)
        if ex is None:
            print("  (no pair satisfied the constraint in this document)")
            return 0
        kind = {"s": "same sentence", "p": "same paragraph"}.get(
            ex.kind, f"within {ex.k} words")
        print(f"\n  constraint      {kind}")
        print(f"  pairs accepted  {ex.n_pairs}")
        print(f"  smallest gap    {ex.best_gap} tokens")
        print(f"  first pairs     " +
              ", ".join(f"({a}->{b}, gap {abs(b - a)})" for a, b in ex.sample))
    else:
        terms = list(node.terms)
        for t in terms:
            pos = _positions(ctx, [t], doc_id)
            print(f"  {t:<22} positions {', '.join(str(p) for p in pos[:14])}"
                  f"{' ...' if len(pos) > 14 else ''}")
        hits = proximity.phrase_occurrences(terms, doc_id, ctx)
        print(f"\n  phrase starts   {', '.join(str(int(h)) for h in hits[:14])}")
        print(f"  occurrences     {len(hits)}  "
              f"(a start survives only if every term sits at start + i)")
    return 0


def _first_positional(node):
    from nyaya.contracts import PhraseNode, ProxNode

    if node is None:
        return None
    if isinstance(node, (ProxNode, PhraseNode)):
        return node
    for c in node.children():
        got = _first_positional(c)
        if got is not None:
            return got
    return None


def _positions(ctx, terms, doc_id):
    import numpy as np

    out = []
    for t in terms:
        p = ctx.index.postings(t)
        if p is not None:
            out.extend(int(x) for x in np.asarray(p.positions_of(doc_id)))
    return sorted(out)


def _print_plan(step, depth: int = 0, trace: bool = False) -> None:
    if step is None:
        print("  (no plan - free text goes straight to ranked retrieval)")
        return
    pad = "  " + "   " * depth
    est = getattr(step, "estimated", -1)
    produced = getattr(step, "produced", -1)
    running = getattr(step, "running", -1)
    label = getattr(step, "label", None) or getattr(step, "op", "step")
    bits = [f"{pad}{label}"]
    if est >= 0:
        bits.append(f"est {est:,}")
    if produced >= 0:
        bits.append(f"-> {produced:,}")
    if running >= 0:
        bits.append(f"running {running:,}")
    note = getattr(step, "note", "")
    if note:
        bits.append(f"({note})")
    print("   ".join(bits))
    for child in getattr(step, "children", []) or []:
        _print_plan(child, depth + 1, trace)


# ==========================================================================
# --analyze / --parse / --postings
# ==========================================================================


def cmd_analyze(args) -> int:
    from nyaya.analyze import analyze, canonicalise_statutes, clean_text
    from nyaya.statutes import iter_statute_spans, pretty
    from nyaya.stoplist import is_scoring_stopword

    text = args.analyze
    print(f"\n{BAR}\nanalyze: {text}\n{BAR}")
    cleaned = clean_text(text)
    canon = canonicalise_statutes(cleaned)
    print(f"\ncleaned       : {cleaned}")
    print(f"canonicalised : {canon}")
    spans = iter_statute_spans(cleaned)
    if spans:
        print("\nstatute spans (pass 1 runs on raw text, before tokenising)")
        for s in spans:
            print(f"  [{s.start:>3}:{s.end:<3}] {s.shape:<9} "
                  f"inherited={str(s.inherited):<5} {s.surface!r} -> "
                  f"{' '.join(s.tokens)}")
            for t in s.tokens:
                print(f"       {t:<12} = {pretty(t)}")

    toks = analyze(text, zone=args.zone, para_id=args.para)
    print(f"\ntokens   ({len(toks)} in the positional index, "
          f"{sum(1 for t in toks if not is_scoring_stopword(t.term))} in the "
          f"tf-idf vector)")
    print(DASH)
    print(f"  {'':2}{'pos':>5} {'sent':>5} {'para':>5}  {'zone':<9}"
          f"{'term':<18}surface")
    for t in toks:
        mark = "*" if is_scoring_stopword(t.term) else " "
        print(f"  {mark} {t.pos:>5} {t.sent_id:>5} {t.para_id:>5}  "
              f"{t.zone:<9}{t.term:<18}{t.raw!r}")
    print("\n  * = kept in the positional index, excluded from tf-idf / BM25F")
    print()
    return 0


def cmd_parse(args) -> int:
    from nyaya.query.parser import explain

    print()
    print(explain(args.parse))
    print()
    return 0


def cmd_postings(args) -> int:
    import numpy as np

    from nyaya.analyze import analyze_terms
    from nyaya.search import load_context

    ctx = load_context(args.mode)
    index = ctx.index
    raw = args.postings
    terms = analyze_terms(raw) or [raw]
    term = terms[0]

    print(f"\n{BAR}\npostings: {raw!r} -> analysed to {term!r}\n{BAR}")
    posts = index.postings(term)
    if posts is None:
        near = index.prefix_terms(term[:4])[:12]
        print(f"\nnot in the dictionary. nearby terms: {', '.join(near)}")
        return 1
    df = index.df(term)
    print(f"\n  df            {df:,} documents of {index.doc_count:,}")
    print(f"  idf           {index.idf(term):.4f}")
    print(f"  positions     {int(np.asarray(posts._positions).size):,}")
    print(f"  skip stride   {posts.skip_stride()}  "
          f"(~sqrt(df); 0 means the list is too short to be worth skipping)")
    print(f"\n  {'doc':>7}  {'tf':>4}  per-zone tf{'':>18}positions")
    print("  " + DASH)
    ids = np.asarray(posts.doc_ids)
    for d in ids[: args.limit].tolist():
        ztf = {k: v for k, v in posts.zone_tf_of(d).items() if v}
        pos = np.asarray(posts.positions_of(d))
        shown = ", ".join(str(int(p)) for p in pos[:8])
        more = f" ... (+{pos.size - 8})" if pos.size > 8 else ""
        print(f"  {d:>7}  {int(pos.size):>4}  {str(ztf):<29}{shown}{more}")
    if ids.size > args.limit:
        print(f"  ... and {ids.size - args.limit:,} more documents")
    print()
    return 0


# ==========================================================================
# --treatment-audit
# ==========================================================================


def cmd_treatment_audit(args) -> int:
    """Delegate to M1's classifier — this flag owns no logic of its own.

    Tried in order, because the classifier's own CLI is still growing:
      1. ``nyaya.eval.treatment`` — the gold-label audit, when it lands;
      2. ``nyaya.authority.treatment audit`` — same, if it grows the subcommand;
      3. ``nyaya.authority.treatment sample`` — which exists today and prints
         labelled edges with the evidence sentence behind each label.
    Option 3 shows the predicted label and the sentence but **not** the gold
    label, so it says so rather than implying a comparison it has not made.
    """
    try:
        from nyaya.eval import treatment as audit            # type: ignore

        if hasattr(audit, "main"):
            argv = ["--only-errors"] if args.only_errors else []
            return int(audit.main(argv + ["--limit", str(args.limit)]))
    except ImportError:
        pass

    try:
        from nyaya.authority import treatment as tre
    except ImportError:
        print("\nthe treatment classifier is not built yet", file=sys.stderr)
        return 2

    # Probe for the subcommand rather than calling and catching argparse's
    # SystemExit: argparse prints its usage error to stderr on the way out, and
    # a demo recording should not show a failed command it then recovers from.
    if hasattr(tre, "_cmd_audit"):
        argv = ["audit", "--n", str(args.limit)]
        if args.only_errors:
            argv.append("--only-errors")
        return int(tre.main(argv))

    print(f"\n{BAR}\ntreatment classifier — labelled edges with their evidence\n"
          f"{BAR}")
    print("The gold-label comparison lives in M1's evaluation harness and is "
          "not wired here yet, so what follows is the predicted label and the "
          "sentence it was predicted from, with no gold column.\n",
          file=sys.stderr)
    rc = 0
    for label in (("distinguished", "overruled") if args.only_errors
                  else ("followed", "distinguished", "overruled")):
        try:
            rc |= int(tre.main(["sample", "--label", label,
                                "--n", str(args.limit)]))
        except SystemExit as exc:                            # pragma: no cover
            rc |= int(exc.code or 0)
    return rc


# ==========================================================================
# --brief
# ==========================================================================


def cmd_brief(args) -> int:
    from nyaya.brief import build, render

    b = build(args.query, k=config.BRIEF_TOP_DOCS, mode=args.mode)
    if args.json:
        print(json.dumps(b.as_json(), indent=1, default=str))
        return 0
    print()
    print(render(b))
    print()
    return 0 if b.text else 1


# ==========================================================================
# --smoke
# ==========================================================================


SMOKE_QUERIES = [
    ("free text", "right to privacy of personal data"),
    ("statute", "S. 302 IPC dying declaration"),
    ("phrase + prox", 'privacy /p "fundamental right"'),
    ("boolean + filter", 'bail! AND act:IPC year:>=2010'),
    ("filters only", 'act:NDPS type:"criminal appeal"'),
    ("malformed", 'unbalanced "quote AND stray ) /x'),
]


def cmd_smoke(args) -> int:
    """The end-to-end check `docs/CONTRACTS.md` section 5.4 requires."""
    from nyaya.search import MODE_SETTINGS, SearchReport, search

    ok = True
    print(f"\n{BAR}\nNyayaTrace smoke test\n{BAR}")
    for label, q in SMOKE_QUERIES:
        for mode in MODES:
            rep = SearchReport()
            try:
                res = search(q, k=3, mode=mode, explain=(mode == "S4"),
                             report=rep)
            except Exception as exc:
                print(f"  FAIL  {mode:<4} {label:<18} {exc.__class__.__name__}: {exc}")
                ok = False
                continue
            flag = "ok  "
            if mode == "S4" and res and res[0].explanation is not None:
                from nyaya.rank.explain import verify

                if not verify(res[0].explanation)["ok"]:
                    flag, ok = "ARITH", False
            print(f"  {flag:<5} {mode:<4} {label:<18} {len(res)} hits  "
                  f"{rep.candidates:>7,} cand  {rep.elapsed_ms:>6.0f} ms")
    print(DASH)
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


# ==========================================================================
# argparse
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m nyaya.cli",
        description="NyayaTrace: vertical search over Supreme Court of India "
                    "judgments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples
  python -m nyaya.cli --mode S4 "right to privacy of personal data"
  python -m nyaya.cli --why 1 --mode S4 'privacy /p "fundamental right"'
  python -m nyaya.cli --explain-plan 'privacy /p "fundamental right" AND bench:>=5'
  python -m nyaya.cli --analyze "convicted u/s 302 IPC read with S. 34"
  python -m nyaya.cli --parse 'bail! AND act:IPC year:>=2010'
  python -m nyaya.cli --postings privacy --limit 12
  python -m nyaya.cli --merge 'privacy /p "fundamental right"' --trace
  python -m nyaya.cli --brief "dying declaration as the sole basis of conviction"
  python -m nyaya.cli --smoke
""")
    ap.add_argument("query", nargs="*", help="the query to run")
    ap.add_argument("--mode", default="S4", choices=list(MODES),
                    help="which system to run (default S4, the full one)")
    ap.add_argument("-k", "--limit", type=int, default=config.TOP_K,
                    help="results, postings rows or audit rows to show")
    ap.add_argument("--why", type=int, metavar="N",
                    help="print the full score explanation for result N")
    ap.add_argument("--explain-plan", action="store_true",
                    help="print the AST and the df-ordered execution plan")
    ap.add_argument("--merge", metavar="QUERY",
                    help="run the merge for QUERY and show the plan")
    ap.add_argument("--trace", action="store_true",
                    help="with --merge: show each pointer step")
    ap.add_argument("--analyze", metavar="TEXT",
                    help="run analyze() over TEXT and print the token stream")
    ap.add_argument("--zone", default="body", choices=list(config.ZONES))
    ap.add_argument("--para", type=int, default=0)
    ap.add_argument("--parse", metavar="QUERY",
                    help="parse QUERY and print the tree, filters and expansions")
    ap.add_argument("--postings", metavar="TERM",
                    help="print one term's postings list")
    ap.add_argument("--treatment-audit", action="store_true",
                    help="citation-treatment classifier errors vs gold labels")
    ap.add_argument("--only-errors", action="store_true",
                    help="with --treatment-audit: only the misclassifications")
    ap.add_argument("--brief", action="store_true",
                    help="Precedent Brief: evidence pack, summary, grounding check")
    ap.add_argument("--smoke", action="store_true",
                    help="run every mode over every query shape")
    ap.add_argument("--snippets", action="store_true",
                    help="show an evidence passage under each result")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = build_parser()
    args = ap.parse_args(argv)
    args.query = " ".join(args.query).strip()

    try:
        if args.brief:
            if not args.query:
                ap.error("--brief needs a query")
            return cmd_brief(args)
        if args.smoke:
            return cmd_smoke(args)
        if args.analyze:
            return cmd_analyze(args)
        if args.parse:
            return cmd_parse(args)
        if args.postings:
            return cmd_postings(args)
        if args.treatment_audit:
            return cmd_treatment_audit(args)
        if args.merge:
            args.query = args.merge
            return cmd_plan(args, trace=args.trace)
        if args.explain_plan:
            if not args.query:
                ap.error("--explain-plan needs a query")
            return cmd_plan(args, trace=args.trace)
        if args.why is not None:
            if not args.query:
                ap.error("--why needs a query")
            return cmd_why(args)
        if args.query:
            return cmd_search(args)
        ap.print_help()
        return 0
    except KeyboardInterrupt:                                # pragma: no cover
        return 130
    except FileNotFoundError as exc:
        print(f"\n{exc}\n\nThe index is not built. Run:\n"
              "  python -m nyaya.ingest build\n"
              "  python -m nyaya.index.positional --build\n"
              "  python -m nyaya.index.paragraphs --build\n", file=sys.stderr)
        return 2


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
