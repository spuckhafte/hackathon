"""The Westlaw-style query language: text in, `contracts.ParsedQuery` out.

Grammar (loosest binding first, parentheses override)
-----------------------------------------------------
::

    query     := or_expr
    or_expr   := and_expr   ( ('OR')          and_expr )*
    and_expr  := not_expr   ( ('AND' | '&' | <adjacency>) not_expr )*
    not_expr  := prox_expr  ( ('NOT' | '%')   prox_expr )*
    prox_expr := atom       ( ('/n'|'w/n'|'/s'|'/p') atom )*
    atom      := '(' or_expr ')'
               | '"' phrase '"'
               | field ':' ( '(' or_expr ')' | value )
               | word [ '!' ]

So ``a OR b AND c`` is ``a OR (b AND c)``, and ``a /3 b NOT c AND d`` is
``((a /3 b) NOT c) AND d`` — proximity binds tightest because it describes a
relationship between two *terms*, and OR loosest because it is the widest net.

Why a hand-written recursive-descent parser
-------------------------------------------
Because the error messages are the feature.  A search box that answers an
unbalanced quote with a traceback is unusable, so **nothing in here raises at
the public boundary**: every malformed query degrades to a ranked free-text
search over the same words, with a warning the UI can show.  That is also the
honest behaviour — the user asked a question, and "did you mean…" beats nothing.

Free text is the fast path
--------------------------
A query with no operators, no filters, no quotes and no zone restriction sets
``is_free_text=True``, leaves ``root=None`` and fills ``free_terms``.  The
planner then skips the Boolean machinery entirely and goes straight to ranked
retrieval, which is what the overwhelming majority of real queries need.

Demo:  ``python -m nyaya.query.parser --explain 'title:(privacy) AND
(dignity OR autonomy) /s surveillance NOT bail! year:>=2015 bench:>=5'``

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

from nyaya.contracts import (
    AndNode,
    FilterNode,
    Node,
    NotNode,
    OrNode,
    ParsedQuery,
    PhraseNode,
    ProxNode,
    TermNode,
)
from nyaya.query import normalise as N

__all__ = ["parse", "Query", "describe_tree", "explain"]

MAX_PROX = 50           # /100 is not a proximity constraint, it is the document
MIN_PREFIX = 3          # bail! is fine; b! would scan the dictionary


# ==========================================================================
# Result type
# ==========================================================================


@dataclass
class Query(ParsedQuery):
    """`ParsedQuery` plus the things a *user interface* needs.

    A subclass, not a change to `contracts.ParsedQuery`: M2's planner takes a
    `ParsedQuery` and this is one, so nothing downstream notices.  The extra
    fields exist because the contract has nowhere to put a warning, and swallowing
    warnings is how a search box becomes mysterious.
    """

    warnings: List[str] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)
    degraded: bool = False          # a parse error forced the free-text fallback
    normalised: str = ""            # the query after statute canonicalisation

    @property
    def scoring_terms(self) -> List[str]:
        """Free terms minus the scoring stop words (see `nyaya.stoplist`)."""
        return N.scoring_terms(self.free_terms)

    def tree(self) -> str:
        return describe_tree(self.root)


# ==========================================================================
# Lexer
# ==========================================================================
_PROX_RE = re.compile(r"^(?:w/|/)(\d{1,3}|s|p)$", re.I)
_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z_]{1,14}):(.*)$", re.S)
#: A word: letters/digits, the canonical statute token, apostrophes, hyphens,
#: and a trailing ! or * for root expansion.
_WORD_CHARS = re.compile(r"[^\s()\"]+")


class _Tok:
    LP, RP, PHRASE, PROX, AND, OR, NOT, WORD, FIELD = (
        "LP", "RP", "PHRASE", "PROX", "AND", "OR", "NOT", "WORD", "FIELD")

    __slots__ = ("kind", "text", "extra", "pos")

    def __init__(self, kind: str, text: str, pos: int, extra=None):
        self.kind, self.text, self.pos, self.extra = kind, text, pos, extra

    def __repr__(self) -> str:                               # pragma: no cover
        return f"{self.kind}({self.text!r})"


class _ParseError(Exception):
    """Internal only; `parse()` converts it into a warning plus a fallback."""


def _lex(text: str, warnings: List[str]) -> List[_Tok]:
    toks: List[_Tok] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            toks.append(_Tok(_Tok.LP, "(", i))
            i += 1
            continue
        if ch == ")":
            toks.append(_Tok(_Tok.RP, ")", i))
            i += 1
            continue
        if ch == '"':
            j = text.find('"', i + 1)
            if j < 0:
                warnings.append(
                    "unbalanced quote: treating the rest of the query as a phrase")
                toks.append(_Tok(_Tok.PHRASE, text[i + 1:].strip(), i))
                break
            toks.append(_Tok(_Tok.PHRASE, text[i + 1:j], i))
            i = j + 1
            continue
        m = _WORD_CHARS.match(text, i)
        raw = m.group(0)
        i = m.end()
        toks.append(_classify(raw, m.start(), warnings))
    return [t for t in toks if t is not None]


def _classify(raw: str, pos: int, warnings: List[str]) -> Optional[_Tok]:
    low = raw.lower()
    if low in ("and", "&", "&&"):
        return _Tok(_Tok.AND, raw, pos)
    if low in ("or", "||"):
        return _Tok(_Tok.OR, raw, pos)
    if low in ("not", "%", "andnot", "and-not"):
        return _Tok(_Tok.NOT, raw, pos)

    m = _PROX_RE.match(raw)
    if m:
        arg = m.group(1).lower()
        if arg == "s":
            return _Tok(_Tok.PROX, raw, pos, ("s", 0))
        if arg == "p":
            return _Tok(_Tok.PROX, raw, pos, ("p", 0))
        k = int(arg)
        if k < 1:
            warnings.append(f"{raw}: a proximity window must be at least 1 word; "
                            "using /1")
            k = 1
        elif k > MAX_PROX:
            warnings.append(f"{raw}: a window of {k} words is wider than a "
                            f"paragraph; using /{MAX_PROX}")
            k = MAX_PROX
        return _Tok(_Tok.PROX, raw, pos, ("n", k))
    if raw.startswith("/") or raw.lower().startswith("w/"):
        warnings.append(f"{raw!r} is not a proximity operator "
                        "(use /5, /s or /p); ignoring it")
        return None

    fm = _FIELD_RE.match(raw)
    if fm and (N.is_filter_field(fm.group(1)) or N.is_zone_field(fm.group(1))):
        return _Tok(_Tok.FIELD, raw, pos, (N.canonical_field(fm.group(1)), fm.group(2)))
    if fm and fm.group(2):
        warnings.append(
            f"unknown field {fm.group(1)!r}: searching for it as ordinary words. "
            f"Fields are {', '.join(N.FILTER_FIELDS)} and the zones "
            f"{', '.join(N.ZONE_FIELDS)}")
        return _Tok(_Tok.WORD, raw.replace(":", " "), pos)
    return _Tok(_Tok.WORD, raw, pos)


# ==========================================================================
# Parser
# ==========================================================================


class _Parser:
    def __init__(self, toks: Sequence[_Tok], warnings: List[str]):
        self.toks = list(toks)
        self.i = 0
        self.warnings = warnings
        self.saw_operator = False
        self.filters: List[FilterNode] = []       # hoisted top-level conjuncts
        self._depth = 0                           # >0 means "inside parentheses"

    # ---- plumbing -------------------------------------------------------
    def peek(self) -> Optional[_Tok]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Optional[_Tok]:
        t = self.peek()
        if t is not None:
            self.i += 1
        return t

    def at(self, *kinds: str) -> bool:
        t = self.peek()
        return t is not None and t.kind in kinds

    # ---- grammar --------------------------------------------------------
    def parse(self) -> Optional[Node]:
        node = self.or_expr()
        if self.peek() is not None:
            stray = self.next()
            if stray.kind == _Tok.RP:
                self.warnings.append("unbalanced ')': ignoring it")
            else:                                            # pragma: no cover
                self.warnings.append(f"ignoring trailing {stray.text!r}")
            rest = self.or_expr()
            node = _and([node, rest]) if rest is not None else node
        return node

    def or_expr(self) -> Optional[Node]:
        ops = [self.and_expr()]
        while self.at(_Tok.OR):
            self.next()
            self.saw_operator = True
            nxt = self.and_expr()
            if nxt is None:
                self.warnings.append("query ends with 'OR'; dropping it")
                break
            ops.append(nxt)
        ops = [o for o in ops if o is not None]
        if not ops:
            return None
        return ops[0] if len(ops) == 1 else OrNode(operands=ops)

    def and_expr(self) -> Optional[Node]:
        ops: List[Node] = []
        first = self.not_expr()
        if first is not None:
            ops.append(first)
        while True:
            if self.at(_Tok.AND):
                self.next()
                self.saw_operator = True
                nxt, moved = self._operand()
                if nxt is not None:
                    ops.append(nxt)
                elif not moved:
                    self.warnings.append("query ends with 'AND'; dropping it")
                    break
                continue
            # Adjacency is an implicit AND.  A plain run of words parses into the
            # same shape, but `saw_operator` stays False, so `parse()` sends it
            # down the free-text fast path instead.
            if self.at(_Tok.WORD, _Tok.PHRASE, _Tok.LP, _Tok.FIELD):
                nxt, moved = self._operand()
                if nxt is not None:
                    ops.append(nxt)
                elif not moved:
                    break
                continue
            break
        ops = [o for o in ops if o is not None]
        if not ops:
            return None
        return ops[0] if len(ops) == 1 else _and(ops)

    def _operand(self) -> Tuple[Optional[Node], bool]:
        """`not_expr`, plus whether it consumed anything.

        A top-level filter is *hoisted* into `self.filters` and contributes no
        tree node, so "returned None" and "there was nothing there" are different
        answers and the caller must not confuse them — that bug silently dropped
        every filter after the first.
        """
        before = self.i
        node = self.not_expr()
        return node, self.i > before

    def not_expr(self) -> Optional[Node]:
        if self.at(_Tok.NOT):
            tok = self.next()
            self.warnings.append(
                f"{tok.text!r} needs something on its left "
                "(write 'privacy NOT bail', not 'NOT bail'); ignoring it")
            return self.not_expr()
        left = self.prox_expr()
        while self.at(_Tok.NOT):
            self.next()
            self.saw_operator = True
            right = self.prox_expr()
            if right is None:
                self.warnings.append("query ends with 'NOT'; dropping it")
                break
            if left is None:                                 # pragma: no cover
                left = right
                continue
            left = NotNode(left=left, right=right)
        return left

    def prox_expr(self) -> Optional[Node]:
        left = self.atom()
        while self.at(_Tok.PROX):
            tok = self.next()
            self.saw_operator = True
            kind, k = tok.extra
            right = self.atom()
            if right is None:
                self.warnings.append(
                    f"query ends with {tok.text!r}; a proximity operator needs a "
                    "term on each side")
                break
            if left is None:
                self.warnings.append(
                    f"{tok.text!r} needs a term on its left; ignoring it")
                left = right
                continue
            left = ProxNode(kind=kind, k=k, left=left, right=right)
        return left

    def atom(self) -> Optional[Node]:
        t = self.peek()
        if t is None:
            return None
        if t.kind == _Tok.LP:
            self.next()
            self._depth += 1
            inner = self.or_expr()
            self._depth -= 1
            if self.at(_Tok.RP):
                self.next()
            else:
                self.warnings.append("unbalanced '(': closing it at the end")
            return inner
        if t.kind == _Tok.RP:
            return None
        if t.kind == _Tok.PHRASE:
            self.next()
            return self._phrase(t.text)
        if t.kind == _Tok.FIELD:
            self.next()
            return self._field(t)
        if t.kind == _Tok.WORD:
            self.next()
            return self._word(t.text)
        return None

    # ---- leaves ---------------------------------------------------------
    def _phrase(self, raw: str) -> Optional[Node]:
        terms = N.phrase_terms(raw)
        if not terms:
            self.warnings.append(f'the phrase "{raw}" has no searchable words')
            return None
        self.saw_operator = True
        if len(terms) == 1:
            return TermNode(term=terms[0], raw=raw)
        return PhraseNode(terms=terms, raw=raw)

    def _word(self, raw: str) -> Optional[Node]:
        if raw.endswith("!") or raw.endswith("*"):
            root = N.prefix_root(raw)
            if not root:
                self.warnings.append(f"{raw!r} has no searchable root")
                return None
            if len(root) < MIN_PREFIX:
                self.warnings.append(
                    f"{raw!r} expands from only {len(root)} characters, which "
                    f"would match most of the dictionary; needs {MIN_PREFIX}")
                return None
            self.saw_operator = True
            return TermNode(term=root, raw=raw, prefix=True)
        terms = N.terms_of(raw)
        if not terms:
            return None
        if len(terms) == 1:
            return TermNode(term=terms[0], raw=raw)
        # one surface word that analysed into several tokens ("302/34" ->
        # ipc§302 ipc§34): an implicit conjunction, not a phrase.
        return _and([TermNode(term=t, raw=raw) for t in terms])

    def _field(self, tok: _Tok) -> Optional[Node]:
        name, rest = tok.extra
        rest = rest.strip()

        # zone restriction:  title:(privacy)  /  title:privacy
        if N.is_zone_field(name) and not _looks_numeric(rest):
            self.saw_operator = True
            return self._zone(name, rest)

        if not N.is_filter_field(name):                      # pragma: no cover
            return self._zone(name, rest)

        self.saw_operator = True
        return self._filter(name, rest)

    def _zone(self, zone: str, rest: str) -> Optional[Node]:
        # The lexer always splits "(" into its own token, so `title:(privacy OR
        # dignity)` arrives as FIELD('title:') LP WORD OR WORD RP and `atom()`
        # handles the group; `rest` is only non-empty for `title:privacy`.
        if rest:
            node = self._word(rest)
        else:
            node = self.atom()
        if node is None:
            self.warnings.append(f"{zone}: needs something to search for")
            return None
        _push_zone(node, zone, self.warnings)
        return node

    def _filter(self, name: str, rest: str) -> Optional[Node]:
        op, value = _split_op(rest)
        if not value:
            value = self._filter_value(name)
            if value is None:
                return None
        elif value.startswith("'") and not value.endswith("'"):
            value = self._join_single_quoted(value)

        rng = N.parse_range(value)
        if rng and name == "year" and op in ("", "="):
            self.filters.extend(N.year_range(*rng))
            return None

        values = [v for v in re.split(r"\s*[,;|]\s*", value) if v] or [value]
        node, warn = N.coerce_filter(name, op or "=", values)
        self.warnings.extend(warn)
        if node is None:
            return None
        if self._depth == 0:
            self.filters.append(node)      # conjunctive restriction, hoisted
            return None
        # Inside parentheses the filter is part of the Boolean shape — hoisting
        # `(year:2000 OR year:2001)` would turn an OR into an AND and change the
        # answer — so it stays in the tree for the planner to execute.
        return node

    def _filter_value(self, name: str) -> Optional[str]:
        """The value of ``field:`` when it did not fit in the same token.

        Three shapes reach here, because the lexer breaks on ``(`` and ``"``:
        ``act:(IPC OR CRPC)``, ``type:"criminal appeal"`` and ``judge: khanna``.
        """
        nxt = self.peek()
        if nxt is None:
            self.warnings.append(f"{name}: needs a value")
            return None
        if nxt.kind == _Tok.LP:
            self.next()
            return self._collect_group()
        if nxt.kind == _Tok.PHRASE:
            return self.next().text
        if nxt.kind == _Tok.WORD:
            text = self.next().text
            if text.startswith("'") and not text.endswith("'"):
                return self._join_single_quoted(text)
            return text
        self.warnings.append(f"{name}: needs a value")
        return None

    def _collect_group(self) -> str:
        """``act:(IPC OR CRPC)`` -> ``"IPC,CRPC"`` — a membership test, so the
        connector inside does not matter."""
        parts: List[str] = []
        while True:
            t = self.peek()
            if t is None:
                self.warnings.append("unbalanced '(' in a filter value")
                break
            self.next()
            if t.kind == _Tok.RP:
                break
            if t.kind in (_Tok.OR, _Tok.AND):
                continue
            parts.append(t.text)
        return ",".join(p.strip() for p in parts if p.strip())

    def _join_single_quoted(self, head: str) -> str:
        """``type:'writ petition'`` — single quotes are not phrase delimiters in
        the lexer (``hon'ble`` would break), so stitch the words back here."""
        parts = [head]
        while not parts[-1].endswith("'"):
            t = self.peek()
            if t is None or t.kind != _Tok.WORD:
                break
            parts.append(self.next().text)
        return " ".join(parts).strip("'")


def _looks_numeric(value: str) -> bool:
    """`bench:>=5` is a filter; `bench:(chandrachud)` is a zone search."""
    v = value.strip()
    return bool(v) and (v[0] in "<>=!" or re.match(r"^\d", v) is not None)


def _split_op(rest: str) -> Tuple[str, str]:
    for op in N.OPS:
        if rest.startswith(op):
            return op, rest[len(op):].strip()
    return "", rest.strip()


def _and(ops: Sequence[Node]) -> Node:
    """Flatten nested ANDs so the planner sees one n-ary intersection."""
    flat: List[Node] = []
    for o in ops:
        if isinstance(o, AndNode):
            flat.extend(o.operands)
        else:
            flat.append(o)
    return flat[0] if len(flat) == 1 else AndNode(operands=flat)


def _push_zone(node: Node, zone: str, warnings: List[str]) -> None:
    """Apply a zone restriction to every leaf beneath it."""
    if isinstance(node, (TermNode, PhraseNode)):
        if node.zone and node.zone != zone:
            warnings.append(f"{node.describe()} is already restricted to "
                            f"{node.zone}; keeping that")
            return
        node.zone = zone
        return
    for child in node.children():
        _push_zone(child, zone, warnings)


# ==========================================================================
# Public entry point
# ==========================================================================


def parse(query: str) -> Query:
    """Text -> `ParsedQuery`.  Never raises; a bad query degrades to free text."""
    raw = query or ""
    warnings: List[str] = []
    normalised = N.normalise_query(raw)

    tree: Optional[Node] = None
    filters: List[FilterNode] = []
    degraded = False
    saw_operator = False
    try:
        toks = _lex(normalised, warnings)
        p = _Parser(toks, warnings)
        tree = p.parse()
        filters = p.filters
        saw_operator = p.saw_operator or bool(filters)
    except Exception as exc:                 # never crash the CLI, ever
        warnings.append(f"could not parse the query ({exc.__class__.__name__}: "
                        f"{exc}); searching for the words instead")
        degraded = True
        tree, filters, saw_operator = None, [], False

    # free_terms comes from the tree's **leaves**, not from re-analysing the raw
    # string: otherwise the operators leak in and `murder /3 intention` acquires
    # a phantom term "3" that then gets expanded to ipc§3.
    if tree is not None:
        free_terms = _dedup(_leaf_terms(tree))
    elif filters and not degraded:
        free_terms = []        # filters only: there is no text to rank by, and
                               # re-analysing the raw string would turn the field
                               # names themselves ("act", "disposal") into terms
    else:
        free_terms = _dedup(N.terms_of(normalised))

    is_free_text = not saw_operator
    root = None if is_free_text else tree

    expansions = N.build_expansions(free_terms + _leaf_terms(root))
    q = Query(
        raw=raw,
        root=root,
        free_terms=free_terms,
        filters=filters,
        is_free_text=bool(is_free_text),
        expansions=expansions,
        warnings=warnings,
        notes=N.expansion_notes(expansions),
        degraded=degraded,
        normalised=normalised,
    )
    return q


def _dedup(terms: List[str]) -> List[str]:
    """Order-preserving de-duplication; a repeated query term adds nothing."""
    seen, out = set(), []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _leaf_terms(node: Optional[Node]) -> List[str]:
    if node is None:
        return []
    if isinstance(node, TermNode):
        return [node.term]
    if isinstance(node, PhraseNode):
        return list(node.terms)
    out: List[str] = []
    for c in node.children():
        out.extend(_leaf_terms(c))
    return out


# ==========================================================================
# --explain:  the AST as a tree (the video shows this)
# ==========================================================================


def describe_tree(node: Optional[Node], empty: str = None) -> str:
    """The AST as an indented tree.  This is the figure in the demo video."""
    if node is None:
        return empty or "(no term tree - free-text fast path)"
    lines: List[str] = []

    def walk(n: Node, prefix: str, is_last: bool, root: bool) -> None:
        if root:
            lines.append(_label(n))
            child_prefix = ""
        else:
            lines.append(prefix + ("`- " if is_last else "|- ") + _label(n))
            child_prefix = prefix + ("   " if is_last else "|  ")
        kids = list(n.children())
        for i, k in enumerate(kids):
            walk(k, child_prefix, i == len(kids) - 1, False)

    walk(node, "", True, True)
    return "\n".join(lines)


def _label(node: Node) -> str:
    if isinstance(node, AndNode):
        return f"AND  ({len(node.operands)} operands)"
    if isinstance(node, OrNode):
        return f"OR   ({len(node.operands)} operands)"
    if isinstance(node, NotNode):
        return "NOT  (left minus right)"
    if isinstance(node, ProxNode):
        return {"s": "PROX same sentence", "p": "PROX same paragraph"}.get(
            node.kind, f"PROX within {node.k} words")
    return node.describe()


def explain(query: str) -> str:
    """Everything the parser decided, as text.  Used by the CLI and the UI."""
    q = parse(query)
    out = [f"raw          : {q.raw}"]
    if q.normalised.strip() != q.raw.strip():
        out.append(f"canonicalised: {q.normalised}")
    out.append(f"path         : {'FREE TEXT (ranked bag of words)' if q.is_free_text else 'BOOLEAN + ranked'}")
    out.append("")
    out.append("parse tree")
    empty = ("(no term tree - this query is filters only)" if q.filters and not q.root
             else None)
    out.append(describe_tree(q.root, empty))
    out.append("")
    out.append(f"free terms   : {' '.join(q.free_terms) or '(none)'}")
    out.append(f"tf-idf terms : {' '.join(q.scoring_terms) or '(none)'}"
               "   <- stop words dropped from scoring, kept in positions")
    if q.filters:
        out.append("filters      :")
        for f in q.filters:
            out.append(f"   {f.describe()}")
    if q.expansions:
        out.append("expansions   :  (query time only - never written to the index)")
        for term, alts in q.expansions.items():
            out.append(f"   {term} OR {' OR '.join(alts)}")
        for note in q.notes:
            if ":" in note:
                out.append(f"     ! {note.split(': ', 1)[-1]}")
    if q.warnings:
        out.append("warnings     :")
        for w in q.warnings:
            out.append(f"   - {w}")
    return "\n".join(out)


EXAMPLES = (
    "right to privacy",
    '"right to privacy" AND dignity',
    "murder /3 intention",
    "privacy /s surveillance NOT bail!",
    "title:(privacy) AND (dignity OR autonomy) year:>=2015 bench:>=5",
    "S. 302 IPC AND dying declaration",
    "sec. 302 r/w 34",
    'act:NDPS AND type:"criminal appeal" disposal:allowed judge:chandrachud',
    "year:2000-2010 AND contempt",
    'bail! /p "anticipatory bail"',
    'unbalanced "quote AND stray ) /x',
)


def _main(argv: Optional[List[str]] = None) -> int:           # pragma: no cover
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.query.parser")
    ap.add_argument("--explain", metavar="QUERY", nargs="?", const="",
                    help="pretty-print the parse of one query")
    ap.add_argument("--examples", action="store_true",
                    help="run --explain over the built-in example queries")
    args = ap.parse_args(argv)

    if args.examples:
        for q in EXAMPLES:
            print("=" * 78)
            print(explain(q))
            print()
        return 0
    if args.explain:
        print(explain(args.explain))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
