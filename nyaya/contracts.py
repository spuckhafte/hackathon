"""Frozen interfaces between the three members' components (guide Section 3.2).

Nothing in this file changes after kick-off without telling the whole team,
because every module is written against it.  The dataclasses are deliberately
plain so they serialise to JSON Lines without a schema library.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ==========================================================================
# 1. Documents  (producer: Member 1 ingest/zones + Member 2 metadata)
# ==========================================================================


@dataclass
class Para:
    """A numbered paragraph: the unit of evidence (guide Section 4.2)."""

    id: int                 # 1-based paragraph number within the judgment
    zone: str               # one of config.ZONES
    text: str
    label: str = ""         # printed paragraph number when the judgment had one


@dataclass
class JudgmentRecord:
    """One Supreme Court judgment, cleaned, zoned and joined to its metadata."""

    doc_id: int
    source_path: str                  # bucket key stem, e.g. "2017_11_1029_1035"
    title: str
    petitioner: str = ""
    respondent: str = ""
    date: str = ""                    # ISO yyyy-mm-dd when parseable
    year: int = 0
    judges: List[str] = field(default_factory=list)
    bench_size: int = 0
    case_no: str = ""                 # "Civil Appeal No. 238 of 2012"
    neutral_cite: str = ""            # "2017 INSC 1136"
    scr_cite: str = ""                # "[2017] 11 S.C.R. 1029"
    disposal: str = ""                # "Dismissed", "Allowed", ...
    case_type: str = ""               # "civil appeal" | "criminal appeal" | ...
    acts: List[str] = field(default_factory=list)            # ["IPC", "CONST"]
    # Canonical statute tokens, lower-case, section sign separated:
    #   "ipc§302", "const§21", "crpc§482", "bns§103".  The section sign is in
    #   cp1252, so it survives a Windows console in the demo.
    statute_tokens: List[str] = field(default_factory=list)
    zones: Dict[str, str] = field(default_factory=dict)      # zone -> raw text
    paras: List[Para] = field(default_factory=list)
    n_tokens: int = 0
    n_pages: int = 0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "JudgmentRecord":
        d = dict(d)
        d["paras"] = [Para(**p) for p in d.get("paras", [])]
        return JudgmentRecord(**d)


# ==========================================================================
# 2. Analysis  (producer: Member 3 - nyaya/analyze.py)
# ==========================================================================


@dataclass
class Token:
    """One indexed token.  `pos` is the running offset within the zone."""

    term: str
    pos: int
    sent_id: int
    para_id: int
    zone: str = "body"
    raw: str = ""           # surface form, kept for snippet highlighting


# analyze(text, zone=..., para_id=...) -> list[Token] is THE shared entry point.
# Documents and queries must go through the same function; if they ever diverge,
# matching silently breaks.


# ==========================================================================
# 3. Postings  (producer: Member 1 - nyaya/index/positional.py)
# ==========================================================================


@dataclass
class Posting:
    """One (term, document) entry of the positional, zoned inverted index."""

    doc_id: int
    zone_tf: Dict[str, int]
    positions: Sequence[int]          # sorted running offsets

    @property
    def tf(self) -> int:
        return sum(self.zone_tf.values())


class PostingsList:
    """Read side of one term's postings.

    Implementations expose `doc_ids` as a sorted numpy int32 array and support
    skip-aware intersection.  `positions_of(doc_id)` returns the sorted position
    array for one document.
    """

    doc_ids: Any          # np.ndarray[int32], ascending
    df: int

    def positions_of(self, doc_id: int) -> Any: ...          # np.ndarray[int32]

    def zone_tf_of(self, doc_id: int) -> Dict[str, int]: ...

    def skip_stride(self) -> int: ...

    def __iter__(self) -> Iterable[Posting]: ...


# ==========================================================================
# 4. Query AST  (producer: Member 3 parser; consumer: Member 2 planner)
# ==========================================================================


class Node:
    """Base class of the query abstract syntax tree."""

    def children(self) -> Sequence["Node"]:
        return ()

    def describe(self) -> str:
        return self.__class__.__name__


@dataclass
class TermNode(Node):
    term: str                      # already normalised by analyze()
    raw: str = ""
    zone: Optional[str] = None     # title:(privacy) restricts to one zone
    prefix: bool = False           # bail!  -> dictionary prefix scan

    def describe(self) -> str:
        z = f"{self.zone}:" if self.zone else ""
        star = "*" if self.prefix else ""
        return f"TERM({z}{self.term}{star})"


@dataclass
class PhraseNode(Node):
    terms: List[str]
    raw: str = ""
    zone: Optional[str] = None

    def describe(self) -> str:
        return "PHRASE[" + " ".join(self.terms) + "]"


@dataclass
class ProxNode(Node):
    kind: str                 # 'n' within k words | 's' same sentence | 'p' same para
    k: int
    left: Node
    right: Node

    def children(self):
        return (self.left, self.right)

    def describe(self) -> str:
        if self.kind == "n":
            return f"PROX_N({self.k})"
        return f"PROX_{self.kind.upper()}"


@dataclass
class AndNode(Node):
    operands: List[Node]

    def children(self):
        return tuple(self.operands)


@dataclass
class OrNode(Node):
    operands: List[Node]

    def children(self):
        return tuple(self.operands)


@dataclass
class NotNode(Node):
    """`left NOT right` - set difference, never a bare negation."""

    left: Node
    right: Node

    def children(self):
        return (self.left, self.right)


@dataclass
class FilterNode(Node):
    """A parametric-index constraint: year:>=2000, bench:>=5, act:IPC."""

    field: str                # year | bench | judge | act | type | disposal
    op: str                   # '=' | '>=' | '<=' | '>' | '<' | 'in'
    value: Any

    def describe(self) -> str:
        return f"FILTER({self.field}{self.op}{self.value})"


@dataclass
class ParsedQuery:
    """What the parser hands to the planner."""

    raw: str
    root: Optional[Node] = None                              # None for pure free text
    free_terms: List[str] = field(default_factory=list)      # normalised bag of words
    filters: List[FilterNode] = field(default_factory=list)
    is_free_text: bool = True                                # no operators -> fast path
    expansions: Dict[str, List[str]] = field(default_factory=dict)   # term -> OR alts


# ==========================================================================
# 5. Authority  (producer: Member 2 citations/pagerank, Member 1 treatment)
# ==========================================================================


@dataclass
class CitationEdge:
    src: int                  # citing doc_id (always in the indexed corpus)
    dst: int                  # cited node id (may be a universe-only judgment)
    treatment: str            # followed | distinguished | overruled | mentioned
    sentence: str = ""        # evidence sentence, shown in the UI
    para_id: int = 0
    cite_string: str = ""
    resolved_by: str = ""     # 'neutral' | 'scr' | 'name' | 'unresolved'


@dataclass
class Authority:
    """Static quality of one judgment."""

    doc_id: int
    g: float                  # final static score in [0, 1]
    pagerank: float = 0.0
    pagerank_norm: float = 0.0
    bench_norm: float = 0.0
    status: str = "ok"        # ok | distinguished | overruled
    penalty: float = 1.0
    in_degree: int = 0
    out_degree: int = 0
    overruled_by: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


# ==========================================================================
# 6. Results  (Member 2 net score + Member 1 snippets + Member 3 explanation)
# ==========================================================================


@dataclass
class ScoreExplanation:
    """The "why this result" panel (feature F5)."""

    text_raw: float = 0.0
    text_norm: float = 0.0
    per_zone: Dict[str, float] = field(default_factory=dict)
    top_terms: List[Tuple[str, float]] = field(default_factory=list)
    proximity: float = 0.0
    prox_window: int = 0
    g: float = 0.0
    pagerank_norm: float = 0.0
    bench_norm: float = 0.0
    penalty: float = 1.0
    alpha: float = 0.0
    beta: float = 0.0
    net: float = 0.0
    scorer: str = "bm25f"
    notes: List[str] = field(default_factory=list)


@dataclass
class Result:
    doc_id: int
    title: str
    year: int
    score: float
    bench_size: int = 0
    judges: List[str] = field(default_factory=list)
    neutral_cite: str = ""
    scr_cite: str = ""
    case_no: str = ""
    disposal: str = ""
    status: str = "ok"
    badge: str = ""                      # "overruled in part by X (2017) para 12"
    snippets: List[str] = field(default_factory=list)
    explanation: Optional[ScoreExplanation] = None


# ==========================================================================
# 7. The single public entry point everything is measured through
# ==========================================================================
# nyaya.search.search(query: str, k: int = 10, mode: str = 'S4',
#                     explain: bool = True) -> list[Result]
#
# modes:  'B0'   flat tf-idf lnc.ltc, plain tokenizer, no zones, no authority
#         'S1'   B0 + statute canonicalisation + legal stop list
#         'S2'   S1 + weighted zone scoring
#         'S2p'  S1 + BM25F
#         'S3'   S2p + authority from plain PageRank
#         'S4'   S3 + treatment-aware g(d) + proximity   (the full system)
MODES = ("B0", "S1", "S2", "S2p", "S3", "S4")
FULL_MODE = "S4"
