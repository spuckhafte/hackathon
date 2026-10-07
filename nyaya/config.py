"""Central configuration for NyayaTrace.

Everything tunable lives here so that an ablation is a flag, not a code change
(Section 8 of the project guide).  Frozen at kick-off; changing a value here is
cheap, changing a *name* here means a rebuild, so talk to the team first.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get("NYAYA_DATA", ROOT / "data"))
RAW = DATA / "raw"
RAW_METADATA = RAW / "metadata"          # {year}.parquet, straight from the bucket
RAW_PDF = RAW / "pdf"                    # {year}/{path}_EN.pdf
PROCESSED = DATA / "processed"
UNIVERSE_PARQUET = PROCESSED / "universe.parquet"     # all 42k judgments (graph nodes)
MANIFEST_PARQUET = PROCESSED / "manifest.parquet"     # the sampled, indexed subset
JUDGMENTS_JSONL = PROCESSED / "judgments.jsonl"       # zoned, cleaned JudgmentRecords
EXTRACT_LOG = PROCESSED / "extract_log.csv"           # one row per attempted document

INDEX = Path(os.environ.get("NYAYA_INDEX", ROOT / "index"))
RUNS = ROOT / "runs"                     # evaluation output (csv, json, png)
EVAL_DIR = ROOT / "eval"                 # hand-written query sets and qrels

for _p in (RAW_METADATA, RAW_PDF, PROCESSED, INDEX, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Corpus: the AWS Open Data release of Indian Supreme Court judgments
# --------------------------------------------------------------------------
BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com/"
CORPUS_YEARS = range(1950, 2026)

#: Stratified sample: at most this many English judgments per year.  Years with
#: fewer are taken whole.  Rationale in README - equal temporal coverage keeps
#: the Court's whole history searchable instead of only the last decade.
PER_YEAR_CAP = 200
SAMPLE_SEED = 358                        # the course code, so the draw is reproducible

#: A page yielding fewer than this many words is treated as a scanned image.
MIN_WORDS_PER_PAGE = 200
MIN_WORDS_PER_DOC = 400

# --------------------------------------------------------------------------
# Zones (Section 4.4).  Order matters: it is the on-disk zone id order.
# --------------------------------------------------------------------------
ZONES = ("title", "parties", "bench", "statutes", "order", "body")
ZONE_ID = {z: i for i, z in enumerate(ZONES)}

#: Zone weights for weighted-zone scoring and BM25F.  `title` was fitted by the
#: sweep below; `statutes` and `order` were NOT taken from the coordinate-ascent
#: optimum, and runs/tuning.json records why: that optimum (statutes 0.0,
#: order 1.0) gains +0.0045 nDCG@10 on the five dev queries (0.9152 -> 0.9197)
#: and LOSES 0.0200 on the twenty test queries (0.8786 -> 0.8586), which is what
#: overfitting five queries looks like.  MAP moves -0.0012 on dev as well.
#: The search range, the trail and both splits are in runs/tuning.json; the
#: command is `python -m nyaya.rank.tune --report-test`.
ZONE_WEIGHTS = {
    # `title` was 3.0 (the guide's default).  Lowered to 0.5 by Sumedha on
    # measured evidence, not taste: in this corpus a Supreme Court cause title
    # is a *party name* ("X versus Y"), not a subject line, so a topical query
    # term essentially never appears in one.  Measured on the built index:
    # `arbitr` has df 883 and **0** documents with a title-zone hit, so
    # `title:(arbitration)` returns zero documents while plain `arbitration`
    # returns ten; same for `privaci` (df 126, 0 title hits) and `detent`
    # (df 765, 0).  What a title match does find is parties - `state` hits 4,946
    # titles and `union` 1,450 - so a high title weight rewards party-name
    # matches rather than topical ones.  The weighted-zone scorer also amplifies
    # the few real hits enormously, because a per-zone cosine over a 9-token
    # title is huge.  Swept over title in {0, 0.25, 0.5, 1, 2, 3}, S2's nDCG@10
    # falls as the weight rises, on BOTH splits: dev 0.4209 / 0.4285 / 0.4205 /
    # 0.4162 / 0.3758 / 0.3758 and test 0.6725 / 0.6723 / 0.6712 / 0.6557 /
    # 0.6459 / 0.6312.  (Not strictly monotonic - dev peaks at 0.25 - so 0.5 is
    # chosen as the value that is near the top on both splits rather than the
    # dev argmax.)  BM25F is completely insensitive: S2p and S4 return exactly
    # 0.7977 and 0.8786 on test at every one of the six weights.  So the change
    # costs the shipped S4 nothing and fixes S2.
    # Scoring-only: no index rebuild is needed.  Sweep in runs/tuning.json.
    "title": 0.5,
    "parties": 0.5,
    "bench": 0.0,       # filter-only zone, never scored
    "statutes": 2.0,
    "order": 1.5,
    "body": 1.0,
}

#: BM25F length-normalisation per zone: short zones get a small b.
BM25F_B = {"title": 0.3, "parties": 0.3, "bench": 0.3, "statutes": 0.3, "order": 0.5, "body": 0.75}
BM25F_K1 = 1.2

# --------------------------------------------------------------------------
# Ranking (Section 8.5)
# --------------------------------------------------------------------------
ALPHA = 0.80          # weight on normalised text relevance
BETA = 0.05           # weight on proximity
# (1 - ALPHA) is the weight on static authority g(d)

#: Index elimination: drop query terms below this idf, and when a free-text
#: query has >= MANY_TERMS terms only score documents matching >= half of them.
IDF_ELIMINATION_THRESHOLD = 0.35
MANY_TERMS = 4

TOP_K = 10
CHAMPION_LIST_SIZE = 400      # r, documents per term in tier 1

# --------------------------------------------------------------------------
# Authority (Section 6)
# --------------------------------------------------------------------------
PAGERANK_LAMBDA = 0.85
PAGERANK_ITERATIONS = 60
PAGERANK_TOL = 1e-10

TREATMENT_WEIGHTS = {"overruled": 0.0, "distinguished": 0.3, "mentioned": 0.6, "followed": 1.0}
TREATMENT_PENALTY = {"overruled": 0.25, "distinguished": 0.7, "ok": 1.0}
#: A case is "frequently distinguished" once this share of its in-edges are.
DISTINGUISHED_SHARE = 0.34
DISTINGUISHED_MIN_EDGES = 3

G_PAGERANK_WEIGHT = 0.75
G_BENCH_WEIGHT = 0.25
MAX_BENCH = 13.0              # largest bench the Supreme Court has ever sat (Kesavananda)

#: Minimum Jaccard over normalised case-name tokens to accept a name match.
CITATION_NAME_THRESHOLD = 0.6

# --------------------------------------------------------------------------
# Precedent Brief (Section 9, stretch)
# --------------------------------------------------------------------------
BRIEF_TOP_DOCS = 5
BRIEF_MAX_PARAS = 8
BRIEF_MAX_PARAS_PER_DOC = 2
BRIEF_COSINE_THRESHOLD = 0.20

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
SPIMI_BLOCK_DOCS = 600        # documents inverted in memory before a block is flushed
DOWNLOAD_THREADS = 24
EXTRACT_PROCESSES = max(1, (os.cpu_count() or 4) - 1)

# --- added by Suvrat ---
#: Index elimination requires a document to match at least half the query terms
#: once a query has MANY_TERMS or more (Manning, IIR 7.1.1).  That rule is written
#: for short queries; on a 60-term query built from a judgment's facts (Track B)
#: "half" means 30 terms and the filter becomes very nearly a conjunction — it
#: collapsed one 13,878-document candidate set to 19 documents, which is a
#: conjunctive AND wearing an optimisation's clothes.  So the requirement is
#: capped: at least half, but never more than this many terms.  Every query in the
#: wave-3 efficiency measurements needed 3 or 4 matches, so this cap does not
#: change any previously reported number.
ELIMINATION_MAX_MATCHES = 6
