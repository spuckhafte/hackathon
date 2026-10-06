# NyayaTrace — team contracts (frozen at kick-off)

Three members build in parallel against the interfaces in `nyaya/contracts.py`
and the constants in `nyaya/config.py`. **Neither file changes without telling
the whole team**, because every module is written against them.

| Member | Name    | Owns |
|--------|---------|------|
| M1     | Rakshit | ingest, zones, positional index, proximity operators, treatment classifier, proximity scoring, snippets, evaluation harness + metrics, README/video |
| M2     | Suvrat  | metadata & parametric index, tiered index + champion lists, citation extraction/resolution, PageRank and g(d), Boolean planner, net score + heap + index elimination, citation-recovery & efficiency evaluation, report assembly |
| M3     | Sumedha | statute canonicaliser + analyze(), paragraph store, collection statistics, query parser + normaliser, tf-idf / zone / BM25F scorers, "why this result", CLI + Streamlit page, operator tests, query-set curation |

## 1. Ownership rule

**Only edit files you own.** If you need a change in someone else's file, add a
TODO comment in your own module and say so in the hand-off note. The only shared
files are `nyaya/config.py`, `nyaya/contracts.py`, `nyaya/utils.py` and
`README.md` — treat these as append-only and never reformat them.

## 2. Data layout

```
data/raw/metadata/{year}.parquet        # straight from the public bucket
data/raw/pdf/{year}/{path}_EN.pdf       # downloaded judgment PDFs
data/processed/universe.parquet         # ALL 37,210 English judgments -> graph nodes
data/processed/manifest.parquet         # the stratified subset we index
data/processed/judgments.jsonl          # one JudgmentRecord per line (zoned, cleaned)
data/processed/extract_log.csv          # one row per attempted document, with drop reason
index/                                  # S1..S5, all build outputs
runs/                                   # evaluation CSV/JSON/PNG
eval/queries/                           # hand-written query sets and qrels
```

### Corpus policy (decided at kick-off, document it in the README)

* **Universe** — every English judgment in the AWS Open Data release,
  1950-2025, **37,210 rows** after de-duplication on `path` (42,371 English rows before dedup; 5,161 are the same judgment scraped once per available language). These are the *nodes*
  of the citation graph and the source of parametric facts, whether or not we
  hold their text.
* **Indexed subset** — a stratified sample of at most `config.PER_YEAR_CAP`
  (200) judgments **per year**, drawn with `config.SAMPLE_SEED` (358). Years
  with fewer English judgments are taken whole. This gives ≈14,000 documents
  with equal temporal coverage, so the Court's whole history is searchable
  rather than only the last decade.
* The sampling rate (≈28% in busy years, 100% in the 1950s) is a **known
  limitation**: a cited precedent is only retrievable if it was sampled (manifest is 37.5% of the universe), which
  caps citation-recovery recall. Say so in the report; do not hide it.

## 3. Stores

| Store | File(s) under `index/` | Owner | Contents |
|-------|------------------------|-------|----------|
| S1 | `paragraphs.jsonl`, `paragraphs.idx.json` | M3 | numbered paragraphs per judgment |
| S2 | `parametric.json`, `parametric_*.npy` | M2 | year, bench size, judges, acts, case type, disposal |
| S3 | `dictionary.json`, `postings.bin`, `docmeta.npz` | M1 | positional, zoned inverted index |
| S4 | `citations.jsonl`, `authority.json` | M2 | citation graph, PageRank, g(d), status |
| S5 | `champions.json`, `tiers.npz` | M2 | champion lists and the two-tier index |

## 4. Key signatures (do not change)

```python
# nyaya/analyze.py                                             (M3)
analyze(text: str, *, zone: str = "body", para_id: int = 0,
        stem: bool = True, canonicalise: bool = True) -> list[Token]
analyze_terms(text: str, **kw) -> list[str]        # terms only, for queries
canonicalise_statutes(text: str) -> str            # raw text -> text with ipc§302 tokens

# nyaya/index/positional.py                                    (M1)
class PositionalIndex:
    @classmethod
    def load(cls, path=config.INDEX) -> "PositionalIndex"
    def postings(self, term: str) -> PostingsList | None
    def df(self, term: str) -> int
    def idf(self, term: str) -> float                  # ln(1+(N-df+0.5)/(df+0.5))
    def prefix_terms(self, prefix: str) -> list[str]   # for  bail!
    doc_count: int
    doc_len: np.ndarray          # total tokens per doc
    zone_len: np.ndarray         # (n_docs, n_zones) tokens per zone
    doc_norm: np.ndarray         # lnc vector length per doc, for the B0 baseline

# nyaya/index/parametric.py                                    (M2)
class ParametricIndex:
    def select(self, field: str, op: str, value) -> np.ndarray   # sorted doc ids

# nyaya/query/parser.py                                        (M3)
parse(query: str) -> ParsedQuery

# nyaya/query/planner.py                                       (M2)
execute(node: Node, ctx: SearchContext) -> np.ndarray           # sorted doc ids

# nyaya/query/proximity.py                                     (M1)
phrase_docs(terms, ctx, zone=None) -> np.ndarray
proximity_docs(left_ids, right_ids, left_terms, right_terms, kind, k, ctx) -> np.ndarray

# nyaya/authority/gscore.py                                    (M3)
class AuthorityStore:
    def g(self, doc_id: int) -> float
    def status(self, doc_id: int) -> str
    def badge(self, doc_id: int) -> str

# nyaya/search.py                                              (M3 wires it)
search(query: str, k: int = 10, mode: str = "S4", explain: bool = True) -> list[Result]
```

`SearchContext` (defined in `nyaya/search.py`) bundles the loaded stores:
`.index` (S3), `.params` (S2), `.paras` (S1), `.authority` (S4), `.tiers` (S5),
`.docs` (list of lightweight doc records), `.mode`.

## 5. Integration points (where the bugs will be)

1. `analyze()` (M3) is called by the index builder (M1) **and** by the query
   normaliser (M3). Freeze it early; changing it means a full rebuild.
2. The planner (M2) calls the positional merges (M1) through `PostingsList`.
3. The scorers (M3) receive candidate doc-id arrays from the planner (M2) and
   return score arrays to the net-score function (M2), which reads g(d) (M2).
4. Before every push: `python -m pytest tests -q` and the smoke test
   `python -m nyaya.cli --smoke` must pass.

## 6. Git conventions

Commit message format, exactly:

```
<Name> - <feat|fix|chore|test|docs|perf|refactor>: <what changed>
```

e.g. `Rakshit - feat: SPIMI block inversion with numpy int32 postings`

* Commit as yourself: `git -c user.name="Rakshit" -c user.email="rakshit@nyayatrace.local" commit -m "..."`.
* **Never `git add -A`.** Add only the files you own, by path.
* If `.git/index.lock` exists another member is mid-commit — wait a second and retry.
* Commit in small steps as work lands, not one giant commit at the end.

## 7. Definition of done for a component

* It runs on the real corpus, not on a stub.
* It has at least one test in `tests/` that a reviewer can read.
* It can print an intermediate output on demand (postings, plan, weights,
  scores) — that is what the demo video must show.
* Its behaviour is explainable in two sentences without saying "the library
  does it".

## 8. Wave-2 facts established by the metadata pass (Suvrat)

* Universe after de-duplication on `path`: **37,210** nodes. Indexed manifest:
  **13,942** documents (37.5%). Quote these numbers, not 42,371.
* **Citation resolution order is `scr` -> `neutral` -> fuzzy name Jaccard.**
  100.00% of universe nodes carry an SCR citation whose normalised key is
  unique (0 collisions); 99.81% carry a neutral citation (116 key collisions,
  one INSC number issued for a batch). Exact keys therefore resolve nearly
  everything and `config.CITATION_NAME_THRESHOLD` matching is a last resort,
  not the main path. `metadata.node_id_by_key()` returns the combined lookup.
* **`bench_size` vs `bench_size_meta`.** 87.4% of metadata rows name a single
  judge, because the column holds the *authoring* judge, not the bench. The
  real bench size comes from Rakshit's CORAM / `[... , JJ.]` parse of the PDF.
  Anything that scores authority must read the merged `bench_size` via
  `metadata.merge_from_record()`; reading `bench_size_meta` makes every
  judgment look like a one-judge bench and turns `config.G_BENCH_WEIGHT` into
  noise.
* `acts` is the only parametric field metadata cannot populate: it comes from
  Sumedha's `canonicalise_statutes()` at S2 build time.
