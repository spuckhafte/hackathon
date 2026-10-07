# NyayaTrace

**Treatment-aware vertical search over Supreme Court of India judgments, 1950–2025.**
CSD358 Information Retrieval · mid-term hackathon · **Track T6 (vertical search)**.
Rakshit · Suvrat · Sumedha

A lawyer looking for precedent is not asking "which document is about privacy".
They are asking "which binding judgment, **still good law**, decided the point I
am arguing, and how have later benches treated it". NyayaTrace answers the second
question as well as the first: the index is zoned and positional, statutes are
canonicalised so that *Section 302 IPC*, *S. 302 of the Indian Penal Code, 1860*
and *u/s 302* become one term, the citation graph carries a **legal relation**
(followed / distinguished / mentioned / overruled) rather than a hyperlink, and a
judgment a later bench overruled is demoted and badged instead of being ranked
first because a hundred earlier judgments cited it.

Everything in the retrieval path is implemented from scratch — SPIMI inversion,
positional postings with skip pointers, the Boolean planner, lnc.ltc / weighted
zone / BM25F scoring, PageRank by our own power iteration. No search engine, no
vector database, no embedding model.

The full write-up is `report/NyayaTrace-report.pdf`; the demo film is in `video/`.

---

## Contents

- [Setup](#setup)
- [Where the data comes from](#where-the-data-comes-from)
- [Corpus policy, stated honestly](#corpus-policy-stated-honestly)
- [How to run — building the stores](#how-to-run--building-the-stores)
- [How to run — querying](#how-to-run--querying)
- [How to reproduce the evaluation](#how-to-reproduce-the-evaluation)
- [Corpus and index statistics](#corpus-and-index-statistics)
- [What works, what is planned](#what-works-what-is-planned)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [Repository map](#repository-map)

---

## Setup

**Python 3.11 or newer** (developed and measured on **3.14.5**, Windows 11;
nothing in the code is OS-specific, but the build fans out over
`ProcessPoolExecutor`, which spawns rather than forks on Windows).

```bash
git clone <this repo>
cd hackathon
python -m venv .venv
.venv/Scripts/activate          # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is deliberately short: `numpy`, `pandas`/`pyarrow` for the
metadata frames, `pymupdf` for PDF text, `nltk` **for the Porter stemmer only**,
`matplotlib` for the evaluation plots, `requests`/`tqdm` for the bulk download,
`networkx` **as a cross-check on our own PageRank, never as the implementation**,
`streamlit` for the optional UI and `pytest` for the tests. Nothing in that list
does retrieval.

**No AWS account, no credentials, no `aws` CLI, no `boto3`.** The dataset is a
public bulk release and we fetch it over plain HTTPS from
`https://indian-supreme-court-judgments.s3.amazonaws.com/` with `requests`.
There is nothing to sign up for and nothing to configure.

Two extras are needed only to rebuild the *report*, not to run the system:
`pip install markdown playwright && playwright install chromium`.
Rebuilding the *video* additionally needs Node (developed on Node 26) and
`npm install` inside `video/`.

On Windows, set `PYTHONIOENCODING=utf-8` before anything that prints: the CLI
emits `ipc§302` and `¶`, and a Windows console is cp1252 by default.

```bash
export PYTHONIOENCODING=utf-8        # PowerShell: $env:PYTHONIOENCODING = "utf-8"
```

---

## Where the data comes from

**Indian Supreme Court Judgments**, from the **AWS Open Data Registry**,
maintained by **Dattam Labs**, licensed **CC-BY-4.0**:

> <https://registry.opendata.aws/indian-supreme-court-judgments>

As the registry asks, we credit it explicitly: the judgment PDFs and the per-year
metadata parquet files used by this project come from that dataset, and the
dataset is the work of its maintainers, not of this team. The licence is
CC-BY-4.0 and this README is the attribution.

**No crawling was performed.** This is a bulk release served from a public object
store, so we read the objects the dataset publishes and nothing else. There is no
crawl frontier, no link following, no rate-limited politeness policy and no
robots.txt question to answer — those do not arise for a bulk download, and we
would rather say so plainly than claim a politeness policy we never needed. The
only courtesy the download owes the bucket is a bounded thread pool, which
`config.DOWNLOAD_THREADS = 24` provides.

**No personal data is collected.** The corpus is published judicial decisions,
which are public records; we extract no party contact details and build no
profiles, and nothing about a *user* of this system is stored anywhere — there is
no account, no query log and no telemetry.

---

## Corpus policy, stated honestly

This is the part to check before believing any recall number.

**The universe — the citation graph's nodes.** The release holds **42,371**
English judgment rows for 1950–2025. **12.2% of them are duplicates**: the same
judgment is scraped once per available language, and de-duplicating on the bucket
`path` removes 5,161 rows, leaving **37,210 distinct judgments**. Those 37,210
are the nodes of the citation graph and the source of every parametric fact,
whether or not we hold the text. Quote 37,210, not 42,371.

**The indexed subset — a stratified sample.** Downloading and extracting 37,210
PDFs was not within the time budget, so we index the text of a stratified sample:
at most **200 judgments per year** (`config.PER_YEAR_CAP`), drawn with
`config.SAMPLE_SEED = 358` so the draw is reproducible, taking whole any year
with fewer. That gives a **13,942-document manifest**, of which **13,878
extracted successfully (99.54%)**. The 64 drops are **53 scanned** documents
(every page under the 200-word floor), **7 under the word floor**, **2 not in
English** and **2 absent from the bucket** — each with its reason in
`data/processed/extract_log.csv`, so the corpus is auditable rather than merely
large.

Sampling per *year* rather than taking the most recent 14,000 judgments keeps the
Court's whole history searchable: *A. K. Gopalan* (1950) and *Puttaswamy* (2017)
are both in the index.

**What that costs, said out loud.** The sampling rate is ≈28% in busy years and
100% in the 1950s, and the manifest is 37.5% of the universe. **A cited precedent
is only retrievable if it happened to be sampled.** That caps citation-recovery
recall directly and measurably: of the 28,028 resolved citations only 15,358
(54.8%) point at a judgment we hold, so **45.2% of resolved citations point at
judgments we never sampled** and are unretrievable by construction. The Track B
recall figures in the report are a lower bound for that reason, and lifting the
per-year cap is the first item on the roadmap.

---

## How to run — building the stores

Nothing under `data/`, `index/` or `runs/` is committed — they are build outputs
and `.gitignore` keeps them out. The sequence below rebuilds all of it from the
public bucket. Run it from the repository root.

```bash
export PYTHONIOENCODING=utf-8

# 1. metadata -> the universe (37,210 nodes) and the sampled manifest (13,942)
python -m nyaya.metadata all

# 2. the PDFs: ~7.7 GB over 24 threads
python -m nyaya.ingest download

# 3. PDF -> cleaned, zoned text; writes judgments.jsonl and extract_log.csv
python -m nyaya.ingest build

# 4. S3, the zoned positional index (and B0's flat baseline index)
python -m nyaya.index.positional build --both

# 5. S1, the paragraph store
python -m nyaya.index.paragraphs --build

# 6. S2, the parametric index
python -m nyaya.index.parametric build

# 7. S4, the citation graph, its treatment labels, and authority
python -m nyaya.authority.citations build
python -m nyaya.authority.treatment label
python -m nyaya.authority.pagerank build                         # plain, for S3
python -m nyaya.authority.pagerank build --weighting treatment   # treatment-aware, for S4

# 8. S5, champion lists and the two-tier index
python -m nyaya.index.tiers build

# check
python -m pytest tests -q
python -m nyaya.cli --smoke
```

**Realistic timings**, on the machine we built on (12 logical cores, consumer
SSD):

| Stage | Cost |
|---|---|
| `nyaya.metadata all` | fetches and joins **47 MB** of per-year parquet, then draws the sample; network-bound, a few minutes |
| `nyaya.ingest download` | **≈7.7 GB** of PDFs (7.69 GB for the 13,942-document manifest, measured on disk). Entirely network-bound: our build ran at **~70 docs/s** over 24 threads (≈3.5 min); a re-measurement for this README on a slower domestic link managed **2.0 docs/s** (300 PDFs in 152 s), i.e. ~2 h for the manifest. Budget by your bandwidth, not by our clock. |
| `nyaya.ingest build` | **~55–65 docs/s** over **11 worker processes** (`cpu_count − 1`), ≈4 min for 13,942 documents. A 1,100-document re-measurement gave 55.5 docs/s. |
| `nyaya.index.positional build` | SPIMI invert + merge in **604 s**, 24 blocks of 600 documents |
| S1 / S2 / S4 / S5 | minutes each, not tens of minutes; the treatment-labelling pass is **32.5 s** over **76,118** edges and PageRank converges in **21 iterations** (0.06 s of power iteration over 37,210 nodes) |

The build is resumable: `download` skips files already on disk, and every stage
reads its input from disk rather than from the stage before it.

**The build is not bit-deterministic.** `SAMPLE_SEED = 358` fixes *which*
judgments are indexed, so a rebuild indexes the same corpus — but two builds from
identical inputs gave vocabularies differing by **36 and 56 types**. Treat any
vocabulary delta under about 100 types as noise.

---

## How to run — querying

Every command in this section was executed against the built stores while writing
this README. All of them go through the same `nyaya.search.search()` the
evaluation harness measures, so nothing shown here is a separate code path.

```bash
export PYTHONIOENCODING=utf-8
```

**Ranked search.** `--mode` is one of `B0 S1 S2 S2p S3 S4`; `S4` is the full
system and the default.

```bash
python -m nyaya.cli --mode S4 "right to privacy"
python -m nyaya.cli --mode S4 "right to privacy" -k 3 --snippets
```

```
mode S4  ·  free-text  ·  686 candidates ·  686 scored  ·  1147 ms  ·  scorer bm25f
  S5 tier 1  -  tier 1 held 686 >= k=3 documents

  1. [0.8502] JUSTICE K S PUTTASWAMY (RETD.), AND ANR.  versus UNION OF INDIA AN
      2017  ·  bench 9  ·  2017 INSC 801  ·  [2017] 10 S.C.R. 569  ·  reference answered
```

**The plan, with the candidate count after every step** — the df-ordering rule
made visible:

```bash
python -m nyaya.cli --explain-plan 'privacy /p "fundamental right" AND bench:>=5 year:>=2015'
```

```
  AND over 3 operands   est 126   -> 9   running 9
      (order: PROX_P[126] -> FILTER(bench>=5)[1154] -> FILTER(year>=2015)[2139])
  result: 9 documents  ·  merge_entries 3375  ·  filters_applied 2
```

**Why this result** — the full arithmetic behind one hit, with a check that the
parts reproduce the net score:

```bash
python -m nyaya.cli --why 1 "right to privacy"
```

**A real postings list** — doc ids, per-zone term frequencies, positions, skip
stride:

```bash
python -m nyaya.cli --postings arbitration
```

```
  df            883 documents of 13,878
  idf           2.7542
  positions     33,653
  skip stride   29  (~sqrt(df))
```

**The merge, traced:**

```bash
python -m nyaya.cli --merge 'murder AND intention' --trace
```

**The analyzer** — the token stream documents *and* queries both go through,
showing statute canonicalisation and act inheritance:

```bash
python -m nyaya.cli --analyze "Section 302 IPC read with Section 34 - common intention."
```

```
canonicalised :  ipc§302  read with  ipc§34  - common intention.
  [  0:15 ] sec_act   inherited=False 'Section 302 IPC' -> ipc§302
  [ 26:36 ] sec_only  inherited=True  'Section 34'      -> ipc§34
```

**The parse tree, filters and query-time expansions**, including the IPC↔BNS
crosswalk:

```bash
python -m nyaya.cli --parse 'ipc 302 AND "common intention" year:>=2000'
```

```
expansions   :  (query time only - never written to the index)
   ipc§302 OR bns§103
     ! IPC 302 is BNS 103(1); BNS 103(2) is a new mob-lynching offence with no IPC counterpart.
```

**The extractive Precedent Brief**, every sentence carrying the paragraph id it
came from:

```bash
python -m nyaya.cli --brief "right to privacy"
```

**The treatment classifier against its gold labels:**

```bash
python -m nyaya.cli --treatment-audit --only-errors
```

**The end-to-end smoke test** `docs/CONTRACTS.md` requires before every push —
six systems × six query shapes, including a malformed query:

```bash
python -m nyaya.cli --smoke          # prints PASS
```

**The optional single-page UI.** It earns no marks and exists so that the "why
this result" panel can be *seen* without reading terminal output:

```bash
streamlit run app.py                 # http://localhost:8501
```

### Query language

| Form | Meaning |
|------|---------|
| `murder bail` | free text; fast path through the tier-1 champion lists |
| `"common intention"` | exact phrase |
| `privacy /5 dignity` | within five words |
| `privacy /s dignity` | same sentence |
| `privacy /p dignity` | same paragraph |
| `bail!` | root expansion by dictionary prefix |
| `title:(privacy)` | restrict to one zone (`title parties bench statutes order body`) |
| `year:>=2000`, `bench:>=5` | parametric filters on real metadata |
| `judge:chandrachud`, `act:IPC`, `type:criminal` | parametric filters |
| `X AND Y`, `X OR Y`, `X NOT Y` | Boolean; `NOT` is a set difference, never a bare negation |

### Inspecting the stores

```bash
python -m nyaya.metadata stats                   # universe, manifest, parametric fields
python -m nyaya.ingest stats                     # extraction outcomes, bench distribution
python -m nyaya.index.stats --no-growth          # corpus + index statistics, Heaps, Zipf
python -m nyaya.index.positional top --n 5       # most frequent terms by df
python -m nyaya.index.positional prefix bail     # the dictionary scan behind `bail!`
python -m nyaya.index.paragraphs --stats         # S1
python -m nyaya.index.parametric stats           # S2
python -m nyaya.authority.citations stats        # S4 extraction and resolution
python -m nyaya.authority.treatment stats        # the label distribution
python -m nyaya.index.tiers stats                # S5 champion-list coverage
```

---

## How to reproduce the evaluation

Each command writes its result into `runs/` and prints it. Everything below was
run while writing this README and reproduced the figures in the report.

```bash
export PYTHONIOENCODING=utf-8

# Track A — 20 judged test queries, six systems, sign test + bootstrap CI
python -m nyaya.eval.run

# operator correctness against an independent naive scan
python -m nyaya.eval.operators --brute-force

# Track B — citation recovery, no human judge
python -m nyaya.eval.citation_recovery run --n 198
python -m nyaya.eval.citation_recovery report        # re-print without recomputing

# Track C — latency, index elimination, two-tier fidelity
python -m nyaya.eval.efficiency run
python -m nyaya.eval.efficiency report
python -m nyaya.eval.efficiency stores               # on-disk size of all five stores

# ablations: one analyser flag off at a time
python -m nyaya.rank.ablate --run

# the zone-weight sweep, and the test split it was NOT fitted on
python -m nyaya.rank.tune
python -m nyaya.rank.tune --report-test

# df-ordered Boolean planning, over eight AND chains
python -m nyaya.query.planner --compare-suite

# the two-tier index: gain and fidelity side by side
python -m nyaya.index.tiers evaluate

# authority: the top of the graph, and what treatment weighting changed
python -m nyaya.authority.pagerank top
python -m nyaya.authority.pagerank compare
python -m nyaya.authority.pagerank validate          # our power iteration vs networkx
```

Three notes, because an evaluation command that quietly measures something else
is worse than no command at all:

- **`--n 198` on the Track B line is not decoration.** `run` reuses
  `runs/citation_recovery_queries.json` when it holds at least *n* queries and
  **re-samples otherwise**; the published figures are the first 198 of the saved
  set. Asking for 200 gives a slightly different sample, with R@10 within 0.002
  of the published table — correct, but not the same run.
- **The `report` subcommands re-print a finished run without recomputing it.** If
  the derived stores have been rebuilt since, run the matching `run` first or you
  will read stale numbers off a fresh index.
- **Latency is machine- and cache-dependent, so we do not quote a speed-up
  ratio.** Three runs of identical code gave monotonically falling wall clock as
  the OS page cache warmed. We have seen warm p50 between **3.7 and 7.3 ms** and
  the two-tier wall-clock gain between **3.2× and 5.0×** on unchanged code, which
  is why the report leads with *documents scored* — an exact count — rather than
  with milliseconds. The algorithmic columns are identical every time.

**The df-ordering result**, as one example of what these commands print.
`python -m nyaya.query.planner --compare-suite` replays eight AND chains under
three operand orders:

```
  TOTAL                                                      69,527   69,527  145,921    2.10x
  identical answers under all three policies: True
  best    327.24x  tax AND sale AND interstate AND trade
  worst     1.00x  bail AND anticipatory AND discretion
```

145,921 → 69,527 postings entries compared, **2.10× less work for identical
answers**. The ratio is computed on the totals, not as a mean of per-query
ratios, so the 327× outlier cannot carry it.

---

## Corpus and index statistics

All of these come from `python -m nyaya.index.stats --no-growth`,
`python -m nyaya.ingest stats`, `python -m nyaya.index.paragraphs --stats` and
`python -m nyaya.eval.efficiency stores`.

| | |
|---|---|
| judgments in the release (English rows) | 42,371 |
| distinct judgments after de-duplicating on `path` | **37,210** — the citation-graph nodes |
| manifest (stratified sample, ≤200/year, seed 358) | 13,942 |
| **documents indexed** | **13,878** (99.54% of the manifest) |
| mean length | **5,842 words**, **31.5 paragraphs** per judgment (median 25) |
| paragraphs in the paragraph store S1 | **437,654** (body 417,184 + order 20,470) |
| sentences | 3,155,187 |
| vocabulary | **544,787** terms (68.6% appear in exactly one document) |
| postings — (term, document) pairs | **11,943,968** |
| token positions | **83,685,567** |
| share of indexed tokens in the `body` zone | 84.8% |
| `postings.bin` | 687.3 MB |
| **S3 index on disk** (postings + dictionary + docmeta) | **717.9 MB** |
| **all five stores** | **1.9 GB** (index-to-text ratio 1.29) |
| cold start | 1.2–1.5 s to memory-map the stores, ~1 s for the first query |

Heaps `|V| = 3.33 · T^0.653` (R² = 0.9957) and Zipf `s = 0.9825` (R² = 0.9930)
over ranks 10–1,000 — a check on the tokenizer, not a result.

One number needs disambiguating, because two stores count paragraphs differently.
**S1 holds 437,654 paragraphs** — the body and order paragraphs that snippets,
`/p` and the Precedent Brief quote from. The positional index's `docmeta` records
**489,627 paragraph starts**, because it records them for all six zones including
the short `title`, `parties`, `bench` and `statutes` zones; that is the figure §1
of the report quotes. Both are measured; they are not the same quantity.

---

## What works, what is planned

### Works — runs live on the real corpus, with a test and a printable intermediate

| Component | What it does |
|---|---|
| **Zoned positional index (S3)** | SPIMI build, six zones, per-zone term frequencies, every token position, a skip pointer every ⌈√df⌉ entries, memory-mapped `int32` postings |
| **Parametric index (S2)** | year, bench size, judges, acts, case type, disposal; `select(field, op, value) → sorted doc ids`, so `bench:>=5` is a real constraint and not a keyword match |
| **Paragraph store (S1)** | numbered paragraphs, the unit of evidence for snippets, `/p` and the brief |
| **Westlaw-style query language** | phrases, `/n` `/s` `/p`, `bail!` root expansion, zone restriction, parametric filters, `AND`/`OR`/`NOT` |
| **Statute canonicalisation** | *Section 302 IPC* / *S. 302 of the Indian Penal Code, 1860* / *u/s 302* → `ipc§302`, with act inheritance across a sentence, plus a **verified IPC↔BNS crosswalk** so a 1960 judgment and a 2025 one answer the same query |
| **df-ordered Boolean planner** | rarest operand first, selective filters folded in first, skip-pointer intersections, index elimination — measured at **2.10× less work for identical answers** over eight AND chains |
| **Three scorers, switchable** | lnc.ltc cosine, weighted-zone, BM25F (per-zone `b`, saturation after the zones are combined) |
| **Proximity** | `/n`, `/s`, `/p` and phrases as one two-pointer walk over two position lists; no cross-product is ever materialised |
| **Treatment-aware authority** | citation graph over all 37,210 nodes, edges labelled followed / distinguished / mentioned / overruled, our own power iteration (λ = 0.85), `g(d)` combining PageRank with bench size and a status penalty; **91 judgments demoted and badged with the citing paragraph** |
| **"Why this result"** | the full arithmetic of a net score, with a check that the parts sum to it |
| **Champion lists and the two-tier index (S5)** | tier 1 is 30.1% of the postings; wired into live search, with its fidelity printed beside its gain |
| **All three evaluation tracks** | judged queries with a sign test and a bootstrap CI; citation recovery with an authority-only control; efficiency and operator correctness |
| **Extractive Precedent Brief** | paragraph-level BM25 over the same index picks ≤8 paragraphs from the top 5 judgments; every sentence is **verbatim** and carries its paragraph id |

### Planned — written, or scoped, but not done

| Not done | Honest status |
|---|---|
| **Abstractive Precedent Brief** | The composer and its prompt are written and complete, and **it has never been executed** — there is no LLM API key in this environment (`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are both unset and the `anthropic` SDK is not installed). **The brief that ships is extractive, with every sentence verbatim from a cited paragraph**, and the output says so. We would rather ship a working extractive feature with a stated boundary than claim an LLM result we cannot reproduce. |
| **OCR for scanned pre-1990 judgments** | 53 documents were dropped as scans, and the older half of the corpus is typeset paper of varying quality. No OCR pass exists. |
| **Issue-level treatment** | A judgment is overruled *on a point*, not wholesale. Our treatment label is per citation edge, not per issue. |
| **Learning to rank** | `α` and `β` are hand-set; 20 judged queries cannot fit them, and the tuning section of the report shows what overfitting five queries looks like. |
| **High Court judgments, statutes as documents** | Precedent in practice runs through both. The graph is already the right shape for it. |
| **Hindi and the other languages in the release** | We index the English rows only; de-duplication discards the other-language copies rather than indexing them. |

---

## Known limitations

These are measured, not hedged, and each is reproducible with a command.

- **`/p` is a ~110-word window on most documents.** Only **28.2%** of judgments
  (3,911 of 13,878) print their own paragraph numbers; PyMuPDF returns no blank
  lines for the rest, so `nyaya.zones` segments those into ~110-word sentence
  windows. `/p` therefore means "within the court's own numbered paragraph" on
  28.2% of the corpus and "within the same ~110-word window" on the other 71.8%.
  That is a property of the source PDFs rather than a design choice, and
  `python -m nyaya.index.paragraphs --stats` prints the share.
- **184 documents report bench 1 because the re-parse rejected an unvalidatable
  bench line.** Re-parsing the CORAM line moved 488 documents — 258 up, but 230
  down, **184 of them to 1**, where the parser could not validate what it read
  and fell back to the metadata's single *authoring* judge. The cost is
  quantifiable, which is what makes this a limitation rather than an untidiness:
  with `bench_norm = min(bench,13)/13` and `G_BENCH_WEIGHT = 0.25`, **each
  missing seat costs 0.0192 of `g(d)`**, so a five-judge bench read as 1
  understates `g` by **0.077** — against an **entire overruled penalty of 0.289**
  on *Pune Municipal Corporation*. One mis-parsed Constitution Bench moves
  authority by the same order as being overruled. *Maneka Gandhi* (1978) is the
  visible case: seven judges recorded as 1, because its bench line OCRs as
  `M. H. BEG, c. J.` and the parser never reaches the other six.
- **67 of the 91 overruled flags rest on a single citation edge.** One edge
  against a hundred that say "followed" is more likely a classifier error than
  overruled law. Those rows are marked `(!)` rather than filtered, because a
  share threshold would also discard the genuine single-overruling cases
  (*Sankari Prasad* is 1 of 73). `python -m nyaya.authority.pagerank compare`
  prints the count.
- **The treatment classifier cannot read a denial.** It matches surface cues, so
  "we are *unable to hold* that *X* stands overruled" is mis-labelled. It is
  92.5% accurate on 40 hand-labelled edges (37/40), and **all three errors lose
  upward** — into a label that demotes, which is the worst possible direction.
  Negation and modality scope is the next item on the roadmap.
- **The index build is not bit-deterministic.** Two builds from identical inputs
  gave vocabularies differing by **36 and 56 types**, so an ablation's vocabulary
  delta under about 100 types is noise rather than signal.
- **The brief's tf-idf cosine verifier is a topicality check, not an entailment
  check.** 36 real extractive sentences score 0.9924–1.0000 against the paragraph
  they cite; 18 deliberate fabrications score 0.0866–0.8451, and at the shipped
  threshold of 0.20 it catches **4 of the 18, with 0 false alarms**. It reliably
  catches the *mechanical* failures and **misses a confident on-topic
  falsehood**: *"The right to privacy was expressly held to be subordinate to the
  right to property"* scores 0.8451 and passes. We did not raise the threshold to
  flatter the number, because that would reject a genuine paraphrase the moment
  the abstractive composer is enabled.
- **Some pre-1990 PDFs are page-range scans of bound SCR volumes.** A file is a
  page range cut out of a volume, so a judgment may open with the tail of the
  previous one — and **18 documents exceed 100,000 tokens because they are
  effectively whole volumes** (the longest is 481,654 tokens against a median of
  4,161).
- **Recall is capped by the sample.** 45.2% of resolved citations point at
  judgments we never sampled — see
  [Corpus policy](#corpus-policy-stated-honestly).
- **One judge, one assessor pool.** The relevance judgments come from one person
  on the team: blind to system identity, but not independent, and 20 test queries
  is a small set. We answered that with a sign test and a 10,000-iteration
  bootstrap rather than with a disclaimer.

---

## Tests

```bash
python -m pytest tests -q          # 1084 passed
python -m nyaya.cli --smoke        # PASS
```

**1,084 tests.** Every component has at least one test a reviewer can read, and
the operator suite checks `/n`, `/s`, `/p`, phrases and Boolean merges against an
independent naive scan over 1,200 judgments sampled across 1950–2025. Both of the
commands above must pass before any push — `docs/CONTRACTS.md` makes that the
rule.

---

## Repository map

```
nyaya/              the system: ingest, analyze, index/, query/, rank/, authority/, brief/, eval/
  cli.py            everything the demo video records
  config.py         every tunable; an ablation is a flag, not a code change
  contracts.py      the interfaces frozen at kick-off
app.py              the optional Streamlit page
tests/              1,084 tests
eval/queries/       the hand-written query sets and the graded qrels
docs/CONTRACTS.md   the team contract: ownership map, corpus policy, stores, git conventions
report/             report.md -> report.html -> NyayaTrace-report.pdf
video/              the Remotion demo film: src/ (scenes and measured data), script/
data/ index/ runs/  build outputs, git-ignored, regenerated by the commands above
```

- **`docs/CONTRACTS.md`** is the single source of truth for who owns which file,
  the corpus policy, the five stores and the commit conventions.
- **`report/`** holds the write-up. Rebuild it with
  `python report/build_html.py && python report/build_pdf.py`.
- **`video/`** holds the demo film. `npm install` then `npm run build` renders it;
  `npx remotion studio` opens it interactively.

## Licence and credit

The code in this repository is coursework. The corpus is **Indian Supreme Court
Judgments** from the **AWS Open Data Registry**, maintained by **Dattam Labs**,
used under **CC-BY-4.0** —
<https://registry.opendata.aws/indian-supreme-court-judgments>.
