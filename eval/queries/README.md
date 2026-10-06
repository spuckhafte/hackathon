# Judged query set (Track A)

25 queries over the indexed corpus, with 373 graded relevance judgments.
Curated by Sumedha (M3); the harness and the metrics are M1's.

## Files

| File | Format |
|---|---|
| `queries.jsonl` | one query per line: `{"qid","query","kind","split","intent"}` |
| `qrels.jsonl` | one judgment per line: `{"qid","doc_id","rel","title","year","pooled_from","why"}` |
| `qrels.txt` | the same judgments in TREC format: `qid 0 doc_id rel` |

`doc_id` is the index's document id, so a judgment joins straight onto a
`Result.doc_id` with no name matching.

## The split

**Five queries are held out for tuning** (`"split": "dev"`) — α, β and the zone
weights are fitted on those and on nothing else. **Twenty are the test set**
(`"split": "test"`) and must never be tuned on. The dev five span all three
query kinds on purpose (`q02` and `q07` free text, `q13` and `q17`
statute-anchored, `q22` operator), so a parameter fitted on them is not fitted
to one query shape.

## The three kinds

| `kind` | n | what it exercises |
|---|---|---|
| `free` | 10 | ranked retrieval over a natural legal question |
| `statute` | 8 | the statute canonicaliser: the query names a provision, and the matching documents write it a dozen different ways |
| `operator` | 7 | phrase, `/5`, `/s`, `/p`, root expansion, zone restriction, `NOT`, and the parametric filters |

`q25` is deliberately **filters only**: there is no text to score, so it must
rank by authority alone. A system that returns nothing for it is wrong.

## How the judgments were made

**Pooling.** For each query the pool is the top 10 of **two deliberately
different systems** — `B0` (flat tf-idf over the plain-tokenizer index) and `S4`
(the full system). Two systems that disagree a lot give a pool that is not
simply the ranking of the system being evaluated. Pool depth is 10 and the
pools run from 5 to 20 documents (mean 15). Documents outside the pool are
**unjudged**, which is the standard TREC pooling assumption and the standard
caveat: recall-based metrics are recall-at-pool, not true recall.

**Grades.**

| `rel` | meaning |
|---|---|
| 2 | a leading authority directly on the point of the query |
| 1 | on topic: applies the provision or doctrine the query asks about |
| 0 | pooled but off topic |

**Evidence used.** The case title and year; my own reading of which Indian
Supreme Court authorities are leading on each point; and an automatic check of
whether the query's decisive terms — including the *canonical statute token*,
e.g. `ipc§302`, `crpc§482`, `cpc§o7r11` — actually occur in the document. The
statute check is what makes the statute-anchored judgments objective rather than
impressionistic: a judgment graded 1 for `q11` demonstrably contains `ipc§302`,
`die` and `declar`.

**Honest limitations.**

* The judgments are one assessor's, not adjudicated between several. There is no
  inter-annotator agreement figure to quote.
* Grade 1 is assigned largely from the automatic term check, so it means "on
  topic" rather than "a lawyer would cite this". Grade 2 and grade 0 are both
  explicit human calls.
* The corpus is a stratified sample of at most 200 judgments per year, so a
  leading case can be absent entirely. Where that happened the query was kept
  only if the pool still contained real authority on the point.

## Sanity of the query set

Every query was checked against the live index before being committed, and the
set is not a set of lucky queries: the leading authority surfaces in the top few
for `q01` (Puttaswamy), `q02` (Gurbaksh Singh Sibbia), `q03` (Khushal Rao),
`q04` (Kesavananda, Minerva Mills), `q09` (Nilabati Behera, D.K. Basu), `q10`
(Daryao), `q13` (Gopalan, Kharak Singh, Maneka Gandhi), `q18` (Srihari
Hanumandas Totala) and `q21` (Kesavananda under a `bench:>=9` filter).

One candidate query was **dropped** because the corpus cannot answer it:
`title:(arbitration) AND act:ARB` returns nothing, because S.C.R. titles are
party names ("X versus Y") and a subject word such as "arbitration" never
appears in one. That is a property of the corpus worth knowing — the `title`
zone is a party-name field, not a subject line — and it is why `q23` restricts
to the synthetic `statutes` zone instead.

## Regenerating

```
python -m nyaya.eval.curate        # if M1's harness wraps it
```

The judgments are checked in, so re-running the pool after a ranking change will
not silently re-judge: new pooled documents arrive unjudged and must be graded
by hand before they count.
