# NyayaTrace — voice-over script

**Film length 7:27 (13,410 frames at 30 fps).** Read at roughly **150 words per
minute**, which is 2.5 words a second; every segment below is written to land a
comfortably under its slot — no segment fills more than 90% of its gap, so
nobody has to rush. Word counts are exact.

**Total narration: 915 words ≈ 6:06 of speech inside 7:27 of film.** The
difference is deliberate: the title, the transitions and the last beat of each
animation breathe without a voice over them.

How to read the headers:

```
## <nn> · <segment id>            <m:ss – m:ss>   <kind>
Speaker: <name>                   <n words · ~n.ns>
```

`ANIM` = read over a Remotion animation · `SCREEN` = read over the live screen
recording · `CAM` = spoken to camera, on screen.

Anything in **[brackets]** is a figure that is still a placeholder in
`video/src/data.ts`. **Do not read a placeholder aloud as if it were a result.**
Record those lines only after the measured numbers are in.

---

## 01 · title  0:00 – 0:17  ANIM
**Speaker: Suvrat** · 31 words · ~12.4s

> NyayaTrace. A search engine for Indian Supreme Court precedent, built for the
> mid-term hackathon in Information Retrieval, on track T6. Rakshit, Suvrat and
> Sumedha. Fourteen thousand judgments, and one unusual idea.

---

## 02 · cam-intro  0:17 – 0:30  CAM
**Speaker: Rakshit** (all three in frame if the shot allows) · 23 words · ~9.2s

> Hi — we're Rakshit, Suvrat and Sumedha. We built a precedent search engine
> over Supreme Court judgments. First, the problem it exists to solve.

---

## 03 · three-ways  0:30 – 0:50  ANIM
**Speaker: Sumedha** · 44 words · ~17.6s

> Lawyers write the same provision three ways. Section 302 IPC. S. 302 of the
> Indian Penal Code, 1860. u slash s 302. Those share almost no terms, so a plain
> tokenizer files one provision in three places. We collapse all three into one
> token.

---

## 04 · anatomy  0:50 – 1:06  ANIM
**Speaker: Rakshit** · 35 words · ~14.0s

> A judgment is not a bag of words either. It has a title, parties, a bench,
> the statutes it turns on, an order, and a very long body. The holding lives
> in the smallest part.

---

## 05 · overruled-trap  1:06 – 1:23  ANIM
**Speaker: Suvrat** · 37 words · ~14.8s

> And here is the trap. A case can be cited hundreds of times and still be
> dangerous to cite, because one later bench overruled it. Every earlier
> citation still points at it. Popularity and reliability come apart.

---

## 06 · cam-t6  1:23 – 1:33  CAM
**Speaker: Suvrat** · 22 words · ~8.8s

> That is why this is a vertical problem, not a web one. The structure —
> zones, metadata, citations, authority — is the whole advantage.

---

## 07 · demo-freetext  1:33 – 1:51  SCREEN
**Speaker: Sumedha** · 37 words · ~14.8s

> So, live. A plain free-text query against the real index. Ten ranked
> judgments, each with its year and bench size, and a badge where the citation
> graph has something to say about whether the case still stands.

---

## 08 · demo-operators  1:51 – 2:09  SCREEN
**Speaker: Sumedha** · 38 words · ~15.2s

> Now the operator language. Privacy within the same paragraph as the phrase
> "fundamental right", on a bench of five or more, since 2015. You can see the
> parse tree, and the plan the engine chose to execute it.

---

## 09 · demo-why  2:09 – 2:24  SCREEN
**Speaker: Sumedha** · 31 words · ~12.4s

> And every result can explain itself. Which zones the score came from, which
> terms carried it, the authority term, the penalty, and the sentence in another
> judgment that produced the badge.

---

## 10 · pipeline-map  2:24 – 2:38  ANIM
**Speaker: Suvrat** · 28 words · ~11.2s

> Here is the whole system. The top row builds the index, the middle row is
> what sits on disk, the third row is what happens when you type.

---

## 11 · cam-analyzer  2:38 – 2:49  CAM
**Speaker: Sumedha** · 24 words · ~9.6s

> I own the analyzer, the query language and the scorers. The rule I care
> about most: documents and queries go through the same function.

---

## 12 · statute-canon  2:49 – 3:05  ANIM
**Speaker: Sumedha** · 34 words · ~13.6s

> Fold and tokenise. Canonicalise statutes with a grammar over citation forms.
> Drop the words every judgment contains. Stem — except statute tokens, which
> are identifiers. And cross-walk the old Penal Code to the new Sanhita.

---

## 13 · analyze-parse  3:05 – 3:17  SCREEN
**Speaker: Sumedha** · 26 words · ~10.4s

> That function, printing its own output: every token with its zone, its
> offset, its sentence, its paragraph. And the parser, turning a query into a
> tree.

---

## 14 · cam-index  3:17 – 3:28  CAM
**Speaker: Rakshit** · 23 words · ~9.2s

> I built the ingest, the zoning, the index and the proximity operators. One
> posting says which document, how often per zone, and where.

---

## 15 · zoned-postings  3:28 – 3:45  ANIM
**Speaker: Rakshit** · 34 words · ~13.6s

> The dictionary points at an offset in one memory-mapped array of 32-bit
> integers. Each posting holds a per-zone term frequency and the full position
> list, and every square-root-of-df entries there is a skip pointer.

---

## 16 · proximity-walk  3:45 – 4:01  ANIM
**Speaker: Rakshit** · 31 words · ~12.4s

> Proximity is one merge with two pointers. Compare offsets and you have
> “within n words”. Compare sentence ids, same sentence. Compare paragraph
> ids, same paragraph. A phrase is the tightest case.

---

## 17 · postings-plan  4:01 – 4:14  SCREEN
**Speaker: Rakshit** · 25 words · ~10.0s

> Printed from the real index: the postings for one term, the skip stride, and
> the merge walking step by step. Nothing here is a mock-up.

---

## 18 · boolean-plan  4:14 – 4:27  ANIM
**Speaker: Suvrat** · 29 words · ~11.6s

> The planner never executes a conjunction in the order you typed it. It sorts
> the operands by document frequency and starts with the rarest — an
> intersection can only shrink.

---

## 19 · bm25f  4:27 – 4:42  ANIM
**Speaker: Sumedha** · 26 words · ~10.4s

> Then ranking. Raw term frequency runs off the top of the chart; BM25F
> saturates. The zones are summed before saturation, each with its own length
> normaliser.

---

## 20 · cam-authority  4:42 – 4:53  CAM
**Speaker: Suvrat** · 20 words · ~8.0s

> My half is the citation graph, the authority score and the planner. A
> citation is a verdict, not a vote.

---

## 21 · citation-graph  4:53 – 5:13  ANIM
**Speaker: Suvrat** · 42 words · ~16.8s

> A cue-phrase classifier labels every edge — followed, distinguished, mentioned,
> overruled — and that label becomes the edge weight. We run our own weighted
> PageRank over it. An overruling edge transfers nothing, and the case it points
> at takes a status penalty on top.

---

## 22 · net-score  5:13 – 5:26  ANIM
**Speaker: Suvrat** · 23 words · ~9.2s

> Text relevance, static authority and proximity meet in one score, one weight
> each. Turn a weight off and you have an earlier system.

---

## 23 · cam-eval  5:26 – 5:36  CAM
**Speaker: Rakshit** · 22 words · ~8.8s

> I ran the evaluation. Six systems, each one change apart, plus two tracks
> where the answer key needs no human at all.

---

## 24 · eval-run  5:36 – 5:50  SCREEN
**Speakers: Rakshit then Suvrat** · 28 words · ~11.2s

> **Rakshit (first ~7s, 14 words):** The harness, running all six systems over the
> same query set and writing the metrics out.
>
> **Suvrat (last ~7s, 14 words):** And the operator suite, checking every operator
> against a brute-force scan of the raw text.

---

## 25 · eval-ladder  5:50 – 6:08  ANIM
**Speaker: Rakshit** · 36 words · ~14.4s
*(Hold this take until the measured numbers are in — every figure on screen is a
placeholder.)*

> Baseline is flat tf-idf. Each step adds exactly one idea: canonicalisation,
> zone weights, BM25F, authority, then treatment. Read the shape rather than any
> one bar — when a step does not move the metric, we say so.

---

## 26 · operator-speed  6:08 – 6:21  ANIM
**Speaker: Suvrat** · 26 words · ~10.4s

> Two things need no judge. Operator correctness, which must match the brute-force
> scan exactly. And speed: champion lists and a two-tier index, against scoring
> every posting.

---

## 27 · limitation  6:21 – 6:39  ANIM
**Speaker: Rakshit** · 40 words · ~16.0s

> Now what it gets wrong. “We are unable to hold that the decision stands
> overruled.” The cue is there, the denial is not, so the classifier marks a
> case that was upheld. The fix is scope, not a longer list.

---

## 28 · limitation-live  6:39 – 6:49  SCREEN
**Speaker: Rakshit** · 22 words · ~8.8s

> The audit output — predicted label, gold label, and the sentence that fooled
> it. We ship the flag rather than hide it.

---

## 29 · precedent-brief  6:49 – 7:06  ANIM
**Speaker: Sumedha** · 37 words · ~14.8s

> Last thing. Paragraph-level BM25 over the same index picks the evidence; the
> model may not write a sentence without a paragraph id; and a cosine check
> re-reads each sentence against the paragraph it claims. No embeddings anywhere.

---

## 30 · cam-close  7:06 – 7:18  CAM
**Speaker: Sumedha** · 26 words · ~10.4s

> What is genuinely new here is treatment-aware authority — ranking a precedent
> by how later courts treated it, not how often they mentioned it. Thanks for
> watching.

---

## 31 · outro  7:18 – 7:27  ANIM
**Speaker: Suvrat** · 15 words · ~6.0s
*(Optional — the card also works silent under the music bed.)*

> NyayaTrace. Fourteen thousand judgments indexed, the full forty-two thousand in
> the citation graph. Thank you.

---

## Word count by speaker

| Speaker | Words | Share | Segments |
|---------|-------|-------|----------|
| Rakshit | 305 | 33% | 02, 04, 14, 15, 16, 17, 23, 24a, 25, 27, 28 |
| Sumedha | 323 | 35% | 03, 07, 08, 09, 11, 12, 13, 19, 29, 30 |
| Suvrat  | 287 | 31% | 01, 05, 06, 10, 18, 20, 21, 22, 24b, 26, 31 |
| **Total** | **915** | | 31 segments | **991** | | 31 segments |

Suvrat carries fewer narration words because he also assembles the written
report; the split was agreed with that in mind.

## Recording notes

* Record each segment as its own take, named `vo-<segment id>.wav`. Do not record
  the whole script in one pass — the slot lengths are fixed and a single long
  take cannot be nudged.
* Leave about half a second of room at the head and tail of every take.
* The `CAM` segments are recorded **with** the webcam take, not separately.
* Numbers in **[brackets]** are placeholders. Segments 25, 26 and 28 state
  measured results; record their final takes only after `video/src/data.ts` is
  updated and `MEASURED` is flipped to `true`.
* Pronunciation: *Nyaya* is "NYAH-yuh". *lnc.ltc* is read "L-N-C dot L-T-C".
  *nDCG* is "N-D-C-G". *BM25F* is "B-M twenty-five F". *g(d)* is "g of d".
