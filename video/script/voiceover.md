# NyayaTrace — voice-over script

**Film length 7:27 (13,410 frames at 30 fps).** Read at roughly **150 words per
minute**, which is 2.5 words a second; every segment below is written to land a
little under its slot so nobody has to rush. Word counts are exact.

**Total narration: 1,008 words ≈ 6:43 of speech inside 7:27 of film.** The
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
**Speaker: Suvrat** · 34 words · ~13.6s

> NyayaTrace. A search engine for Indian Supreme Court precedent, built for the
> mid-term hackathon in Information Retrieval, on track T6. Rakshit, Suvrat and
> Sumedha. Fourteen thousand judgments, and one unusual idea.

---

## 02 · cam-intro  0:17 – 0:30  CAM
**Speaker: Rakshit** (all three in frame if the shot allows) · 29 words · ~11.6s

> Hi — we're Rakshit, Suvrat and Sumedha. We built a precedent search engine
> over Supreme Court judgments. Before we show it running, here is the problem
> it exists to solve.

---

## 03 · three-ways  0:30 – 0:50  ANIM
**Speaker: Sumedha** · 46 words · ~18.4s

> Lawyers write the same provision three different ways. Section 302 IPC. S. 302
> of the Indian Penal Code, 1860. u slash s 302. To a general tokenizer those
> share almost no terms, so one provision ends up in three places in the index.
> We collapse all three to one token.

---

## 04 · anatomy  0:50 – 1:06  ANIM
**Speaker: Rakshit** · 37 words · ~14.8s

> A judgment is not a bag of words either. It has a title, parties, a bench, the
> statutes it turns on, an operative order, and a very long body. The holding
> lives in the smallest part.

---

## 05 · overruled-trap  1:06 – 1:23  ANIM
**Speaker: Suvrat** · 39 words · ~15.6s

> And here is the trap. A case can be cited hundreds of times and still be
> dangerous to cite, because one later bench overruled it. Every earlier
> citation still points at it. Popularity and reliability come apart.

---

## 06 · cam-t6  1:23 – 1:33  CAM
**Speaker: Suvrat** · 24 words · ~9.6s

> That is why this is a vertical search problem, not a web one. The structure —
> zones, metadata, citations, authority — is the whole advantage.

---

## 07 · demo-freetext  1:33 – 1:51  SCREEN
**Speaker: Sumedha** · 41 words · ~16.4s

> So, live. A plain free-text query against the real index. Ten ranked
> judgments, each with its year and bench size, and a badge where the citation
> graph has something to say about whether the case still stands.

---

## 08 · demo-operators  1:51 – 2:09  SCREEN
**Speaker: Sumedha** · 42 words · ~16.8s

> Now the operator language. Privacy within the same paragraph as the phrase
> "fundamental right", on a bench of five or more, since 2015. You can see the
> parse tree, and the plan the engine chose to execute it.

---

## 09 · demo-why  2:09 – 2:24  SCREEN
**Speaker: Sumedha** · 35 words · ~14.0s

> And every result can explain itself. Which zones the score came from, which
> terms carried it, the authority term, the penalty, and the sentence in another
> judgment that produced the badge.

---

## 10 · pipeline-map  2:24 – 2:38  ANIM
**Speaker: Suvrat** · 33 words · ~13.2s

> Here is the whole system. The top row builds the index, the middle row is what
> sits on disk, the third row is what happens when you type. Colour is who owns
> what.

---

## 11 · cam-analyzer  2:38 – 2:49  CAM
**Speaker: Sumedha** · 26 words · ~10.4s

> I own the analyzer, the query language and the scorers. The rule I care about
> most: documents and queries go through exactly the same function.

---

## 12 · statute-canon  2:49 – 3:05  ANIM
**Speaker: Sumedha** · 37 words · ~14.8s

> Fold and tokenise. Canonicalise statutes with a grammar over citation forms.
> Drop the words every judgment contains. Stem — except statute tokens, which
> are identifiers. And at query time, cross-walk the old Penal Code to the new
> Sanhita.

---

## 13 · analyze-parse  3:05 – 3:17  SCREEN
**Speaker: Sumedha** · 28 words · ~11.2s

> This is that function printing its own output: every token with its zone, its
> offset, its sentence and its paragraph. And the parser, turning a query into a
> tree.

---

## 14 · cam-index  3:17 – 3:28  CAM
**Speaker: Rakshit** · 26 words · ~10.4s

> I built the ingest, the zoning and the index itself, plus the proximity
> operators. One posting carries three things: which document, how often per
> zone, and exactly where.

---

## 15 · zoned-postings  3:28 – 3:45  ANIM
**Speaker: Rakshit** · 39 words · ~15.6s

> The dictionary points at an offset in one memory-mapped array of 32-bit
> integers. Each posting holds a per-zone term frequency and the full position
> list, and every square-root-of-df entries there is a skip pointer.

---

## 16 · proximity-walk  3:45 – 4:01  ANIM
**Speaker: Rakshit** · 38 words · ~15.2s

> Proximity is then one merge with two pointers. Compare offsets and you have
> "within n words". Compare sentence ids and you have same sentence. Compare
> paragraph ids and you have same paragraph. A phrase is just the tightest case.

---

## 17 · postings-plan  4:01 – 4:14  SCREEN
**Speaker: Rakshit** · 30 words · ~12.0s

> Printed from the real index: the postings for one term, the skip stride, and
> the merge walking step by step. Nothing here is a mock-up.

---

## 18 · boolean-plan  4:14 – 4:27  ANIM
**Speaker: Suvrat** · 30 words · ~12.0s

> The planner never executes a conjunction in the order you typed it. It sorts
> the operands by document frequency and starts with the rarest, because an
> intersection can only shrink.

---

## 19 · bm25f  4:27 – 4:42  ANIM
**Speaker: Sumedha** · 35 words · ~14.0s

> Then ranking. Raw term frequency runs off the top of the chart; BM25F
> saturates. The zones are summed before saturation, each with its own length
> normaliser, so a long body cannot out-shout a title.

---

## 20 · cam-authority  4:42 – 4:53  CAM
**Speaker: Suvrat** · 26 words · ~10.4s

> My half is the citation graph, the authority score and the planner. The idea
> the whole project rests on: a citation is a verdict, not a vote.

---

## 21 · citation-graph  4:53 – 5:13  ANIM
**Speaker: Suvrat** · 46 words · ~18.4s

> A cue-phrase classifier labels every edge — followed, distinguished, mentioned,
> overruled — and that label becomes the edge weight. We run our own weighted
> PageRank over it. An overruling edge transfers nothing, and the case it points
> at takes a status penalty on top.

---

## 22 · net-score  5:13 – 5:26  ANIM
**Speaker: Suvrat** · 31 words · ~12.4s

> Text relevance, static authority and proximity meet in one score, with one
> weight each. Turn a weight off and you have an earlier system — which is how
> the evaluation is built.

---

## 23 · cam-eval  5:26 – 5:36  CAM
**Speaker: Rakshit** · 25 words · ~10.0s

> I ran the evaluation. Six systems, each one change apart. Judged queries, plus
> two tracks where the answer key does not need a human at all.

---

## 24 · eval-run  5:36 – 5:50  SCREEN
**Speakers: Rakshit then Suvrat** · 33 words · ~13.2s

> **Rakshit (first ~7s, 17 words):** This is the harness running all six systems
> over the same query set and writing the metrics out.
>
> **Suvrat (last ~7s, 16 words):** And the operator suite, checking every
> operator against a brute-force scan of the raw text.

---

## 25 · eval-ladder  5:50 – 6:08  ANIM
**Speaker: Rakshit** · 41 words · ~16.4s
*(Hold this take until the measured numbers are in — every figure on screen is a
placeholder.)*

> Baseline is flat tf-idf. Each step adds exactly one idea: canonicalisation,
> zone weights, BM25F, authority, then treatment. Read the shape rather than any
> one bar — when a step does not move the metric, we say so.

---

## 26 · operator-speed  6:08 – 6:21  ANIM
**Speaker: Suvrat** · 30 words · ~12.0s

> Two things need no judge. Operator correctness, which must match the brute-force
> scan exactly. And speed: champion lists and a two-tier index, against scoring
> every posting.

---

## 27 · limitation  6:21 – 6:39  ANIM
**Speaker: Rakshit** · 41 words · ~16.4s

> Now what it gets wrong. "We are unable to hold that the decision stands
> overruled." The cue is there, the denial is not, so the classifier marks a case
> that was upheld. The fix is scope, not a longer phrase list.

---

## 28 · limitation-live  6:39 – 6:49  SCREEN
**Speaker: Rakshit** · 24 words · ~9.6s

> Here it is in the audit output — predicted label, gold label, and the sentence
> that fooled it. We ship the flag rather than hide it.

---

## 29 · precedent-brief  6:49 – 7:06  ANIM
**Speaker: Sumedha** · 39 words · ~15.6s

> Last thing. Paragraph-level BM25 over the same index picks the evidence; the
> model may not write a sentence without a paragraph id; and a cosine check
> re-reads each sentence against the paragraph it claims. No embeddings anywhere.

---

## 30 · cam-close  7:06 – 7:18  CAM
**Speaker: Sumedha** · 28 words · ~11.2s

> What is genuinely new here is treatment-aware authority — ranking a precedent
> by how later courts treated it, not how often they mentioned it. Thanks for
> watching.

---

## 31 · outro  7:18 – 7:27  ANIM
**Speaker: Suvrat** · 17 words · ~6.8s
*(Optional — the card also works silent under the music bed.)*

> NyayaTrace. Fourteen thousand judgments indexed, the full forty-two thousand in
> the citation graph. Thank you.

---

## Word count by speaker

| Speaker | Words | Share | Segments |
|---------|-------|-------|----------|
| Rakshit | 353 | 35% | 02, 04, 14, 15, 16, 17, 23, 24a, 25, 27, 28 |
| Sumedha | 357 | 35% | 03, 07, 08, 09, 11, 12, 13, 19, 29, 30 |
| Suvrat  | 298 | 30% | 01, 05, 06, 10, 18, 20, 21, 22, 24b, 26, 31 |
| **Total** | **1,008** | | 31 segments |

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
