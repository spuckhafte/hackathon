# NyayaTrace — speaker map

Every second of the 7:45 film: who is speaking, whether they are on camera, what
is on screen, and the line. Timings are generated from `video/src/timeline.ts`;
run `npm run timeline` in `video/` to reprint them after any change.

**The shape of the cut.** The film is a continuous narrated screen recording with
short concept animations cut into it. It is not a chaptered explainer and there
are no slides anywhere in it.

| | Time | Share | Count |
|---|---|---|---|
| **screen** — the real system being driven | 5:01 | **64.7%** | 12 clips |
| **animation** — concept inserts (13–16 s each) | 1:28 | **18.9%** | 6 inserts |
| **animation** — title and outro | 0:16 | 3.4% | 2 cards |
| **webcam** | 1:00 | **12.9%** | 6 clips |
| **total** | **7:45** | | 26 segments |

Legend — `SCREEN` live recording · `ANIM` Remotion insert · `CAM` webcam.

---

| # | Time | Kind | On camera | Voice | On screen | Line |
|---|------|------|-----------|-------|-----------|------|
| 01 | 0:00–0:09 | ANIM | — | **Rakshit** | Title card; a citation constellation drifts behind | "NyayaTrace. Treatment-aware precedent search for Indian case law. CSD358, Track T6." |
| 02 | 0:09–0:21 | CAM | **Rakshit** (all three in frame if the shot allows) | Rakshit | Webcam | "We're Rakshit, Suvrat and Sumedha. This is a search engine over Supreme Court precedent. Everything from here is the real system running — so let's run it." |
| 03 | 0:21–0:49 | SCREEN | — | **Sumedha** | `drive-freetext.mp4`: `--mode S4` then `--mode B0` on the same query. S4 ranks 1–4 are Puttaswamy 2018 (bench 5), Puttaswamy 2017 (**bench 9**), Electoral Bonds 2024, Puttaswamy 2015 | "A plain English question… eleven hundred and seventy-four candidates, and the top four are the Puttaswamy line — 2018, 2017, 2015 — then Electoral Bonds. Now the same query through the flat tf-idf baseline … never reaches the other two, because it reads a judgment as one bag of words, and" |
| 04 | 0:49–1:02 | ANIM | — | **Rakshit** | **Insert.** A page of running text separates into six zone bands; the real weights grow beside them; `bench` is marked filter-only | "a judgment is not a bag of words. It has a title, parties, a bench, statutes, an order and a long body. Bench is a filter, not evidence." |
| 05 | 1:02–1:30 | SCREEN | — | **Sumedha** | `drive-why.mp4`: `--why 1 --mode S4`, scrolled. Rank 1 is Puttaswamy 2017, bench 9, net 0.900210 | "Every result takes itself apart… Then the net score, with the arithmetic checked against itself. One tf of two hundred and thirty-eight for the whole document, because" |
| 06 | 1:30–1:45 | ANIM | — | **Sumedha** | **Insert.** Raw tf leaves the top of the plot while the BM25F curve saturates; k₁ marked; zones summed before saturation | "BM25F adds the zones together before it saturates, not after… A long body cannot out-shout a title." |
| 07 | 1:45–2:03 | SCREEN | — | **Sumedha** | `drive-ui.mp4`: the Streamlit page, parse expander open, mode flipped S4 → B0 | "Same function behind a page. The parse tree, the candidates, the time. A result card with its badge and a paragraph-cited snippet. Drop the mode to B0 and the list changes underneath me." |
| 08 | 2:03–2:12 | CAM | **Suvrat** | Suvrat | Webcam | "That's the vertical case in a line: the structure of a judgment is the whole advantage." |
| 09 | 2:12–2:44 | SCREEN | — | **Suvrat** | `drive-operators.mp4`: `--explain-plan` then `planner --compare` | "Privacy, in the same paragraph as the phrase 'fundamental right'… starts with the proximity node, which has a hundred and twenty-six documents, while the two filters have more than a thousand each. An intersection" |
| 10 | 2:44–2:59 | ANIM | — | **Suvrat** | **Insert.** Three operand bars sized by the real df re-sort so the rarest is first; the running count walks 126 → 66 → 16 → 9 | "can only shrink, so starting small keeps every later merge small. A hundred and twenty-six, sixty-six, sixteen, nine — and the same answer either way, for seventy-four fewer comparisons." |
| 11 | 2:59–3:09 | CAM | **Sumedha** | Sumedha | Webcam | "I own the analyzer, the parser and the scorers. Documents and queries go through the same function — that rule came first." |
| 12 | 3:09–3:31 | SCREEN | — | **Sumedha** | `drive-analyzer.mp4`: `statutes --spans`, `--crosswalk`, `cli --analyze` | "Three ways of writing a statute, in one sentence. All three resolve… four-ninety-eight-A crosswalks to two sections of the new Sanhita, so the query has to fire both. Because" |
| 13 | 3:31–3:45 | ANIM | — | **Sumedha** | **Insert.** Three surface forms of section 302 split into tokens, diverge into three postings lists, then converge and fuse into `ipc§302` | "to a standard tokenizer those three strings share no terms — one provision in three postings lists. We fuse them at build time into one identifier, with stemming off." |
| 14 | 3:45–3:55 | CAM | **Rakshit** | Rakshit | Webcam | "I built the ingest, the zoning, the index and the proximity operators. One posting carries three things at once." |
| 15 | 3:55–4:23 | SCREEN | — | **Rakshit** | `drive-index.mp4`: `positional term "ipc§302"` then `--merge --trace` | "There they are. The statute token appears in one thousand and eighty-one judgments; the skip stride is thirty-two… Here is the proximity merge, traced:" |
| 16 | 4:23–4:39 | ANIM | — | **Rakshit** | **Insert.** Doc 239's two real position lists walked by pointers i and j; the `/p` match at ¶8 lights green; the three predicates at right | "two position lists in document 239, walked by two pointers. Compare paragraph ids and you have slash p; sentence ids, slash s; offsets, slash n. One walk, one comparison changed. Eight pairs matched here." |
| 17 | 4:39–4:49 | CAM | **Suvrat** | Suvrat | Webcam | "The citation graph is mine. The idea the whole project turns on: a citation is a verdict, not a vote." |
| 18 | 4:49–5:17 | SCREEN | — | **Suvrat** | `drive-authority.mp4`: `citations stats` then `pagerank compare` | "Seventy-six thousand nine hundred and fifty-eight citations found in prose… Ninety-one judgments come out flagged overruled. Pune Municipal Corporation: cited seventy-three times, seventeen of those say overruled, and its authority" |
| 19 | 5:17–5:32 | ANIM | — | **Suvrat** | **Insert.** Pune Municipal Corporation's authority bar fills from its 73 citations, then one red `overruled` edge drains it 0.377 → 0.087; NO LONGER GOOD LAW lands | "drops from zero point three seven seven to zero point zero eight seven. An overruling edge transfers nothing, and its target takes a penalty. We flag that, we don't hide it." |
| 20 | 5:32–6:04 | SCREEN | — | **Rakshit** | `drive-tracka.mp4`: `nyaya.eval.run` printing the full Track A table and the significance block | "Twenty judged queries, graded blind… Seventeen queries improve, none tie, three get worse — p equals nought point nought nought two six, and the bootstrap excludes zero. Now the last two rows: S3 beats S4. We will not pretend otherwise. What S4 buys, not recommending dead law, is what recall at k cannot see." |
| 21 | 6:04–6:24 | SCREEN | — | **Rakshit** | `drive-ablation.mp4`: `nyaya.rank.ablate --run` — four arms with their Δ nDCG column | "Then turn one thing off at a time. No stemming costs a hundredth of nDCG and a bigger vocabulary. No statute canonicalisation costs twelve points — and it all lands on the statute queries, one of which drops to zero." |
| 22 | 6:24–6:50 | SCREEN | — | **Suvrat** | `drive-tracksbc.mp4`: `citation_recovery report` then `efficiency report` | "Citation recovery… S3 leads at nought point one seven six. G nought ranks by authority alone and never reads the query — nought point nought one two. Forty-five percent of the gold was never sampled. And the champion lists cost no recall at all." |
| 23 | 6:50–7:10 | SCREEN | — | **Sumedha** | `drive-brief.mp4`: `--brief` on a dying-declaration query — the summary, the sources, the grounding check | "The Precedent Brief ships extractive — we had no model key here, so every sentence is verbatim from a judgment, and it says so. The verifier catches four fabrications in eighteen with no false alarms. Cosine is topicality, not entailment." |
| 24 | 7:10–7:29 | SCREEN | — | **Rakshit** | `drive-limits.mp4`: `--treatment-audit --only-errors --limit 3` | "And here is what it gets wrong. Ninety-two and a half percent on forty hand-labelled edges, and every error runs the same way — upward, into a label that demotes. Our worst bug was duller: a correct fix that never reached the store." |
| 25 | 7:29–7:38 | CAM | **Sumedha** | Sumedha | Webcam | "What's new is treatment-aware authority: ranking precedent by how later courts treated it, not how often they mentioned it." |
| 26 | 7:38–7:45 | ANIM | — | **Suvrat** | End card: the mark, the three names, the corpus line, CC-BY-4.0 | "Thirteen thousand judgments indexed, thirty-seven thousand in the graph. Thank you." |

---

## Where the voice crosses a cut

These are the five places where one passage runs into the next without a pause.
They are the reason the narration is recorded as seven long takes rather than
twenty-four short ones.

| At | Out of | Into | Who | The sentence that straddles it |
|----|--------|------|-----|--------------------------------|
| 0:49 | drive-freetext | anatomy | Sumedha → **Rakshit** | "…it reads a judgment as one bag of words, **and a judgment is not a bag of words.**" |
| 1:30 | drive-why | bm25f | Sumedha | "…one tf of two hundred and thirty-eight for the whole document, **because BM25F adds the zones together before it saturates**." |
| 2:44 | drive-operators | boolean-plan | Suvrat | "…An intersection **can only shrink**, so starting small keeps every later merge small." |
| 4:23 | drive-index | proximity-walk | Rakshit | "Here is the proximity merge, traced: **two position lists in document 239**…" |
| 5:17 | drive-authority | overruled-trap | Suvrat | "…and its authority **drops from zero point three seven seven to zero point zero eight seven.**" |

The two remaining inserts (`three-ways` at 3:31) also pick up mid-sentence from
`drive-analyzer`: "…so the query has to fire both. **Because** to a standard
tokenizer those three strings share no terms…".

---

## On-camera rotation

Six takes, 9–12 seconds each: 60 seconds of camera in 7:45, **12.9%** of the
film. The rotation is Rakshit → Suvrat → Sumedha, twice, so each person appears
exactly twice and no two consecutive segments put the same face on screen.

| Slot | Time | Person | Why them |
|------|------|--------|----------|
| cam-intro | 0:09 | Rakshit | opens; he owns video production |
| cam-t6 | 2:03 | Suvrat | frames the track, and he owns the structural half |
| cam-analyzer | 2:59 | Sumedha | analyzer, parser, scorers and the page are hers |
| cam-index | 3:45 | Rakshit | index and proximity operators are his |
| cam-authority | 4:39 | Suvrat | citation graph, PageRank, g(d) are his |
| cam-close | 7:29 | Sumedha | closes, since Rakshit opened |

There is no webcam take in Movement 6 on purpose: the five measurement clips run
back to back, nearly two minutes of unbroken live system, so the numbers arrive
without interruption. That run is the argument of the film and a talking head in
the middle of it would only slow it down.

---

## Narration split against ownership

Everyone narrates only what `docs/CONTRACTS.md` says they built.

| Person | Owns (contracts) | Narrates | Words |
|--------|------------------|----------|-------|
| **Rakshit** | ingest, zones, positional index, proximity operators, treatment classifier, snippets, evaluation harness | 01 title · 02 cam · 04 zones · 14 cam · 15 postings · 16 proximity · 20 Track A · 21 ablations · 24 the audit | 329 (34.2%) |
| **Suvrat** | parametric index, citation graph, PageRank and g(d), Boolean planner, net score, tiered index, report | 08 cam · 09 the plan · 10 planner · 17 cam · 18 the graph · 19 the overruled drain · 22 Tracks B and C · 26 outro | 292 (30.4%) |
| **Sumedha** | statute canonicaliser and analyzer, paragraph store, query parser, tf-idf / BM25F, "why this result", CLI, Streamlit page, Precedent Brief | 03 the demo · 05 why · 06 BM25F · 07 the page · 11 cam · 12 statutes · 13 canonicalisation · 23 the brief · 25 cam close | 341 (35.4%) |

Counts are generated, not estimated: `python scripts/wordcount.py script/voiceover.md src/timeline.ts` from `video/` prints them and fails loudly if any passage grows past 90% of its slot.

---

## What is no longer in the film

Nine animated scenes from the previous cut were deleted rather than shortened,
because the terminal shows the same thing better and a 7-minute film with a tight
argument beats a padded one:

`pipeline-map`, `statute-canon`, `zoned-postings`, `citation-graph`, `net-score`,
`eval-ladder`, `operator-speed`, `limitation`, `precedent-brief`.

The architecture diagram survives in `report/report.html` as Figure 1, where a
reader can study it; the evaluation ladder survives as Figure 2 and as the live
table `nyaya.eval.run` prints at 5:40. Nothing that was deleted is unexplained —
it is explained by the system itself, on screen.
