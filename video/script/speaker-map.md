# NyayaTrace — speaker map

Every second of the 7:27 film, who is speaking, whether they are on camera, what
is on screen, and the exact line. Timings are generated from
`video/src/timeline.ts`; run `npm run timeline` in `video/` to reprint them after
any change to a segment length.

Legend — **kind**: `ANIM` Remotion animation · `SCREEN` live screen recording ·
`CAM` webcam, speaker visible.

| # | Time | Kind | On camera | Voice | On screen | Line |
|---|------|------|-----------|-------|-----------|------|
| 01 | 0:00–0:17 | ANIM | — | **Suvrat** | Title card: "NyayaTrace", the tagline, CSD358 / Mid-term Hackathon / Track T6, the three names; a citation constellation drifts behind | "NyayaTrace. A search engine for Indian Supreme Court precedent, built for the mid-term hackathon in Information Retrieval, on track T6. Rakshit, Suvrat and Sumedha. Fourteen thousand judgments, and one unusual idea." |
| 02 | 0:17–0:30 | CAM | **Rakshit** (all three in frame if possible) | Rakshit | Webcam | "Hi — we're Rakshit, Suvrat and Sumedha. We built a precedent search engine over Supreme Court judgments. Before we show it running, here is the problem it exists to solve." |
| 03 | 0:30–0:50 | ANIM | — | **Sumedha** | Three surface forms of section 302 split into tokens, diverge into three postings lists, then converge and fuse into `ipc§302` | "Lawyers write the same provision three different ways. Section 302 IPC. S. 302 of the Indian Penal Code, 1860. u slash s 302. To a general tokenizer those share almost no terms, so one provision ends up in three places in the index. We collapse all three to one token." |
| 04 | 0:50–1:06 | ANIM | — | **Rakshit** | A page of running text separates into six zone bands; the zone weights grow beside them; `bench` is marked filter-only | "A judgment is not a bag of words either. It has a title, parties, a bench, the statutes it turns on, an operative order, and a very long body. The holding lives in the smallest part." |
| 05 | 1:06–1:23 | ANIM | — | **Suvrat** | Eleven green citations fill a node's authority bar; one red `overruled` edge arrives and drains it; the NO LONGER GOOD LAW stamp lands | "And here is the trap. A case can be cited hundreds of times and still be dangerous to cite, because one later bench overruled it. Every earlier citation still points at it. Popularity and reliability come apart." |
| 06 | 1:23–1:33 | CAM | **Suvrat** | Suvrat | Webcam | "That is why this is a vertical search problem, not a web one. The structure — zones, metadata, citations, authority — is the whole advantage." |
| 07 | 1:33–1:51 | SCREEN | — | **Sumedha** | `footage/demo-freetext.mp4`: free-text query, ten ranked results with year, bench and badges | "So, live. A plain free-text query against the real index. Ten ranked judgments, each with its year and bench size, and a badge where the citation graph has something to say about whether the case still stands." |
| 08 | 1:51–2:09 | SCREEN | — | **Sumedha** | `footage/demo-operators.mp4`: `privacy /p "fundamental right" AND bench:>=5 year:>=2015`, the AST and the df-ordered plan | "Now the operator language. Privacy within the same paragraph as the phrase 'fundamental right', on a bench of five or more, since 2015. You can see the parse tree, and the plan the engine chose to execute it." |
| 09 | 2:09–2:24 | SCREEN | — | **Sumedha** | `footage/demo-why.mp4`: one result card open — snippet, treatment badge, the "why this result" decomposition | "And every result can explain itself. Which zones the score came from, which terms carried it, the authority term, the penalty, and the sentence in another judgment that produced the badge." |
| 10 | 2:24–2:38 | ANIM | — | **Suvrat** | The architecture map: build row, store row (S1–S5), query row, explanation row; boxes colour-coded by owner | "Here is the whole system. The top row builds the index, the middle row is what sits on disk, the third row is what happens when you type. Colour is who owns what." |
| 11 | 2:38–2:49 | CAM | **Sumedha** | Sumedha | Webcam | "I own the analyzer, the query language and the scorers. The rule I care about most: documents and queries go through exactly the same function." |
| 12 | 2:49–3:05 | ANIM | — | **Sumedha** | Four analyzer stages transform the same clause; then the `ipc§302 ⇄ bns§103` crosswalk band | "Fold and tokenise. Canonicalise statutes with a grammar over citation forms. Drop the words every judgment contains. Stem — except statute tokens, which are identifiers. And at query time, cross-walk the old Penal Code to the new Sanhita." |
| 13 | 3:05–3:17 | SCREEN | — | **Sumedha** | `footage/analyze-parse.mp4`: `--analyze` token dump and `--parse` AST | "This is that function printing its own output: every token with its zone, its offset, its sentence and its paragraph. And the parser, turning a query into a tree." |
| 14 | 3:17–3:28 | CAM | **Rakshit** | Rakshit | Webcam | "I built the ingest, the zoning and the index itself, plus the proximity operators. One posting carries three things: which document, how often per zone, and exactly where." |
| 15 | 3:28–3:45 | ANIM | — | **Rakshit** | Dictionary entry opens into a postings list; per-zone tf bars grow; position tails appear; skip arcs draw over every 4th cell | "The dictionary points at an offset in one memory-mapped array of 32-bit integers. Each posting holds a per-zone term frequency and the full position list, and every square-root-of-df entries there is a skip pointer." |
| 16 | 3:45–4:01 | ANIM | — | **Rakshit** | Two position lists for doc 8142 walked by pointers i and j; the `/p` match lights green; the three predicates listed at right | "Proximity is then one merge with two pointers. Compare offsets and you have 'within n words'. Compare sentence ids and you have same sentence. Compare paragraph ids and you have same paragraph. A phrase is just the tightest case." |
| 17 | 4:01–4:14 | SCREEN | — | **Rakshit** | `footage/postings-plan.mp4`: `--postings privacy` and a traced `/p` merge | "Printed from the real index: the postings for one term, the skip stride, and the merge walking step by step. Nothing here is a mock-up." |
| 18 | 4:14–4:27 | ANIM | — | **Suvrat** | Three operand bars sized by df re-sort so the rarest is first; two candidate-set paths fall at different rates | "The planner never executes a conjunction in the order you typed it. It sorts the operands by document frequency and starts with the rarest, because an intersection can only shrink." |
| 19 | 4:27–4:42 | ANIM | — | **Sumedha** | Raw tf leaves the top of the plot while the BM25F curve saturates; k₁ marked; the two formulas; the zone-weight bar | "Then ranking. Raw term frequency runs off the top of the chart; BM25F saturates. The zones are summed before saturation, each with its own length normaliser, so a long body cannot out-shout a title." |
| 20 | 4:42–4:53 | CAM | **Suvrat** | Suvrat | Webcam | "My half is the citation graph, the authority score and the planner. The idea the whole project rests on: a citation is a verdict, not a vote." |
| 21 | 4:53–5:13 | ANIM | — | **Suvrat** | Edges take their treatment colours and weights; mass pulses along them through 60 power iterations; A's bar settles, then the overruled penalty drops it | "A cue-phrase classifier labels every edge — followed, distinguished, mentioned, overruled — and that label becomes the edge weight. We run our own weighted PageRank over it. An overruling edge transfers nothing, and the case it points at takes a status penalty on top." |
| 22 | 5:13–5:26 | ANIM | — | **Suvrat** | text̂, g(d) and prox fly in from three sides and assemble into `net = α·text̂ + (1−α)·g(d) + β·prox`; the explanation strip appears | "Text relevance, static authority and proximity meet in one score, with one weight each. Turn a weight off and you have an earlier system — which is how the evaluation is built." |
| 23 | 5:26–5:36 | CAM | **Rakshit** | Rakshit | Webcam | "I ran the evaluation. Six systems, each one change apart. Judged queries, plus two tracks where the answer key does not need a human at all." |
| 24a | 5:36–5:43 | SCREEN | — | **Rakshit** | `footage/eval-run.mp4`, first half: the harness printing metrics for B0…S4 | "This is the harness running all six systems over the same query set and writing the metrics out." |
| 24b | 5:43–5:50 | SCREEN | — | **Suvrat** | `footage/eval-run.mp4`, second half: the operator suite against the brute-force scan | "And the operator suite, checking every operator against a brute-force scan of the raw text." |
| 25 | 5:50–6:08 | ANIM | — | **Rakshit** | Six bars rise B0→S4 with their one-line differences; the P@5 / P@10 / R@20 / MAP / nDCG@10 table fills in; "shape only" stamp until measured | "Baseline is flat tf-idf. Each step adds exactly one idea: canonicalisation, zone weights, BM25F, authority, then treatment. Read the shape rather than any one bar — when a step does not move the metric, we say so." |
| 26 | 6:08–6:21 | ANIM | — | **Suvrat** | Two sets slide into coincidence (index vs brute force); the latency bars; the citation-recovery strip | "Two things need no judge. Operator correctness, which must match the brute-force scan exactly. And speed: champion lists and a two-tier index, against scoring every posting." |
| 27 | 6:21–6:39 | ANIM | — | **Rakshit** | The negated sentence; the cue lights red; the wrong label lands; the negation scope arc is drawn; the three-station roadmap | "Now what it gets wrong. 'We are unable to hold that the decision stands overruled.' The cue is there, the denial is not, so the classifier marks a case that was upheld. The fix is scope, not a longer phrase list." |
| 28 | 6:39–6:49 | SCREEN | — | **Rakshit** | `footage/limitation-live.mp4`: `--treatment-audit --only-errors` | "Here it is in the audit output — predicted label, gold label, and the sentence that fooled it. We ship the flag rather than hide it." |
| 29 | 6:49–7:06 | ANIM | — | **Sumedha** | Paragraph BM25 picks six paragraphs; four sentences appear each with a `[doc ¶id]`; the cosine verifier flags the last one amber | "Last thing. Paragraph-level BM25 over the same index picks the evidence; the model may not write a sentence without a paragraph id; and a cosine check re-reads each sentence against the paragraph it claims. No embeddings anywhere." |
| 30 | 7:06–7:18 | CAM | **Sumedha** | Sumedha | Webcam | "What is genuinely new here is treatment-aware authority — ranking a precedent by how later courts treated it, not how often they mentioned it. Thanks for watching." |
| 31 | 7:18–7:27 | ANIM | — | **Suvrat** (optional) | End card: the mark, the three names, the corpus line, the CC-BY-4.0 credit | "NyayaTrace. Fourteen thousand judgments indexed, the full forty-two thousand in the citation graph. Thank you." |

---

## On-camera rotation

| Slot | Time | Person | Why them |
|------|------|--------|----------|
| cam-intro | 0:17 | Rakshit | team introduction; he owns video production |
| cam-t6 | 1:23 | Suvrat | frames the track, and he owns the structural half |
| cam-analyzer | 2:38 | Sumedha | analyzer, parser, scorers are hers |
| cam-index | 3:17 | Rakshit | index and proximity operators are his |
| cam-authority | 4:42 | Suvrat | citation graph, PageRank, g(d) are his |
| cam-eval | 5:26 | Rakshit | he owns the evaluation harness |
| cam-close | 7:06 | Sumedha | closes, since Rakshit opened |

Seven webcam slots, 10 to 13 seconds each: 78 seconds of camera in 7:27, about 17% of
the film. Each person appears at least twice, and no two consecutive segments put
the same face on screen.

## Narration split against ownership

Everyone narrates only what `docs/CONTRACTS.md` says they built.

| Person | Owns (contracts) | Narrates |
|--------|------------------|----------|
| **Rakshit** | ingest, zones, positional index, proximity operators, treatment classifier, snippets, evaluation harness | 02 intro · 04 zones · 14 cam · 15 postings · 16 proximity · 17 postings screen · 23 cam · 24a eval run · 25 ladder · 27 limitation · 28 audit |
| **Suvrat** | parametric index, citation graph, PageRank and g(d), Boolean planner, net score, tiered index | 01 title · 05 the trap · 06 cam · 10 architecture · 18 planner · 20 cam · 21 citation graph · 22 net score · 24b operators · 26 speed · 31 outro |
| **Sumedha** | statute canonicaliser and analyzer, query parser, tf-idf / BM25F, "why this result", CLI | 03 three forms · 07–09 the live demo · 11 cam · 12 analyzer · 13 analyze screen · 19 BM25F · 29 Precedent Brief · 30 cam close |

Words: Rakshit 353 (35%), Sumedha 357 (35%), Suvrat 298 (30%). Suvrat's share is
lower by agreement because he also assembles the written report.
