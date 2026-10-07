# NyayaTrace — voice-over script

**Film length 7:45 (13,950 frames at 30 fps).**

This is **one continuous narration track** laid over a screen recording of the
real system being driven. It is not a sequence of chapter voice-overs: a sentence
regularly *starts* over the terminal and *finishes* over an animation insert, and
the insert then cuts back to the terminal mid-paragraph. Read it straight through
as prose; the cue markers tell the editor where the picture changes, not where
the voice stops.

```
[SCREEN: id]   the live recording footage/<id>.mp4 is on screen
[ANIM: id ⟶]   an animation insert cuts IN here, mid-sentence
[⟵ ANIM]       the insert cuts OUT and we are back on the terminal
[CAM: name]    a webcam take; this person is visible and speaking
```

Read at roughly **150 words per minute** — 2.5 words a second. Every passage
below is written to land comfortably inside its slot; **no passage fills more
than 90% of the time available**, so nobody has to rush and the terminal always
gets a beat of silence to breathe.

<!--TOTALS-->**Total narration: 962 words ≈ 6:24 of speech inside 7:45 of film.**<!--/TOTALS-->

Nothing in this script states a number the screen is not showing at that moment.
Where the terminal prints a figure, the narration reads *that* figure.

---

## Movement 0 — open (0:00 – 0:21)

### `title` · 0:00 – 0:09 · **Rakshit** · 11 words · ~4.4s

> NyayaTrace. Treatment-aware precedent search for Indian case law. CSD358,
> Track T6.

*Nine seconds of card, four and a half of voice. Let the constellation drift.*

### `[CAM: intro]` · 0:09 – 0:21 · **Rakshit** · 26 words · ~10.4s

> We're Rakshit, Suvrat and Sumedha. This is a search engine over Supreme Court
> precedent. Everything from here is the real system running — so let's run it.

*Cut on "run it", straight into a terminal that is already open.*

---

## Movement 1 — the headline result (0:21 – 2:03)

### `[SCREEN: drive-freetext]` · 0:21 – 0:49 · **Sumedha** · 62 words · ~24.8s

> A plain English question: the right to privacy of personal data. Eleven hundred
> and seventy-four candidates, and the top four are the Puttaswamy line — 2018,
> 2017, 2015 — then Electoral Bonds. Now the same query through the flat tf-idf
> baseline. It puts the 2015 reference first and never reaches the other two,
> because it reads a judgment as one bag of words, and **[ANIM: anatomy ⟶]**

### `anatomy` · 0:49 – 1:02 · **Rakshit** · 29 words · ~11.6s

> a judgment is not a bag of words. It has a title, parties, a bench, statutes,
> an order and a long body. Bench is a filter, not evidence. **[⟵ ANIM]**

*Rakshit picks up Sumedha's unfinished sentence. That hand-off is deliberate —
the narration is one track with three voices, not three monologues.*

### `[SCREEN: drive-why]` · 1:02 – 1:30 · **Sumedha** · 57 words · ~22.8s

> Every result takes itself apart. The text score split by zone — order and body.
> The terms that carried it. The authority term, g of d, with its PageRank and
> bench parts and the status penalty. Then the net score, with the arithmetic
> checked against itself. One tf of two hundred and thirty-eight for the whole
> document, because **[ANIM: bm25f ⟶]**

### `bm25f` · 1:30 – 1:45 · **Sumedha** · 32 words · ~12.8s

> BM25F adds the zones together before it saturates, not after. Raw frequency
> runs off the chart; this curve flattens at k-one of one point two. A long body
> cannot out-shout a title. **[⟵ ANIM]**

### `[SCREEN: drive-ui]` · 1:45 – 2:03 · **Sumedha** · 33 words · ~13.2s

> Same function behind a page. The parse tree, the candidates, the time. A result
> card with its badge and a paragraph-cited snippet. Drop the mode to B0 and the
> list changes underneath me.

---

## Movement 2 — the query language (2:03 – 2:59)

### `[CAM: t6]` · 2:03 – 2:12 · **Suvrat** · 16 words · ~6.4s

> That's the vertical case in a line: the structure of a judgment is the whole
> advantage.

### `[SCREEN: drive-operators]` · 2:12 – 2:44 · **Suvrat** · 70 words · ~28.0s

> Privacy, in the same paragraph as the phrase "fundamental right", on a bench of
> five or more, since 2015. The parser builds a tree; the planner builds a plan —
> and it does not run it in the order I typed. It sorts by document frequency and
> starts with the proximity node, which has a hundred and twenty-six documents,
> while the two filters have more than a thousand each. An intersection
> **[ANIM: boolean-plan ⟶]**

### `boolean-plan` · 2:44 – 2:59 · **Suvrat** · 29 words · ~11.6s

> can only shrink, so starting small keeps every later merge small. A hundred and
> twenty-six, sixty-six, sixteen, nine — and the same answer either way, for
> seventy-four fewer comparisons. **[⟵ ANIM]**

---

## Movement 3 — the analyzer (2:59 – 3:45)

### `[CAM: analyzer]` · 2:59 – 3:09 · **Sumedha** · 21 words · ~8.4s

> I own the analyzer, the parser and the scorers. Documents and queries go
> through the same function — that rule came first.

### `[SCREEN: drive-analyzer]` · 3:09 – 3:31 · **Sumedha** · 48 words · ~19.2s

> Three ways of writing a statute, in one sentence. All three resolve. "Under
> section thirty-four" has no act attached, so it inherits the Penal Code from
> the span before it. And four-ninety-eight-A crosswalks to two sections of the
> new Sanhita, so the query has to fire both. Because **[ANIM: three-ways ⟶]**

### `three-ways` · 3:31 – 3:45 · **Sumedha** · 29 words · ~11.6s

> to a standard tokenizer those three strings share no terms — one provision in
> three postings lists. We fuse them at build time into one identifier, with
> stemming off. **[⟵ ANIM]**

---

## Movement 4 — the index (3:45 – 4:39)

### `[CAM: index]` · 3:45 – 3:55 · **Rakshit** · 19 words · ~7.6s

> I built the ingest, the zoning, the index and the proximity operators. One
> posting carries three things at once.

### `[SCREEN: drive-index]` · 3:55 – 4:23 · **Rakshit** · 58 words · ~23.2s

> There they are. The statute token appears in one thousand and eighty-one
> judgments; the skip stride is thirty-two, about the square root of that. Each
> row gives the per-zone term frequency, the positions, and under them the
> sentence and paragraph id for each position. Which is what makes the next
> command possible. Here is the proximity merge, traced: **[ANIM: proximity-walk ⟶]**

### `proximity-walk` · 4:23 – 4:39 · **Rakshit** · 35 words · ~14.0s

> two position lists in document 239, walked by two pointers. Compare paragraph
> ids and you have slash p; sentence ids, slash s; offsets, slash n. One walk,
> one comparison changed. Eight pairs matched here. **[⟵ ANIM]**

---

## Movement 5 — the citation graph (4:39 – 5:32)

### `[CAM: authority]` · 4:39 – 4:49 · **Suvrat** · 20 words · ~8.0s

> The citation graph is mine. The idea the whole project turns on: a citation is
> a verdict, not a vote.

### `[SCREEN: drive-authority]` · 4:49 – 5:17 · **Suvrat** · 61 words · ~24.4s

> Seventy-six thousand nine hundred and fifty-eight citations in prose;
> sixty-six percent of them resolved to a node, which is seventy-six thousand
> edges. Every edge is then classified — followed, distinguished, mentioned,
> overruled — and the label becomes the weight. Run PageRank over that and
> compare. Ninety-one judgments come out flagged overruled. Pune Municipal
> Corporation: cited seventy-three times, seventeen say overruled, and its
> authority **[ANIM: overruled-trap ⟶]**

### `overruled-trap` · 5:17 – 5:32 · **Suvrat** · 32 words · ~12.8s

> drops from zero point three seven seven to zero point zero eight seven. An
> overruling edge transfers nothing, and its target takes a penalty. We flag
> that, we don't hide it. **[⟵ ANIM]**

---

## Movement 6 — the measurement (5:32 – 7:29)

*Five screen clips back to back and no animation at all. By this point the viewer
can read a table on screen without help, and the honest numbers are the argument.
This is the longest unbroken run of live system in the film, and that is the
point of the cut.*

### `[SCREEN: drive-tracka]` · 5:32 – 6:04 · **Rakshit** · 70 words · ~28.0s

> Twenty judged queries, graded blind, six systems one flag apart. nDCG at ten
> goes from point six one to point eight eight. Seventeen queries improve, none
> tie, three get worse — p equals nought point nought nought two six, and the
> bootstrap excludes zero. Now the last two rows: S3 beats S4. We will not
> pretend otherwise. What S4 buys, not recommending dead law, is what recall at
> k cannot see.

### `[SCREEN: drive-ablation]` · 6:04 – 6:24 · **Rakshit** · 39 words · ~15.6s

> Then turn one thing off at a time. No stemming costs a hundredth of nDCG and a
> bigger vocabulary. No statute canonicalisation costs twelve points — and it all
> lands on the statute queries, one of which drops to zero.

*The strongest single argument that this is a vertical problem: the domain work
is worth ten times what the generic IR tuning is worth.*

---

### `[SCREEN: drive-tracksbc]` · 6:24 – 6:50 · **Suvrat** · 53 words · ~21.2s

> Citation recovery: hide a judgment's own citations, try to get them back. S3
> leads at nought point one seven six. G nought ranks by authority alone and
> never reads the query — nought point nought one two. Forty-five percent of the
> gold was never sampled. And the champion lists cost no recall at all.

### `[SCREEN: drive-brief]` · 6:50 – 7:10 · **Sumedha** · 39 words · ~15.6s

> The Precedent Brief ships extractive — we had no model key here, so every
> sentence is verbatim from a judgment, and it says so. The verifier catches four
> fabrications in eighteen with no false alarms. Cosine is topicality, not
> entailment.

---

### `[SCREEN: drive-limits]` · 7:10 – 7:29 · **Rakshit** · 42 words · ~16.8s

> And here is what it gets wrong. Ninety-two and a half percent on forty
> hand-labelled edges, and every error runs the same way — upward, into a label
> that demotes. Our worst bug was duller: a correct fix that never reached the
> store.

---

## Movement 7 — close (7:29 – 7:45)

### `[CAM: close]` · 7:29 – 7:38 · **Sumedha** · 19 words · ~7.6s

> What's new is treatment-aware authority: ranking precedent by how later courts
> treated it, not how often they mentioned it.

### `outro` · 7:38 – 7:45 · **Suvrat** · 11 words · ~4.4s

> Thirteen thousand judgments indexed, thirty-seven thousand in the graph. Thank
> you.

---

## Word count by speaker

<!--SPK-TABLE-->
| Speaker | Words | Share | Film time | Passages |
|---|---|---|---|---|
| Rakshit | 329 | 34.2% | 159s (34.2%) | title · cam-intro · anatomy · cam-index · drive-index · proximity-walk · drive-tracka · drive-ablation · drive-limits |
| Suvrat | 292 | 30.4% | 142s (30.5%) | cam-t6 · drive-operators · boolean-plan · cam-authority · drive-authority · overruled-trap · drive-tracksbc · outro |
| Sumedha | 341 | 35.4% | 164s (35.3%) | drive-freetext · drive-why · bm25f · drive-ui · cam-analyzer · drive-analyzer · three-ways · drive-brief · cam-close |
| **Total** | **962** | | **7:45** | **26 passages** |
<!--/SPK-TABLE-->

Everyone narrates only the components `docs/CONTRACTS.md` assigns them. Suvrat's
share is the smallest by agreement, because he also assembles the written report.
`drive-ablation` goes to Rakshit rather than Sumedha even though the arms test
her canonicaliser: he owns the evaluation harness, and a measurement of someone
else's component is more credible from the person who ran it.

---

## Slot fill check

No passage exceeds **90%** of its slot, and the counts below are generated from
the script itself rather than estimated — run
`python scripts/wordcount.py script/voiceover.md src/timeline.ts` from `video/`
to re-check after any edit. The tightest passages are <!--TIGHT-->`anatomy` (89%), `drive-freetext` (89%), `drive-limits` (88%)<!--/TIGHT-->;
every animation insert sits below 90% as well, which leaves the last beat of each
diagram silent. That is intentional: the picture needs a moment to land before
the cut back to the terminal.

<!--FILL-TABLE-->
| Slot | Kind | Length | Words | Speech | Fill |
|---|---|---|---|---|---|
| `title` | explain | 9s | 11 | 4.4s | 49% |
| `cam-intro` | cam | 12s | 26 | 10.4s | 87% |
| `drive-freetext` | screen | 28s | 62 | 24.8s | 89% |
| `anatomy` | explain | 13s | 29 | 11.6s | 89% |
| `drive-why` | screen | 28s | 57 | 22.8s | 81% |
| `bm25f` | explain | 15s | 33 | 13.2s | 88% |
| `drive-ui` | screen | 18s | 33 | 13.2s | 73% |
| `cam-t6` | cam | 9s | 16 | 6.4s | 71% |
| `drive-operators` | screen | 32s | 70 | 28.0s | 88% |
| `boolean-plan` | explain | 15s | 29 | 11.6s | 77% |
| `cam-analyzer` | cam | 10s | 21 | 8.4s | 84% |
| `drive-analyzer` | screen | 22s | 48 | 19.2s | 87% |
| `three-ways` | explain | 14s | 29 | 11.6s | 83% |
| `cam-index` | cam | 10s | 19 | 7.6s | 76% |
| `drive-index` | screen | 28s | 58 | 23.2s | 83% |
| `proximity-walk` | explain | 16s | 35 | 14.0s | 88% |
| `cam-authority` | cam | 10s | 20 | 8.0s | 80% |
| `drive-authority` | screen | 28s | 61 | 24.4s | 87% |
| `overruled-trap` | explain | 15s | 32 | 12.8s | 85% |
| `drive-tracka` | screen | 32s | 70 | 28.0s | 88% |
| `drive-ablation` | screen | 20s | 39 | 15.6s | 78% |
| `drive-tracksbc` | screen | 26s | 53 | 21.2s | 82% |
| `drive-brief` | screen | 20s | 39 | 15.6s | 78% |
| `drive-limits` | screen | 19s | 42 | 16.8s | 88% |
| `cam-close` | cam | 9s | 19 | 7.6s | 84% |
| `outro` | explain | 7s | 11 | 4.4s | 63% |
<!--/FILL-TABLE-->

---

## Recording notes

* **Record the narration in long takes, not per segment.** Each *movement* above
  is one take: `vo-m0.wav` through `vo-m7.wav`. Movement 6 is the longest, five
  screen clips with three voices and no animation between them. The whole point of this cut is
  that the voice does not stop at a segment boundary, and a per-segment take
  cannot be made to sound continuous afterwards.
* The person speaking a passage that *ends* mid-sentence and the person who
  *continues* it should record together, in the room, so the pick-up has the
  right pitch and pace. The hand-offs that matter are drive-freetext → anatomy
  (Sumedha → Rakshit) and drive-index → proximity-walk (Rakshit, same voice but
  a change of picture).
* The `[CAM: …]` passages are recorded **with** the webcam take, not separately.
* Leave about half a second of room at the head and tail of every take.
* **Every number in this script is on screen at the moment it is spoken.** If the
  terminal output changes — a rebuild, a re-run — re-check the figure before
  recording. The rule is not "do not read a placeholder"; the rule is "the
  narration and the terminal must never disagree".
* Pronunciation: *Nyaya* is "NYAH-yuh". *lnc.ltc* is "L-N-C dot L-T-C". *nDCG* is
  "N-D-C-G". *BM25F* is "B-M twenty-five F". *g(d)* is "g of d". *S2′* is
  "S-two-prime". *G0* is "G nought". *¶* is read "paragraph".
* Decimals are read as digits — "zero point eight eight two", not "nought point
  eight eighty-two" — because that is how they appear on screen.
