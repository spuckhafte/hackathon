# NyayaTrace — recording checklist

Everything the film still needs is a file in `video/public/footage/`. Nothing in
`src/` has to change: each slot probes for its own file at render time with a
`HEAD` request, plays it if it is there, and draws the instruction card if it is
not. Drop a file in, re-render, done.

```
video/public/footage/
  drive-freetext.mp4    drive-why.mp4         drive-ui.mp4
  drive-operators.mp4   drive-analyzer.mp4    drive-index.mp4
  drive-authority.mp4   drive-tracka.mp4      drive-ablation.mp4
  drive-tracksbc.mp4    drive-brief.mp4       drive-limits.mp4
  cam-intro.mp4   cam-t6.mp4        cam-analyzer.mp4
  cam-index.mp4   cam-authority.mp4 cam-close.mp4
```

Twelve screen clips (5:01), six webcam clips (1:00). File names must match exactly,
lower case, `.mp4`, H.264. The folder is git-ignored for `*.mp4`, so footage
stays out of the repo — keep the masters in shared storage and copy them in
before the final render.

---

## 0 · Before you record anything

**Every command below was run on this machine and verified to work.** The
timings in the "runtime" column are what each command actually took, measured.
Budget for them: a clip running two commands pays the ~2 s index load twice.

```bash
cd "C:/Users/tech/rakshit/IR assignments/hackathon"
export PYTHONIOENCODING=utf-8        # or: $env:PYTHONIOENCODING = "utf-8"
```

**Set `PYTHONIOENCODING=utf-8` or `§` and `¶` will not render**, and the statute
tokens are the point of two of these clips.

**Terminal setup.** 1920×1080 exactly, dark background, **18–20 pt monospace**,
window maximised, cursor blink off, no notification pop-ups, no window chrome
with your name in it. The frame is shown at full size, so anything under 18 pt
will not read on a projector. Clear the screen before each command
(`clear` / `cls`) so the output starts at the top.

**Type at a normal pace.** The clip can be trimmed; it cannot be slowed down.
Record each clip 5–10 seconds longer than the target length and trim to exactly
the length in the table, or the last frames will cut mid-sentence.

**One known-broken command.** `python -m nyaya.authority.pagerank top` raises
`KeyError: 'top'` today. Do not record it. `pagerank compare` is the verified
command and shows more. If the owner fixes `top`, it still does not go in the
film — the compare table is the better shot.

---

## 1 · Screen recordings

### `drive-freetext.mp4` — **28.0 s** — Sumedha

The headline result, and the baseline that gets it wrong.

```bash
python -m nyaya.cli --mode S4 "right to privacy of personal data"
python -m nyaya.cli --mode B0 "right to privacy of personal data"
```

Runtime: 4.5 s + 1.5 s. Leave ~6 s of reading time on the S4 list before typing
the second command, and ~6 s on the B0 list at the end.

**Must be legible:**
* the typed query, both times, and the `mode S4` / `mode B0` line
* S4 ranks 1–4: Puttaswamy **2018** (bench 5), Puttaswamy **2017** (**bench 9**),
  Electoral Bonds / *Association for Democratic Reforms* **2024** (bench 6),
  Puttaswamy **2015** (bench 3)
* B0 rank 1: the **2015** three-judge reference — and that neither the 2017 nor
  the 2018 judgment appears anywhere in B0's ten
* the `1,174 candidates · 1,174 scored` line, and the `S5 tier 1` note under it

> The contrast is the whole shot. If the two lists ever stop differing, tell
> Rakshit before re-recording — the narration says B0 "never reaches the 2017 or
> 2018 judgments" and that must stay true on screen.

### `drive-why.mp4` — **28.0 s** — Sumedha

```bash
python -m nyaya.cli --why 1 --mode S4 'privacy /p "fundamental right"'
```

Runtime: 8.8 s. Output is 73 lines, so it will not fit one screenful at 18 pt:
**scroll it slowly, once, top to bottom** over the remaining ~19 s. Do not scroll
back up.

**Must be legible:**
* the header: rank 1 is **Puttaswamy 2017, bench 9**, doc 12211
* `per-zone share of tf~` — `order 27.010918`, `body 210.903478`, `total 237.914396`
* `query terms contributing most` — `privaci 4.630196` at the top
* the `static authority g(d)` block: `pagerank_norm × 0.75`, `bench_norm × 0.25`
  (**0.692308**, because the bench is 9), `treatment penalty 1.000000`, `g 0.251052`
* the `net score` block and the line
  `arithmetic check: the parts reproduce the net score (0.900210 vs 0.900210)`
* at least one `evidence passages` entry with its `[¶ n]` id

### `drive-ui.mp4` — **18.0 s** — Sumedha

The only clip that is not a terminal.

```bash
streamlit run app.py
```

Then, on the page, in this order:
1. type `privacy /p "fundamental right"` into the query box
2. open the **"how the query was understood"** expander
3. scroll to one result card and leave the score explanation visible
4. in the sidebar, change **mode** from `S4` to `B0`

Start recording *after* the page has loaded and the index is cached — the first
load spins for a few seconds and that is not interesting.

**Must be legible:**
* the four metric tiles: **path · candidates · scored · time**
* the parse tree inside the expander, and the `free terms` / `tf-idf terms` rows
* one result card with its treatment badge and a paragraph-cited snippet
* the result list **visibly changing** when the mode drops to B0

### `drive-operators.mp4` — **32.0 s** — Suvrat

```bash
python -m nyaya.cli --explain-plan 'privacy /p "fundamental right" AND bench:>=5 year:>=2015'
python -m nyaya.query.planner --compare 'privacy /p "fundamental right" AND bench:>=5 year:>=2015'
```

Runtime: 7.1 s + 11.5 s = 18.6 s of the 32 s. That leaves ~13 s for typing and
reading, which is tight — **have the second command ready to paste**.

**Must be legible:**
* the parsed tree: `PROX same paragraph` over `TERM(privaci)` and
  `PHRASE[fundament right]`
* the order line
  `PROX_P[126] -> FILTER(bench>=5)[1154] -> FILTER(year>=2015)[2139]`
* the `running` column: **126 → 66 → 16 → 9**
* from `--compare`: `same answer: True`, and the three policies with
  `compared 3,375` (size, the default) against `3,449` (as-written)

> `--compare` reports the default is only **1.02×** cheaper *on this query*, so
> the narration reads the comparison counts instead — 3,375 against 3,449, same
> answer either way. **Do not say "two point one times" over this shot.** That is
> a mean over the whole query set, it is part of Track C, and Track C is being
> re-run.

### `drive-analyzer.mp4` — **22.0 s** — Sumedha

Three fast commands; this clip is mostly typing.

```bash
python -m nyaya.statutes --spans "convicted u/s 302 I.P.C. and under Section 34, and Art. 21 of the Constitution"
python -m nyaya.statutes --crosswalk "ipc§498a"
python -m nyaya.cli --analyze "convicted u/s 302 IPC read with S. 34"
```

Runtime: 0.2 s + 0.2 s + 1.8 s. **Paste these**, do not type them — the long
first string will eat the slot.

**Must be legible:**
* three different surface forms resolving: `u/s 302 I.P.C. -> ipc§302`,
  `under Section 34 -> ipc§34`, `Art. 21 of the Constitution -> const§21`
* **`inherited=True`** on the `sec_only` span — the act carried over from the
  span before it
* `ipc§498a` → `['bns§85', 'bns§86']` and the `NOT one-to-one` note
* the `--analyze` token table with its `pos / sent / para / zone` columns and the
  `*` marking the token kept in positions but dropped from scoring

### `drive-index.mp4` — **28.0 s** — Rakshit

```bash
python -m nyaya.index.positional term "ipc§302"
python -m nyaya.cli --merge 'privacy /p "fundamental right"' --trace
```

Runtime: 1.4 s + 6.1 s.

**Must be legible:**
* `df : 1081 (7.79% of 13878 documents)`, `idf : 2.5520`
* `skip stride : 32 (34 skip pointers, ~sqrt(df))`
* at least three posting rows with their `zone tfs` and the
  `first 3 -> sent [...], para [...]` line underneath
* from `--merge`: the `pointer walk inside doc 239` block —
  `left ['privaci'] positions 536, 7027`, the right-hand position list,
  `pairs accepted 8`, `smallest gap 2 tokens`, and the `first pairs` line

> The `proximity-walk` animation at 4:29 draws **exactly** these positions and
> their paragraph ids. If this output ever changes, the insert is wrong — update
> `DEMO_WALK` in `video/src/data.ts`.

### `drive-authority.mp4` — **28.0 s** — Suvrat

```bash
python -m nyaya.authority.citations stats
python -m nyaya.authority.pagerank compare
```

Runtime: 1.1 s + 1.9 s. Both print a full screen; leave ~11 s on each.

**Must be legible:**
* `citation occurrences : 76,958`, `resolved : 50,795 (66.00%)`,
  `edges written : 76,118`
* the **by resolution rung** table (`scr_exact 46.52%` at the top)
* `nodes : 37,210` and
  `status under S4 : {'ok': 37033, 'overruled': 91, 'distinguished': 86}`
* the **Pune Municipal Corporation** row: `2014 · 73 · 17 · overruled ·
  0.377 → 0.087 · −0.289`
* the **T. Devadasan** row: `1964 · 61 · 3 · 0.460 → 0.115`
* the `(!)` footnote: *67 of 91 overruled judgments rest on a single overruling
  citation*

> Maneka Gandhi used to sit at row 2 of this table and was a **known false
> positive**. It is gone. It was being served from an authority store built
> before the re-classification pass that fixed it, and the rebuild cleared it;
> the overruled count fell from 93 to **91** and distinguished from 96 to **86**
> in the same rebuild, which is why those two figures changed. The cases the film
> names — Pune Municipal Corporation (17 of 73) and T. Devadasan (3 of 61) — are
> corroborated, legally verifiable, and both are in this one shot.
>
> If a later shot does surface Maneka Gandhi, note that it prints **bench 1**
> although it is a seven-judge bench: its CORAM line OCRs as `M. H. BEG, c. J.`
> and the parser never reaches the other six names. Name it as the OCR limitation
> it is rather than pretending it is not on screen.

### `drive-tracka.mp4` — **32.0 s** — Rakshit

```bash
python -m nyaya.eval.run
```

Runtime: **24.3 s** — this clip is almost entirely the harness running, which is
the point. Start recording on the `Enter` key.

**Must be legible:**
* the six per-system lines appearing one at a time (`B0 20 queries in 6.4s` …)
* the full Track A table, all six rows, all five metric columns
* **`S3 … 0.8833`** and **`S4 … 0.8786`** adjacent, so the viewer can see S4 does
  not win
* `per query: 17 improved, 0 tied, 3 worsened`
* `sign test: 17-3 (ties 0), p = 0.0026  SIGNIFICANT`
* `bootstrap 95% CI on the mean difference: [+0.1623, +0.3781]  excludes 0`

### `drive-ablation.mp4` — **20.0 s** — Rakshit

The clearest piece of evidence in the project that this is a *vertical* problem:
switch off one piece of domain engineering at a time and see what it was worth.

```bash
python -m nyaya.rank.ablate --run
```

Runtime: **37.8 s**, which is longer than the slot. Start it, let it run, and
record the **table** — it prints all at once at the end. Do not film the wait.

**Must be legible:**
* all four arms with the `d nDCG` column
* **`nocanon  0.7582  -0.1205`** — by a factor of ten the largest single loss
* `nostem  0.8669  -0.0117`, and its vocabulary rising 544,787 → **618,852**
* the note under the table: act inheritance is **query side only**, scores
  `+0.0000`, and is therefore a **partial** ablation

> The vocabulary column is not trustworthy below about 100 types: the build is
> not bit-deterministic and two builds from identical inputs differed by 36 and
> 56 types. `noinherit` sits 56 from `full`, which is noise, and the narration
> does not mention it.
>
> Where the −0.1205 lands matters as much as its size. It is concentrated on the
> statute-anchored queries — q23 falls **0.7761 → 0.0000**, q14 0.9545 → 0.2440,
> q15 0.8128 → 0.4053 — which is exactly where canonicalisation is supposed to
> be doing the work.

---

### `drive-tracksbc.mp4` — **26.0 s** — Suvrat

Both of these read the finished run out of `runs/*.json`, so they are instant.

```bash
python -m nyaya.eval.citation_recovery report
python -m nyaya.eval.efficiency report
```

Runtime: 1.1 s + 0.8 s. Leave ~11 s on each.

**Must be legible:**
* the R@10 column with its CIs: **S3 0.176** best, and the
  **`G0  0.012 … <- no query signal`** row
* `DISTINCTIVE leak (df<=100) : 0/1222 gold pairs (0.00%)`
* `excluded as unretrievable: 45.2%`
* the `scored p50` column at **1,030** for every mode, and the warm p50 column at
  **4–7 ms**
* the two-tier block, including the worst-case line

> **Two things not to say over this shot.**
>
> **1. Do not quote an end-to-end latency speed-up.** Three runs of identical
> code gave falling wall clock as the OS page cache warmed — B0 went 6.18 → 5.22
> → 4.00 ms — while the algorithmic columns were identical every time. Any ratio
> would be measuring the page cache. Quote the algorithmic reduction instead:
> candidates p50 12,548 → 1,059, documents scored p50 8,032 → 1,030, and
> **159,328 → 17,718 over the whole query set, 8.99× fewer**.
>
> **2. The tier fidelity block on this screen is the HARNESS path, not the
> product.** It reads 12/20 identical with a 3/10 worst case because the harness
> builds candidates from `scoring_terms`, so *appellate jurisdiction of the
> supreme court* collapses to the single term `suprem` and tier 1 is one champion
> list. `search()` builds from `free_terms`, so tier 1 is a union of six. On the
> shipped path it is **18/20 identical, mean overlap 0.990, rank 1 unchanged
> 20/20, worst case 9/10**, and on the judged set the champion lists cost
> *exactly zero* — nDCG@10, MAP and P@10 identical to four decimals with tiers on
> and off. If you mention 3/10 at all, say which path it came from.
>
> The unretrievable-by-construction share stays on screen beside Track B, and the
> tier worst case stays on screen beside the tier gain. That is not negotiable.

### `drive-brief.mp4` — **20.0 s** — Sumedha

The stretch feature, shown with its reach honestly bounded rather than oversold.

```bash
python -m nyaya.cli --brief "dying declaration as the sole basis of conviction"
```

Runtime: 5.1 s cold, under a second warm. Leave ~12 s on the output.

**Must be legible:**
* the summary, with a **`[Pn]` tag on every single sentence**
* the `sources` block, with real `¶` ids and `window n of m` ids
* the `grounding check` header: **`composer   extractive`**, `threshold tf-idf
  cosine >= 0.20`, `sentences 6   verified 6   flagged 0`
* the per-sentence `cos=1.000` column

> **Say that it ships extractive, and say why.** No LLM API key was available in
> this environment — `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are both unset and
> the `anthropic` SDK is not installed — so the abstractive path, although
> written and prompt-complete, **has never been executed**. Every brief carries a
> note saying its output is verbatim. A working extractive feature with an
> honest boundary is worth more than an LLM demo we cannot run, and the word
> `extractive` is on screen in this shot.
>
> **The verifier is measured, not asserted.** 36 real extractive sentences score
> cosine 0.9924–1.0000; 18 deliberately written fabrications score 0.0866–0.8451.
> At the shipped threshold of 0.20 it catches **4 of 18 fabrications with 0 false
> alarms on the 36 real ones**. It reliably catches the mechanical failures — no
> citation, or a citation outside the evidence pack — but **not a confident
> on-topic falsehood**: *"The right to privacy was expressly held to be
> subordinate to the right to property"* scores 0.8451 and passes. A bag-of-words
> cosine is a topicality check, not an entailment check. The threshold was *not*
> raised to 0.90, because that would reject a genuine paraphrase from the
> abstractive composer, and a test pins the limitation.

---

### `drive-limits.mp4` — **19.0 s** — Rakshit

```bash
python -m nyaya.cli --treatment-audit --only-errors --limit 3
```

Runtime: 4.0 s. The whole 16-line output fits one screen.

**Must be legible:**
* `edges judged : 40 of 40 gold entries` and `accuracy : 92.5%`
* the per-label block: `overruled` precision 80%, `distinguished` 100%,
  `followed` 90%, `mentioned` 100% precision but **77% recall**
* all **three** error rows — every one is `gold=mentioned` read as something
  stronger, which is the worst direction for this error to run
* the final `documented limitation:` line

> The classifier prints **92.5%** (37 of 40), not 97.5%. The higher figure
> belonged to an older gold file, replaced when the gold keys gained a window
> fingerprint; the two files share no keys, and four historical versions of
> `classify()` all score 37/40 against the current gold, so nothing regressed.
> The tool is the authority and the narration reads 92.5%.
>
> `python -m nyaya.authority.treatment audit` prints an identical block if you
> prefer that entry point; the film uses the `cli` form for consistency.

### Optional pickup — the operator suite

`python -m nyaya.eval.operators --brute-force` runs 94 checks — 10 Boolean, 14
phrase, 50 `/n` across *k* ∈ {1, 3, 5, 10, 25}, 10 `/s`, 10 `/p` — over 1,200
judgments spanning 1950–2025, and proves **94/94 = 100.00%** agreement against an
independent naive token scan: the token stream rebuilt from text and walked with
nested loops, no postings, no skip pointers, no planner. Against a raw regex it
agrees on **2,740 of 2,888 = 94.88%** of document decisions, and all 148
differences are classified with **zero unexplained** — 135 stemming conflation,
9 regex prefix over-match, 3 token count, 1 OCR.

It takes over two minutes and cannot be filmed live inside a 7-minute cut, so it
is not in the current edit. If a reviewer asks for it, capture it as a separate
pickup: start the command, let it finish off-camera, and record the final summary
block — which prints its own wall clock on screen, so the trim is self-evident
rather than hidden.

If you narrate it, quote **both** numbers. The per-*check* regex figure alone
(27 of 44) badly overstates the disagreement, because a check that differs on one
document out of 1,092 counts exactly the same as one that differs on all of them.

---

## 2 · Webcam takes

**Framing.** Waist-up or chest-up, camera at eye height, eyes on the lens and not
on your own preview. Leave headroom; keep the face in the left or right third if
you can, so the lower-left name strip has somewhere to sit. Plain wall or a tidy
background — the film is very dark, so anything cluttered behind you will be the
brightest thing in the frame. Window light in front of you, never behind.

**Audio.** Record the voice on the same take. A phone on a stand with a lapel mic
beats a laptop webcam with laptop audio.

**These are short on purpose.** Nine to twelve seconds is long enough to be a
person and short enough not to stall a film whose job is to show software
running.

| File | Length | Person | Line | Beats to hit |
|------|--------|--------|------|--------------|
| `cam-intro.mp4` | **12.0s** | Rakshit (all three in frame if the shot allows) | voiceover §Movement 0 | name the team · name the project · "everything from here is the real system running" |
| `cam-t6.mp4` | **9.0s** | Suvrat | §Movement 2 | why this is a vertical problem · one sentence, then straight back to the terminal |
| `cam-analyzer.mp4` | **10.0s** | Sumedha | §Movement 3 | what she owns · "documents and queries go through the same function" |
| `cam-index.mp4` | **10.0s** | Rakshit | §Movement 4 | what he owns · "one posting carries three things at once" |
| `cam-authority.mp4` | **10.0s** | Suvrat | §Movement 5 | what he owns · "a citation is a verdict, not a vote" |
| `cam-close.mp4` | **9.0s** | Sumedha | §Movement 7 | the one genuinely new thing · do not re-list features |

Shoot three takes of each and keep the one where the person is clearly *talking
to the viewer*, not reading.

---

## 3 · Voice-over

Record from `script/voiceover.md`, **one take per movement** — seven takes,
`vo-m0.wav` through `vo-m7.wav` — not one take per segment.

This cut's narration is a single continuous track: five sentences begin over the
terminal and finish over an animation insert (the table in `speaker-map.md`
lists them). A per-segment take cannot be made to sound continuous afterwards,
and the slot lengths are fixed in `src/timeline.ts`, so a single end-to-end take
cannot be nudged either. One take per movement is the compromise that works.

The two speakers either side of a hand-off should record together, in the room.
The hand-offs are at 0:49 (Sumedha → Rakshit), 1:30, 2:50, 3:37, 4:29 and 5:25.

Lay the voice tracks under the rendered `.mp4` in the editor;
`video/public/audio/*.wav` is git-ignored if you would rather keep them beside
the project.

---

## 4 · Dropping footage in and re-rendering

```bash
cd video
cp /path/to/takes/drive-freetext.mp4 public/footage/drive-freetext.mp4
npm run studio          # check the slot now plays the clip, not the card
npm run build           # renders out/NyayaTrace.mp4 at 1920x1080, 30fps
```

A slot switches from card to footage automatically; nothing has to be edited.
Partial footage is fine and expected — render whenever you like and the remaining
cards tell you what is still missing.

Useful while editing:

```bash
npm run timeline                                    # segment table, totals and the three percentages
npx remotion studio src/index.ts                    # scrub the whole film
npx remotion still src/index.ts seg-overruled-trap out/x.png --frame=340
npx remotion render src/index.ts seg-bm25f out/bm25f.mp4          # one segment alone
npx remotion render src/index.ts NyayaTrace out/preview.mp4 --scale=0.35   # fast whole-film check
```

---

## 5 · Before the final render — the checklist that actually matters

- [ ] All **18** files present in `public/footage/`, each trimmed to the exact length above.
- [ ] `npm run timeline` reports **7:45**, screen **64.7%**, animation **22.4%**, webcam **12.9%**.
- [ ] `python scripts/wordcount.py script/voiceover.md src/timeline.ts` says *all passages are inside 90% of their slot*.
- [ ] **No amber dashed underline anywhere in the render.** Every figure in `video/src/data.ts` is measured; an amber mark means something regressed, not something pending.
- [ ] `npm run timeline` reports **7:15**, screen **62.3%**, animation **23.9%**, webcam **13.8%**, and prints no warning line.
- [ ] No amber dashed underline anywhere in the render except the two knowingly-pending figures (the tuned α/β sweep and the Precedent Brief faithfulness numbers). Nothing else may be amber.
- [ ] Every number the voice says is visible on screen at that moment. Spot-check 0:21, 2:18, 4:01, 4:55, 5:40 and 6:40.
- [ ] The terminal clips show `§` and `¶` correctly — if they show `?` or a box, `PYTHONIOENCODING` was not set and the clip has to be re-shot.
- [ ] No window title bar, taskbar, notification or file path containing anything you would not want a grader to read.
- [ ] Watched once end to end with sound, on a screen you did not make it on.
