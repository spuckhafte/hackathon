# NyayaTrace — recording checklist

Everything the film still needs is a file in `video/public/footage/`. Nothing in
`src/` has to change: each slot probes for its own file at render time, plays it
if it is there, and draws the instruction card if it is not. Drop a file in,
re-render, done.

```
video/public/footage/
  demo-freetext.mp4      demo-operators.mp4     demo-why.mp4
  analyze-parse.mp4      postings-plan.mp4      eval-run.mp4
  limitation-live.mp4
  cam-intro.mp4   cam-t6.mp4      cam-analyzer.mp4   cam-index.mp4
  cam-authority.mp4  cam-eval.mp4  cam-close.mp4
```

The file name must match exactly, lower case, `.mp4`, H.264. The folder is
git-ignored for `*.mp4`, so footage stays out of the repo — keep the masters in
shared storage and copy them in before the final render.

---

## 1 · Screen recordings

**Capture settings for all seven.** 1920×1080 exactly, 30 fps or better, no
cursor trails, no notification pop-ups. Terminal at **dark background, 18–20 pt
monospace, window maximised** — the frame is shown at full size, so anything
smaller than 18 pt will not read on a projector. Type at a normal pace; the clip
can be trimmed, it cannot be slowed down.

**Record each clip slightly longer than its slot** and trim to exactly the length
below, or the last frames will be cut mid-sentence.

| File | Length | Run exactly this | Must be legible |
|------|--------|------------------|-----------------|
| `demo-freetext.mp4` | **18.0s** | `python -m nyaya.cli --mode S4 "right to privacy of personal data"` | the typed query · ten ranked titles with year and bench · at least one treatment badge |
| `demo-operators.mp4` | **18.0s** | `python -m nyaya.cli --explain-plan 'privacy /p "fundamental right" AND bench:>=5 year:>=2015'` | the parsed AST · the df-ordered plan · candidate counts shrinking |
| `demo-why.mp4` | **15.0s** | `python -m nyaya.cli --why 1 --mode S4 'privacy /p "fundamental right"'` | per-zone contributions · top terms · pagerank, bench, penalty, net · the evidence sentence behind the badge |
| `analyze-parse.mp4` | **12.0s** | `python -m nyaya.cli --analyze "convicted u/s 302 IPC read with S. 34"` then `python -m nyaya.cli --parse 'bail! AND act:IPC year:>=2010'` | the token list with zone, pos, sent, para · the AST with its expansions |
| `postings-plan.mp4` | **13.0s** | `python -m nyaya.cli --postings privacy --limit 12` then `python -m nyaya.cli --merge 'privacy /p "fundamental right"' --trace` | doc ids with per-zone tf and positions · the skip stride · the pointer steps |
| `eval-run.mp4` | **14.0s** | `python -m nyaya.eval.run --systems B0,S1,S2,S2p,S3,S4 --qrels eval/queries` then `python -m nyaya.eval.operators --brute-force` | per-system metric rows as they print · the operator pass/fail summary |
| `limitation-live.mp4` | **10.0s** | `python -m nyaya.cli --treatment-audit --only-errors --limit 3` | the negated sentence · the predicted label · the gold label |

Notes:

* The two queries in `data.ts` (`DEMO_FREETEXT`, `DEMO_QUERY`) are the ones the
  animation names out loud. If a query has to change because the index gives a
  dull result, change it in `video/src/data.ts` **and** in
  `script/voiceover.md`, then re-record — do not let the narration and the
  terminal disagree.
* If a CLI flag above does not exist yet, ask the owner for the real one rather
  than inventing output: segment 17's line is literally *"nothing here is a
  mock-up"*.
* `eval-run.mp4` carries two voices (Rakshit then Suvrat). Cut it so the harness
  output fills the first seven seconds and the operator suite the last seven.

## 2 · Webcam takes

**Framing.** Waist-up or chest-up, camera at eye height, eyes on the lens and not
on your own preview. Leave headroom; keep the face in the left or right third if
you can, so the lower-left name strip has somewhere to sit. Plain wall or a
tidy background — the film is very dark, so anything cluttered behind you will
be the brightest thing in the frame. Window light in front of you, never behind.

**Audio.** Record the voice on the same take. A phone on a stand with a lapel mic
beats a laptop webcam with laptop audio.

| File | Length | Person | What to say | Beats to hit |
|------|--------|--------|-------------|--------------|
| `cam-intro.mp4` | **13.0s** | Rakshit (all three in frame if the shot allows) | voiceover.md §02 | name the team · name the project · hand off to the problem |
| `cam-t6.mp4` | **10.0s** | Suvrat | §06 | why this is a vertical problem · one sentence only, then hand to the demo |
| `cam-analyzer.mp4` | **11.0s** | Sumedha | §11 | what she owns · "documents and queries go through the same function" |
| `cam-index.mp4` | **11.0s** | Rakshit | §14 | what he owns · "one posting carries three things" |
| `cam-authority.mp4` | **11.0s** | Suvrat | §20 | what he owns · "a citation is a verdict, not a vote" |
| `cam-eval.mp4` | **10.0s** | Rakshit | §23 | six systems one change apart · the judge could not see which system retrieved what |
| `cam-close.mp4` | **12.0s** | Sumedha | §30 | the one genuinely new thing · thanks; do not re-list features |

Shoot three takes of each and keep the one where the person is clearly *talking
to the viewer*, not reading. These are short on purpose: 7 to 14 seconds is long
enough to be a person and short enough not to stall the film.

## 3 · Voice-over for the animated and screen segments

Record from `script/voiceover.md`, **one take per segment**, named
`vo-<segment id>.wav` (for example `vo-citation-graph.wav`). Do not record the
whole script in one pass: the slot lengths are fixed in `src/timeline.ts` and a
single long take cannot be nudged segment by segment.

Keep the voice tracks out of the Remotion project for now and lay them under the
rendered `.mp4` in the editor — `video/public/audio/*.wav` is git-ignored if you
would rather keep them beside the project.

## 4 · Dropping footage in and re-rendering

```bash
cd video
cp /path/to/takes/cam-intro.mp4 public/footage/cam-intro.mp4
npm run studio          # check the slot now plays the clip, not the card
npm run build           # renders out/NyayaTrace.mp4 at 1920x1080, 30fps
```

A slot switches from card to footage automatically; nothing has to be edited.
Partial footage is fine and expected — render whenever you like and the
remaining cards tell you what is still missing.

Useful while editing:

```bash
npm run timeline                                   # print the segment table and total length
npx remotion studio src/index.ts                   # scrub the whole film
npx remotion still src/index.ts seg-citation-graph out/x.png --frame=300
npx remotion render src/index.ts seg-bm25f out/bm25f.mp4   # one segment on its own
npx remotion render src/index.ts NyayaTrace out/preview.mp4 --scale=0.5  # fast check
```

## 5 · Before the final render — the checklist that actually matters

- [ ] All 14 files present in `public/footage/`, each trimmed to the exact length in the tables above.
- [ ] `video/src/data.ts` updated with the measured numbers and `MEASURED = true`.
- [ ] No amber dashed underline anywhere in the render — that marker means a figure is still a placeholder.
- [ ] No "shape only — awaiting measurement" stamp on the evaluation charts.
- [ ] Segments 25, 26 and 28 re-recorded after the numbers landed.
- [ ] Total length still between 5:00 and 8:00 (`npm run timeline` prints it).
- [ ] Watched once end to end with sound, on a screen you did not make it on.
