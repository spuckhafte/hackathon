/**
 * The edit, as data.
 *
 * ## The shape of the film
 *
 * This is a **continuous narrated screen recording of the real system being
 * driven**, with a small number of short concept animations cut in where a
 * viewer genuinely needs a picture to follow what is on screen at that moment.
 * It is not a deck and there are no slides: every `explain` segment is a diagram
 * that moves, and most of them draw numbers that the terminal is printing in the
 * segment immediately before or after.
 *
 * One array, three kinds of segment:
 *
 *   - `screen`  a gap for a live screen recording (CLI or the Streamlit page)
 *   - `explain` a concept animation, 13-16 s, cut in over the running narration
 *   - `cam`     a gap for a webcam take
 *
 * Durations are in seconds; the composition length is their sum, so the running
 * time can never drift from the script.
 *
 * ## Budget
 *
 *   screen     301 s   64.7%   12 clips      <- the majority of the film
 *   explain    104 s   22.4%    8 scenes     (88 s / 18.9% is concept inserts;
 *                                             title + outro are the other 16 s)
 *   cam         60 s   12.9%    6 clips      Rakshit -> Suvrat -> Sumedha, twice
 *   ------------------------------------
 *   total      465 s   7:45                  hard limits 5:00 - 8:00
 *
 * The voice-over is ONE continuous track: a sentence may start over the terminal
 * and finish over an animation insert. `script/voiceover.md` marks every
 * [ANIM: ...] and [CAM: ...] in and out point so the editor can align it.
 */
import type {Person} from './data';
import type {CamSlotProps, ScreenSlotProps} from './components/Slots';

export type Segment =
  | {kind: 'explain'; id: string; seconds: number; narrator: Person; beat: string}
  | ({kind: 'screen'; id: string; seconds: number; narrator: Person} & Omit<ScreenSlotProps, 'seconds' | 'name'>)
  | ({kind: 'cam'; id: string; seconds: number} & Omit<CamSlotProps, 'seconds' | 'slug'>);

export const TIMELINE: Segment[] = [
  // ------------------------------------------------------------------- open
  {
    kind: 'explain',
    id: 'title',
    seconds: 9,
    narrator: 'Rakshit',
    beat: 'Title, course, track, three names. Nine seconds, then we are in the terminal.',
  },
  {
    kind: 'cam',
    id: 'intro',
    seconds: 12,
    person: 'Rakshit',
    topic: 'Who we are and what you are about to watch',
    beats: [
      'All three in frame if the shot allows, otherwise Rakshit alone.',
      'Name the project, the course, the track, and the one-line claim.',
      'End on "everything from here is the real system running" - then cut to the terminal.',
    ],
  },

  // -------------------------------------------- the headline result, live
  {
    kind: 'screen',
    id: 'drive-freetext',
    seconds: 28,
    narrator: 'Sumedha',
    capture: 'The same query in S4 and in the tf-idf baseline, back to back',
    commands: [
      'python -m nyaya.cli --mode S4 "right to privacy of personal data"',
      'python -m nyaya.cli --mode B0 "right to privacy of personal data"',
    ],
    mustShow: [
      'S4: Puttaswamy at ranks 1, 2 and 4 \u2014 2018 bench 5, 2017 bench 9, 2015 bench 3',
      'S4: Electoral Bonds 2024 bench 6 at rank 3',
      'B0: the 2015 three-judge reference first, and no 2017 or 2018 Puttaswamy in the ten',
      'the candidate line on both: 1,174 candidates, and the S5 tier 1 note under it',
    ],
  },
  {
    kind: 'explain',
    id: 'anatomy',
    seconds: 13,
    narrator: 'Rakshit',
    beat:
      'INSERT over the result list: a page of running text separates into the six ' +
      'zones with their real weights, and `bench` is marked filter-only - which is ' +
      'why every result line carries a bench number but no bench score.',
  },

  // --------------------------------------------------- why, and the UI
  {
    kind: 'screen',
    id: 'drive-why',
    seconds: 28,
    narrator: 'Sumedha',
    capture: 'One result taken apart: --why on the top hit, scrolled slowly',
    commands: ['python -m nyaya.cli --why 1 --mode S4 \'privacy /p "fundamental right"\''],
    mustShow: [
      'rank 1 is Puttaswamy 2017, bench 9 \u2014 doc 12211',
      'per-zone share of tf~ (order 27.01, body 210.90, total 237.914396)',
      'query terms contributing most, privaci at the top',
      'g(d) = penalty x (0.75 x pagerank_norm + 0.25 x bench_norm) = 0.251052',
      'the net line 0.900210 and the arithmetic check that reproduces it',
      'the evidence passages with their paragraph ids',
    ],
  },
  {
    kind: 'explain',
    id: 'bm25f',
    seconds: 15,
    narrator: 'Sumedha',
    beat:
      'INSERT over the --why output: raw tf runs off the top of the chart while ' +
      'BM25F saturates at k1 = 1.2, and the zones are summed BEFORE saturation - ' +
      'which is exactly why the output prints one tf~ of 165.34 and not six scores.',
  },
  {
    kind: 'screen',
    id: 'drive-ui',
    seconds: 18,
    narrator: 'Sumedha',
    capture: 'The Streamlit page: the same search(), with the explanation panel open',
    commands: [
      'streamlit run app.py',
      '# type: privacy /p "fundamental right"   then open "how the query was understood"',
      '# then flip mode S4 -> B0 in the sidebar and watch the list change',
    ],
    mustShow: [
      'the path / candidates / scored / time metric row',
      'the parse tree in the expander',
      'a result card with its treatment badge and paragraph-cited snippet',
      'the list visibly changing when the mode drops to B0',
    ],
  },

  // ------------------------------------------- the operator language
  {
    kind: 'cam',
    id: 't6',
    seconds: 9,
    person: 'Suvrat',
    topic: 'Why this is Track T6',
    beats: [
      'Domain structure: zones, parametric metadata, citations, authority.',
      'One sentence only, then hand straight back to the terminal.',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-operators',
    seconds: 32,
    narrator: 'Suvrat',
    capture: 'The query language and the plan the engine chose to execute it',
    commands: [
      'python -m nyaya.cli --explain-plan \\',
      '  \'privacy /p "fundamental right" AND bench:>=5 year:>=2015\'',
      'python -m nyaya.query.planner --compare \\',
      '  \'privacy /p "fundamental right" AND bench:>=5 year:>=2015\'',
    ],
    mustShow: [
      'the parsed tree: PROX same paragraph over TERM(privaci) and PHRASE[fundament right]',
      'the order line PROX_P[126] -> FILTER(bench>=5)[1154] -> FILTER(year>=2015)[2139]',
      'the running column 126 -> 66 -> 16 -> 9',
      'the ordering experiment: same answer True, compared 3,375 vs 3,449',
    ],
  },
  {
    kind: 'explain',
    id: 'boolean-plan',
    seconds: 15,
    narrator: 'Suvrat',
    beat:
      'INSERT over the plan: the three operand bars, sized by the real df the ' +
      'terminal just printed, physically re-sort so the rarest goes first, and the ' +
      'candidate count walks down 126 -> 66 -> 15 -> 8.',
  },

  // ------------------------------------------- the analyzer and statutes
  {
    kind: 'cam',
    id: 'analyzer',
    seconds: 10,
    person: 'Sumedha',
    topic: 'The analyzer, the parser and the scorers',
    beats: [
      'Say what you own: analyze(), the query language, tf-idf and BM25F, why-this-result, the page.',
      'End on "documents and queries go through the same function".',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-analyzer',
    seconds: 22,
    narrator: 'Sumedha',
    capture: 'The statute canonicaliser and analyze(), printing their own output',
    commands: [
      'python -m nyaya.statutes --spans \\',
      '  "convicted u/s 302 I.P.C. and under Section 34, and Art. 21 of the Constitution"',
      'python -m nyaya.statutes --crosswalk "ipc§498a"',
      'python -m nyaya.cli --analyze "convicted u/s 302 IPC read with S. 34"',
    ],
    mustShow: [
      'three different surface forms resolving to ipc§302, ipc§34, const§21',
      'inherited=True on "under Section 34" - the act carried over from the previous span',
      'ipc§498a -> [bns§85, bns§86] and the NOT one-to-one note',
      'the token table with zone, pos, sent, para',
    ],
  },
  {
    kind: 'explain',
    id: 'three-ways',
    seconds: 14,
    narrator: 'Sumedha',
    beat:
      'INSERT over the span output: three surface forms of one section diverge into ' +
      'three postings lists, then fuse into ipc§302 - the picture of what the ' +
      'terminal just did to the sentence.',
  },

  // ------------------------------------------- the index and proximity
  {
    kind: 'cam',
    id: 'index',
    seconds: 10,
    person: 'Rakshit',
    topic: 'The zoned positional index and the proximity operators',
    beats: [
      'Say what you own: ingest and zones, SPIMI, postings, /n /s /p, snippets, the eval harness.',
      'End on "one posting carries where, how often per zone, and exactly where".',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-index',
    seconds: 28,
    narrator: 'Rakshit',
    capture: 'A real postings list, then the real two-pointer merge, traced',
    commands: [
      'python -m nyaya.index.positional term "ipc§302"',
      'python -m nyaya.cli --merge \'privacy /p "fundamental right"\' --trace',
    ],
    mustShow: [
      'df 1081, idf 2.5520, skip stride 32 (34 skip pointers)',
      'per-zone tf and the sent / para ids under each row',
      'the pointer walk inside doc 239: left 536, 7027 against the phrase positions',
      'pairs accepted 8, smallest gap 2 tokens, and the first pairs line',
    ],
  },
  {
    kind: 'explain',
    id: 'proximity-walk',
    seconds: 16,
    narrator: 'Rakshit',
    beat:
      'INSERT over the trace: doc 239’s two real position lists walked pointer by ' +
      'pointer, comparing paragraph ids, until 536 meets 533 in paragraph 8.',
  },

  // ------------------------------------------- the citation graph
  {
    kind: 'cam',
    id: 'authority',
    seconds: 10,
    person: 'Suvrat',
    topic: 'The citation graph, treatment edges and weighted PageRank',
    beats: [
      'Say what you own: parametric index, citation extraction, PageRank, g(d), planner, net score, tiers.',
      'End on "an edge is a verdict, not a vote" - this is the novelty.',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-authority',
    seconds: 28,
    narrator: 'Suvrat',
    capture: 'The graph we actually built, and the g(d) that treatment changed',
    commands: [
      'python -m nyaya.authority.citations stats',
      'python -m nyaya.authority.pagerank compare',
    ],
    mustShow: [
      '76,958 occurrences, 66.00% resolved, 76,118 edges, and the resolution-rung table',
      'nodes 37,210 and the status line {ok 37033, overruled 91, distinguished 86}',
      'Maneka Gandhi is NOT in this table any more - it was a stale-store false positive',
      'Pune Municipal Corporation 2014: cited 73, ovr 17, g 0.377 -> 0.087',
      'T. Devadasan 1964: cited 61, ovr 3, g 0.460 -> 0.115',
      'the (!) footnote: 67 of 91 rest on a single overruling citation',
    ],
  },
  {
    kind: 'explain',
    id: 'overruled-trap',
    seconds: 15,
    narrator: 'Suvrat',
    beat:
      'INSERT over the compare table: Pune Municipal Corporation’s authority bar ' +
      'fills from 73 citations, then one red overruled edge arrives and drains it ' +
      'from 0.377 to 0.087 - the row the terminal is showing, animated.',
  },

  // ------------------------------------------- the three measurement tracks
  {
    kind: 'screen',
    id: 'drive-tracka',
    seconds: 32,
    narrator: 'Rakshit',
    capture: 'The judged evaluation running live across all six systems',
    commands: ['python -m nyaya.eval.run'],
    mustShow: [
      'the six per-system timing lines as they print',
      'the full Track A table: P@5, P@10, R@20, MAP@20, nDCG@10 for B0 through S4',
      'S3 0.8833 and S4 0.8786 side by side - S4 does NOT win here',
      'the sign test 17 improved / 0 tied / 3 worsened, p = 0.0026 SIGNIFICANT',
      'the bootstrap 95% CI [+0.1623, +0.3781], excludes 0',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-ablation',
    seconds: 20,
    narrator: 'Rakshit',
    capture: 'One thing switched off at a time \u2014 what the domain work is actually worth',
    commands: ['python -m nyaya.rank.ablate --run'],
    mustShow: [
      'the four arms with their d nDCG column',
      'nocanon at -0.1205 - by far the largest single loss',
      'nostem at -0.0117, and its vocabulary rising 544,787 -> 618,852',
      'the act-inheritance note underneath: query side only, +0.0000, a PARTIAL ablation',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-tracksbc',
    seconds: 26,
    narrator: 'Suvrat',
    capture: 'The two tracks that need no human judge',
    commands: [
      'python -m nyaya.eval.citation_recovery report',
      'python -m nyaya.eval.efficiency report',
    ],
    mustShow: [
      'the R@10 column: S3 0.176 best, and G0 0.012 marked "no query signal"',
      'excluded as unretrievable: 45.2%, and DISTINCTIVE leak 0.00%',
      'documents scored p50 1,030 and the warm p50 column at 4-7 ms',
      'NOTE: the tier block here is the HARNESS path (12/20, 3/10). The shipped',
      'path is 18/20, top-1 20/20, worst case 9/10 - do not narrate the 3/10',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-brief',
    seconds: 20,
    narrator: 'Sumedha',
    capture: 'The Precedent Brief, shipped extractive, with its verifier',
    commands: [
      'python -m nyaya.cli --brief \\',
      '  "dying declaration as the sole basis of conviction"',
    ],
    mustShow: [
      'the summary with a [Pn] tag on every single sentence',
      'the sources block with real paragraph ids and window ids',
      'the grounding check: composer EXTRACTIVE, threshold 0.20, 6 verified, 0 flagged',
      'the per-sentence cosine column',
    ],
  },
  {
    kind: 'screen',
    id: 'drive-limits',
    seconds: 19,
    narrator: 'Rakshit',
    capture: 'The classifier audit, errors only - the limitation, on screen',
    commands: ['python -m nyaya.cli --treatment-audit --only-errors --limit 3'],
    mustShow: [
      'accuracy 92.5%, edges judged 40 of 40',
      'the per-label block - mentioned recall 77%, so every error runs upward',
      'all three error rows: every one is gold=mentioned read as something stronger',
      'the "documented limitation" line at the bottom',
    ],
  },

  // ------------------------------------------------------------------ close
  {
    kind: 'cam',
    id: 'close',
    seconds: 9,
    person: 'Sumedha',
    topic: 'Closing',
    beats: [
      'One sentence on what is genuinely new: treatment-aware authority.',
      'One sentence on what you would fix first.',
      'Thank the viewer; do not list features again.',
    ],
  },
  {
    kind: 'explain',
    id: 'outro',
    seconds: 7,
    narrator: 'Suvrat',
    beat: 'Mark, names, corpus line, licence. Seven seconds, no bullets.',
  },
];

export const FPS = 30;

/** Frame offsets, so a scene can be located without re-adding the array. */
export const LAYOUT = (() => {
  let at = 0;
  return TIMELINE.map((s) => {
    const from = Math.round(at * FPS);
    at += s.seconds;
    return {...s, from, durationInFrames: Math.round(s.seconds * FPS)};
  });
})();

export const TOTAL_SECONDS = TIMELINE.reduce((a, s) => a + s.seconds, 0);
export const TOTAL_FRAMES = LAYOUT.reduce((a, s) => Math.max(a, s.from + s.durationInFrames), 0);

export const mmss = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};
