/**
 * The edit, as data.
 *
 * One array, three kinds of segment: `explain` (animation), `screen` (a gap for
 * a live screen recording) and `cam` (a gap for a webcam take). Durations are in
 * seconds; the composition length is their sum, so the running time can never
 * drift from the script.
 *
 * Target: 7:00 - 7:30. Hard limits for the hackathon are 5:00 - 8:00.
 */
import type {Person} from './data';
import type {CamSlotProps, ScreenSlotProps} from './components/Slots';

export type Segment =
  | {kind: 'explain'; id: string; seconds: number; narrator: Person; beat: string}
  | ({kind: 'screen'; id: string; seconds: number; narrator: Person} & Omit<ScreenSlotProps, 'seconds' | 'name'>)
  | ({kind: 'cam'; id: string; seconds: number} & Omit<CamSlotProps, 'seconds' | 'slug'>);

export const TIMELINE: Segment[] = [
  // ---------------------------------------------------------------- act 1
  {
    kind: 'explain',
    id: 'title',
    seconds: 17,
    narrator: 'Rakshit',
    beat: 'Title, course, track, three names.',
  },
  {
    kind: 'cam',
    id: 'intro',
    seconds: 13,
    person: 'Rakshit',
    topic: 'Team introduction',
    beats: [
      'All three on camera if you can fit, otherwise Rakshit alone.',
      'Name the project, the course, the track, and the one-line claim.',
      'Hand off: "here is why precedent search is not web search".',
    ],
  },

  // ---------------------------------------------------------------- act 2
  {
    kind: 'explain',
    id: 'three-ways',
    seconds: 20,
    narrator: 'Sumedha',
    beat: 'Three surface forms of one section diverge, then fuse into ipc-302.',
  },
  {
    kind: 'explain',
    id: 'anatomy',
    seconds: 16,
    narrator: 'Rakshit',
    beat: 'A page of running text separates into six zones with their weights.',
  },
  {
    kind: 'explain',
    id: 'overruled-trap',
    seconds: 17,
    narrator: 'Suvrat',
    beat: 'A heavily cited node is drained by one red overruling edge.',
  },
  {
    kind: 'cam',
    id: 't6',
    seconds: 10,
    person: 'Suvrat',
    topic: 'Why this is Track T6',
    beats: [
      'Domain structure: zones, metadata, citations, authority.',
      'One sentence only, then hand straight to the demo.',
    ],
  },

  // ---------------------------------------------------------------- act 3
  {
    kind: 'screen',
    id: 'demo-freetext',
    seconds: 18,
    narrator: 'Sumedha',
    capture: 'Free-text query on the real index, top ten with treatment badges',
    commands: ['python -m nyaya.cli --mode S4 "right to privacy of personal data"'],
    mustShow: ['the typed query', 'ten ranked titles with year and bench', 'at least one badge'],
  },
  {
    kind: 'screen',
    id: 'demo-operators',
    seconds: 18,
    narrator: 'Sumedha',
    capture: 'The same query language doing real work: /p, a phrase, and two filters',
    commands: [
      'python -m nyaya.cli --explain-plan \\',
      '  \'privacy /p "fundamental right" AND bench:>=5 year:>=2015\'',
    ],
    mustShow: ['the parsed AST', 'the df-ordered plan', 'candidate counts shrinking'],
  },
  {
    kind: 'screen',
    id: 'demo-why',
    seconds: 15,
    narrator: 'Sumedha',
    capture: 'One result card opened: snippet, treatment badge, "why this result"',
    commands: ['python -m nyaya.cli --why 1 --mode S4 \'privacy /p "fundamental right"\''],
    mustShow: [
      'per-zone contributions',
      'top terms',
      'pagerank, bench, penalty, net',
      'the evidence sentence behind the badge',
    ],
  },

  // ---------------------------------------------------------------- act 4
  {
    kind: 'explain',
    id: 'pipeline-map',
    seconds: 14,
    narrator: 'Suvrat',
    beat: 'The whole architecture, colour-coded by owner.',
  },
  {
    kind: 'cam',
    id: 'analyzer',
    seconds: 11,
    person: 'Sumedha',
    topic: 'The analyzer, the parser and the scorers',
    beats: [
      'Say what you own: analyze(), the query language, tf-idf and BM25F, why-this-result.',
      'End on "documents and queries go through the same function".',
    ],
  },
  {
    kind: 'explain',
    id: 'statute-canon',
    seconds: 16,
    narrator: 'Sumedha',
    beat: 'Four analyzer stages transform real tokens; then the IPC-BNS crosswalk.',
  },
  {
    kind: 'screen',
    id: 'analyze-parse',
    seconds: 12,
    narrator: 'Sumedha',
    capture: 'analyze() and parse() printing their actual output',
    commands: [
      'python -m nyaya.cli --analyze "convicted u/s 302 IPC read with S. 34"',
      'python -m nyaya.cli --parse \'bail! AND act:IPC year:>=2010\'',
    ],
    mustShow: ['the token list with zone, pos, sent, para', 'the AST with expansions'],
  },
  {
    kind: 'cam',
    id: 'index',
    seconds: 11,
    person: 'Rakshit',
    topic: 'The zoned positional index and the proximity operators',
    beats: [
      'Say what you own: ingest and zones, SPIMI, postings, /n /s /p, snippets, the eval harness.',
      'End on "one posting carries where, how often per zone, and exactly where".',
    ],
  },
  {
    kind: 'explain',
    id: 'zoned-postings',
    seconds: 17,
    narrator: 'Rakshit',
    beat: 'Dictionary opens into a postings list; zone tf, positions, skip arcs.',
  },
  {
    kind: 'explain',
    id: 'proximity-walk',
    seconds: 16,
    narrator: 'Rakshit',
    beat: 'Two position lists walked pointer by pointer for /p.',
  },
  {
    kind: 'screen',
    id: 'postings-plan',
    seconds: 13,
    narrator: 'Rakshit',
    capture: 'The real postings list and the real merge, printed',
    commands: [
      'python -m nyaya.cli --postings privacy --limit 12',
      'python -m nyaya.cli --merge \'privacy /p "fundamental right"\' --trace',
    ],
    mustShow: ['doc ids with per-zone tf and positions', 'the skip stride', 'pointer steps'],
  },
  {
    kind: 'explain',
    id: 'boolean-plan',
    seconds: 13,
    narrator: 'Suvrat',
    beat: 'Operand bars re-sort by df; candidate counts fall faster.',
  },
  {
    kind: 'explain',
    id: 'bm25f',
    seconds: 15,
    narrator: 'Sumedha',
    beat: 'Raw tf runs off the chart; BM25F saturates. Zones combine before saturation.',
  },
  {
    kind: 'cam',
    id: 'authority',
    seconds: 11,
    person: 'Suvrat',
    topic: 'The citation graph, treatment edges and weighted PageRank',
    beats: [
      'Say what you own: parametric index, citation extraction, PageRank, g(d), planner, net score, tiers.',
      'End on "an edge is a verdict, not a vote" - this is the novelty.',
    ],
  },
  {
    kind: 'explain',
    id: 'citation-graph',
    seconds: 20,
    narrator: 'Suvrat',
    beat: 'Edges get treatments, power iteration runs, the overruled penalty lands.',
  },
  {
    kind: 'explain',
    id: 'net-score',
    seconds: 13,
    narrator: 'Suvrat',
    beat: 'The three streams assemble into net = a.text + (1-a).g + b.prox.',
  },

  // ---------------------------------------------------------------- act 5
  {
    kind: 'cam',
    id: 'eval',
    seconds: 10,
    person: 'Rakshit',
    topic: 'How we evaluated it',
    beats: [
      'Six systems, one flag apart; judged queries plus two automatic tracks.',
      'Say out loud that the person who judged did not see which system retrieved what.',
    ],
  },
  {
    kind: 'screen',
    id: 'eval-run',
    seconds: 14,
    narrator: 'Rakshit',
    capture: 'The evaluation harness running across all six systems',
    commands: [
      'python -m nyaya.eval.run --systems B0,S1,S2,S2p,S3,S4 --qrels eval/queries',
      'python -m nyaya.eval.operators --brute-force',
    ],
    mustShow: ['per-system metric rows as they print', 'the operator pass/fail summary'],
  },
  {
    kind: 'explain',
    id: 'eval-ladder',
    seconds: 18,
    narrator: 'Rakshit',
    beat: 'The B0 to S4 ladder with the P@5 / P@10 / R@20 / MAP / nDCG@10 table.',
  },
  {
    kind: 'explain',
    id: 'operator-speed',
    seconds: 13,
    narrator: 'Suvrat',
    beat: 'Operator set agreement against brute force; tiered speed-up; citation recovery.',
  },

  // ---------------------------------------------------------------- act 6
  {
    kind: 'explain',
    id: 'limitation',
    seconds: 18,
    narrator: 'Rakshit',
    beat: 'A negated cue phrase fools the treatment classifier; then the roadmap.',
  },
  {
    kind: 'screen',
    id: 'limitation-live',
    seconds: 10,
    narrator: 'Rakshit',
    capture: 'The mis-labelled edge, live, with the sentence that caused it',
    commands: ['python -m nyaya.cli --treatment-audit --only-errors --limit 3'],
    mustShow: ['the negated sentence', 'the predicted label', 'the gold label'],
  },

  // ---------------------------------------------------------------- act 7
  {
    kind: 'explain',
    id: 'precedent-brief',
    seconds: 17,
    narrator: 'Sumedha',
    beat: 'Paragraph BM25, forced paragraph ids, cosine verifier flagging one sentence.',
  },
  {
    kind: 'cam',
    id: 'close',
    seconds: 12,
    person: 'Sumedha',
    topic: 'Closing',
    beats: [
      'One sentence on what is genuinely new: treatment-aware authority.',
      'One sentence on what you would do next.',
      'Thank the reader; do not list features again.',
    ],
  },
  {
    kind: 'explain',
    id: 'outro',
    seconds: 9,
    narrator: 'Sumedha',
    beat: 'Mark, names, corpus line, licence.',
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
