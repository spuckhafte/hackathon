/**
 * ============================================================================
 *  MEASURED - every figure below was produced by running the system.
 * ============================================================================
 *
 *  EVERY number, score, timing and count that the video SPEAKS OR SHOWS lives
 *  in this file and nowhere else. No component may hard-code a figure.
 *
 *  Anything whose `measured` flag is false is still a PLACEHOLDER: the renderer
 *  draws it with a dashed amber underline so that an un-updated figure is
 *  impossible to miss in a review render.
 *
 *  Provenance for everything marked measured:
 *    runs/trackA.json          python -m nyaya.eval.run
 *    runs/citation_recovery.json  python -m nyaya.eval.citation_recovery report
 *    runs/efficiency.json      python -m nyaya.eval.efficiency report
 *    runs/operators.json       python -m nyaya.eval.operators --brute-force
 *    runs/corpus_stats.json    python -m nyaya.index.stats --no-growth
 *    live CLI                  nyaya.cli / nyaya.authority.* / nyaya.index.*
 *
 *  Where a figure disagrees with an earlier hand-written note, the value that
 *  the tool printed wins and the note is recorded in the comment. Nothing here
 *  is rounded in the system's favour.
 * ============================================================================
 */

/**
 * Master switch. True, and there are no exceptions left: every figure in this
 * file was produced by running the system, and all of them were re-verified
 * after the rebuild of S2, both authority stores and S5, and after S5 was wired
 * into the live `search()` path.
 *
 * If a figure ever loses its measurement, flip its `measured` flag to false
 * rather than deleting it - the renderer then draws it with an amber dashed
 * underline and it cannot survive a review render.
 */
export const MEASURED = true;

/** A figure that is still a placeholder renders with a dashed amber underline. */
export type Figure = {value: string; measured?: boolean};
export const fig = (value: string, measured = false): Figure => ({value, measured});

// ---------------------------------------------------------------------------
// 1. Corpus and build
// ---------------------------------------------------------------------------
export const CORPUS = {
  /** English judgment rows in the AWS Open Data release. */
  universe: fig('42,371', true),
  /** After de-duplicating on bucket path: the citation-graph nodes. */
  graphNodes: fig('37,210', true),
  /** 42,371 - 37,210 = 5,161 rows were the same judgment, once per language. */
  duplicateShare: fig('12.2%', true),

  /** Stratified sample actually indexed. */
  indexed: fig('13,878', true),
  indexedExact: fig('13,878', true),
  manifest: fig('13,942', true),
  extractionRate: fig('99.54%', true),
  /** 53 scanned, 7 too short, 2 non-English, 2 missing from the bucket. */
  dropped: fig('64', true),
  perYearCap: fig('200', true),
  years: fig('1950 - 2025', true),
  seed: fig('358', true),
  licence: fig('AWS Open Data | CC-BY-4.0', true),

  // --- runs/corpus_stats.json, python -m nyaya.index.stats --no-growth ---
  paragraphs: fig('489,627', true),
  meanParagraphs: fig('35.3', true),
  /** index tokens per document, mean / median */
  meanLength: fig('6,030', true),
  medianLength: fig('4,161', true),
  tokens: fig('83,685,567', true),
  vocabulary: fig('544,787', true),
  postings: fig('11,943,968', true),
  hapaxShare: fig('68.6%', true),

  /** dictionary.json + postings.bin + docmeta.npz */
  s3IndexMB: fig('717.9 MB', true),
  /** every store, including the B0 baseline index and the S5 champion lists */
  allStores: fig('1.9 GB', true),
  indexToTextRatio: fig('1.29', true),
  buildSeconds: fig('604 s', true),
  /** fresh interpreter, S4: memory-map the stores, then the first query */
  coldLoadMs: fig('1,249 ms', true),
  coldFirstQueryMs: fig('977 ms', true),
  warmQueryMs: fig('2.8 ms', true),
  /** two builds from identical inputs differed by 36 and 56 vocabulary types */
  buildDeterminism: fig('not bit-deterministic: vocabulary varies by ~50 types between builds', true),

  // --- citation graph ---
  citationOccurrences: fig('76,958', true),
  citationEdges: fig('76,118', true),
  resolvedPct: fig('66.0%', true),
  overruledCases: fig('91', true),
  distinguishedCases: fig('86', true),

  /** share of all indexed tokens that fall in the `body` zone */
  bodyTokenShare: fig('84.8%', true),
  /** documents with at least one statute token in the statutes zone */
  statuteDocs: fig('10,924', true),
  statuteDocShare: fig('78.7%', true),
  statuteTokens: fig('61,314', true),
  distinctActs: fig('110', true),
  /** judgments that print their own paragraph numbers; the rest use windows */
  ownParaNumbering: fig('28.2%', true),
  windowWords: fig('110', true),

  // --- laws of the collection ---
  heaps: fig('|V| = 3.33 . T^0.653   R2 0.9957', true),
  zipf: fig('s = 0.9825   R2 0.9930', true),
};

/** Document-frequency probes the film puts on screen. */
export const DF_PROBES = [
  {term: 'ipc§302', df: fig('1,081', true), note: 'one statute token, three surface forms'},
  {term: 'const§21', df: fig('623', true), note: 'Article 21'},
  {term: 'privaci', df: fig('126', true), note: 'the whole privacy line'},
  {term: 'surveil', df: fig('58', true), note: 'rare enough to drive a plan'},
  {term: 'court', df: fig('13,877', true), note: 'in every judgment but one - idf 0.000'},
] as const;

// ---------------------------------------------------------------------------
// 2. Architecture constants - mirrored from nyaya/config.py.
// ---------------------------------------------------------------------------
export const CONFIG = {
  zones: ['title', 'parties', 'bench', 'statutes', 'order', 'body'] as const,
  /**
   * `title` is 0.5, not the textbook 3.0. Lowered on measured evidence: a
   * Supreme Court cause title is a party name, not a subject line, so a query
   * term almost never appears in one and amplifying the rare hit hurt S2 on
   * both splits. BM25F was insensitive, so the change is free for S4.
   */
  zoneWeights: {title: 0.5, parties: 0.5, bench: 0.0, statutes: 2.0, order: 1.5, body: 1.0},
  bm25fK1: fig('1.2', true),
  bm25fB: {title: 0.3, parties: 0.3, bench: 0.3, statutes: 0.3, order: 0.5, body: 0.75},
  alpha: fig('0.80', true),
  beta: fig('0.05', true),
  pagerankLambda: fig('0.85', true),
  pagerankIterations: fig('60', true),
  treatmentWeights: {followed: 1.0, mentioned: 0.6, distinguished: 0.3, overruled: 0.0},
  treatmentPenalty: {ok: 1.0, distinguished: 0.7, overruled: 0.25},
  gPagerankWeight: fig('0.75', true),
  gBenchWeight: fig('0.25', true),
  championListSize: fig('400', true),
  topK: fig('10', true),
  maxBench: fig('13', true),
  spimiBlockDocs: fig('600', true),
  idfElimination: fig('0.35', true),
  manyTerms: fig('4', true),
  distinguishedShare: fig('34%', true),

  /**
   * The tuner ran and we REJECTED its answer. Coordinate ascent, 2 passes, 57
   * configurations, fitted on the five dev queries only - the CLI refuses
   * `--split test`. It moved neither alpha nor beta: the shipped 0.80 / 0.05 is
   * already the grid optimum. Its only proposal was statutes 2.0 -> 0.0 and
   * order 1.5 -> 1.0, which gained +0.0045 dev nDCG@10 and cost -0.0200 on the
   * 20 test queries. Five queries is not enough to move a weight on, and
   * config.py records why the proposal was refused.
   */
  tunedWeights: fig('shipped unchanged: alpha 0.80, beta 0.05, zone weights as above', true),
  tuningRange: fig(
    'title [0 .. 3] \u00b7 parties [0 .. 1] \u00b7 statutes [0 .. 4] \u00b7 order [0.5 .. 3] \u00b7 '
      + 'alpha [0.60 .. 1.00] \u00b7 beta [0.00 .. 0.20]',
    true
  ),
  tuningRejected: fig(
    'the dev optimum gained +0.0045 on dev and lost -0.0200 on test, so it was not shipped',
    true
  ),
};

/**
 * One flag off at a time, 20 test queries, mode S4. `python -m nyaya.rank.ablate
 * --run`, written to runs/ablations.json.
 *
 * The vocabulary column is NOT precise to better than about 100 types: the build
 * is not bit-deterministic, and two builds from identical inputs differed by 36
 * and 56. `noinherit` is 56 types from `full` and that difference is noise.
 */
export const ABLATIONS = [
  {arm: 'full', what: 'the shipped analyser', ndcg: fig('0.8786', true), delta: fig('\u2014', true),
   map: fig('0.5473', true), p10: fig('0.933', true), vocab: fig('544,787', true)},
  {arm: 'nostem', what: 'Porter stemming off, both sides', ndcg: fig('0.8669', true),
   delta: fig('-0.0117', true), map: fig('0.5137', true), p10: fig('0.873', true),
   vocab: fig('618,852', true)},
  {arm: 'nocanon', what: 'statute canonicalisation off', ndcg: fig('0.7582', true),
   delta: fig('-0.1205', true), map: fig('0.4631', true), p10: fig('0.783', true),
   vocab: fig('539,096', true)},
  {arm: 'noinherit', what: 'act inheritance off in the body zone', ndcg: fig('0.8670', true),
   delta: fig('-0.0116', true), map: fig('0.5473', true), p10: fig('0.933', true),
   vocab: fig('544,843', true)},
];

export const ABLATION_NOTES = {
  headline: fig('statute canonicalisation is worth -0.1205 nDCG@10 \u2014 by far the largest', true),
  whereItLands: fig('on the statute-anchored queries: q23 0.7761 -> 0.0000, q14 0.9545 -> 0.2440, q15 0.8128 -> 0.4053', true),
  stemming: fig('stemming buys +0.0117 nDCG and a 14.5% smaller vocabulary', true),
  partial: fig(
    'noinherit is a PARTIAL ablation and says so in the code: the statutes zone is '
      + 'synthesised from tokens computed at ingest, so a true end-to-end number needs a re-ingest',
    true
  ),
  vocabNoise: fig('vocabulary deltas under ~100 types are build noise, not signal', true),
};

/** Why the title zone is weighted 0.5 and not 3.0 - the evidence, not the taste. */
export const TITLE_EVIDENCE = {
  zeroHits: fig('arbitr df 883, const\u00a721-style terms aside: 0 documents with a title-zone hit', true),
  examples: fig('arbitr 883/0 \u00b7 detent 765/0 \u00b7 privaci 126/0 title hits', true),
  partyNames: fig('what IS in a title: state 4,946 \u00b7 union 1,450', true),
  consequence: fig('title:(arbitration) returns 0 results; plain arbitration returns 10', true),
};

// ---------------------------------------------------------------------------
// 3. Track A - judged queries. runs/trackA.json, split=test.
// ---------------------------------------------------------------------------
export type SystemRow = {
  id: string;
  label: string;
  blurb: string;
  p5: Figure;
  p10: Figure;
  r20: Figure;
  map: Figure;
  ndcg10: Figure;
};

export const SYSTEMS: SystemRow[] = [
  {
    id: 'B0',
    label: 'B0',
    blurb: 'flat tf-idf lnc.ltc, plain tokenizer',
    p5: fig('0.790', true), p10: fig('0.745', true), r20: fig('0.625', true),
    map: fig('0.537', true), ndcg10: fig('0.610', true),
  },
  {
    id: 'S1',
    label: 'S1',
    blurb: 'plus statute canonicalisation and a legal stop list',
    p5: fig('0.860', true), p10: fig('0.855', true), r20: fig('0.690', true),
    map: fig('0.597', true), ndcg10: fig('0.642', true),
  },
  {
    id: 'S2',
    label: 'S2',
    blurb: 'plus weighted zone scoring',
    p5: fig('0.900', true), p10: fig('0.850', true), r20: fig('0.645', true),
    map: fig('0.581', true), ndcg10: fig('0.671', true),
  },
  {
    id: 'S2p',
    label: 'S2′',
    blurb: 'BM25F instead of cosine',
    p5: fig('0.960', true), p10: fig('0.910', true), r20: fig('0.767', true),
    map: fig('0.707', true), ndcg10: fig('0.798', true),
  },
  {
    id: 'S3',
    label: 'S3',
    blurb: 'plus authority from plain PageRank - best on both judged tracks',
    p5: fig('0.970', true), p10: fig('0.920', true), r20: fig('0.774', true),
    map: fig('0.723', true), ndcg10: fig('0.883', true),
  },
  {
    id: 'S4',
    label: 'S4',
    blurb: 'plus treatment-aware g(d) and proximity - level with S3 on these metrics',
    p5: fig('0.970', true), p10: fig('0.920', true), r20: fig('0.771', true),
    map: fig('0.721', true), ndcg10: fig('0.879', true),
  },
];

/** Bar heights for the nDCG@10 ladder, normalised against the best row. */
export const LADDER_HEIGHTS: Record<string, number> = {
  B0: 0.610 / 0.883,
  S1: 0.642 / 0.883,
  S2: 0.671 / 0.883,
  S2p: 0.798 / 0.883,
  S3: 0.883 / 0.883,
  S4: 0.879 / 0.883,
};

export const EVAL = {
  nQueries: fig('20', true),
  nQueriesTotal: fig('25', true),
  nHeldOut: fig('5', true),
  nJudged: fig('373', true),
  gradedScale: fig('0 - 3', true),
  /** headline deltas, S4 over B0, on the 20 test queries */
  deltaNdcg: fig('+0.2686', true),
  deltaMap: fig('+0.184', true),
  deltaP10: fig('+0.175', true),
  /** nDCG@10, 20 test queries */
  signTest: fig('17 improved, 0 tied, 3 worsened \u2014 p = 0.0026', true),
  bootstrapCI: fig('[+0.1623, +0.3781], 10,000 iterations, excludes 0', true),
  /** the same test on the other two headline metrics */
  signTestMap: fig('15 / 0 / 5 \u2014 p = 0.0414', true),
  signTestP10: fig('10 / 8 / 2 \u2014 p = 0.0386', true),
  /**
   * The honest reading. S4 does not beat S3 here and we do not claim it does -
   * what S4 buys is not recommending dead law, and recall@k cannot see that.
   */
  s4VersusS3: fig('S4 = S3 within noise; the gain S4 exists for is invisible to recall@k', true),
};

// ---------------------------------------------------------------------------
// 4. Track B - automatic citation recovery. runs/citation_recovery.json.
// ---------------------------------------------------------------------------
/** Track B, refreshed after the rebuild. runs/citation_recovery.json. */
export const CITATION_RECOVERY = {
  nQueries: fig('198', true),
  goldPairs: fig('1,222', true),
  recallAt10: {
    B0: fig('0.1465', true),
    S1: fig('0.0603', true),
    S2: fig('0.0782', true),
    S2p: fig('0.1571', true),
    S3: fig('0.1760', true),
    S4: fig('0.1734', true),
    G0: fig('0.0115', true),
  },
  recallAt50: {
    B0: fig('0.3003', true),
    S1: fig('0.1539', true),
    S2: fig('0.1559', true),
    S2p: fig('0.3067', true),
    S3: fig('0.3392', true),
    S4: fig('0.3404', true),
    G0: fig('0.0352', true),
  },
  best: fig('S3 on R@10 and MRR, S4 on R@50 \u2014 level, as on Track A', true),
  /** G0 ranks by authority alone and never reads the query at all. */
  authorityOnlyFloor: fig('0.0115 \u2014 about 7% of S3, so the circularity objection is bounded', true),
  unretrievable: fig('45.2% of resolved citations point at judgments never sampled into the index', true),
  leak: fig('0.00% distinctive leak, 0 of 1,222 gold pairs; topical overlap 6.4%', true),
  ceilingNote: fig('recall is a lower bound: being cited is not the same as being relevant', true),
  /** the champion lists were free on this track, which is worth saying out loud */
  s5CostNothing: fig(
    'wiring the champion lists into live search cost zero recall: across all six systems '
      + 'not one query lost R@10 or R@50',
    true
  ),
};

// ---------------------------------------------------------------------------
// 5. Operator correctness. runs/operators.json, --brute-force, 150.7 s.
// ---------------------------------------------------------------------------
export const OPERATORS = {
  nCases: fig('94', true),
  /** vs an independent naive token scan: no postings, no skips, no planner */
  agreement: fig('94/94 = 100.00%', true),
  docsVerified: fig('1,200', true),
  /**
   * vs a raw surface regex. Report this PER DOCUMENT, not per check: a check
   * that differs on 1 document of 1,092 counts the same as one that differs on
   * all of them, so the per-check figure (27/44) overstates the disagreement.
   */
  regexAgreement: fig('2,740/2,888 = 94.88% of document decisions', true),
  regexExplained: fig(
    '148 differ: 135 stemming, 9 regex prefix over-match, 3 token count, 1 OCR, 0 unexplained',
    true
  ),
  families: ['phrase', '/1 /3 /5 /10 /25', '/s', '/p', 'AND OR NOT', 'root!', 'filters', 'zone:(...)'] as const,
};

// ---------------------------------------------------------------------------
// 6. Track C - efficiency. runs/efficiency.json.
// ---------------------------------------------------------------------------
/**
 * Track C, refreshed with S5 wired into the live `search()` path.
 *
 * ONE RULE, AND IT IS NOT NEGOTIABLE: **never quote an end-to-end latency
 * speed-up for S5.** Three runs of identical code gave monotonically falling
 * wall clock as the OS page cache warmed (B0 went 6.18 -> 5.22 -> 4.00 ms) while
 * the algorithmic columns were identical every time. Any latency ratio would be
 * measuring the page cache, not the champion lists.
 *
 * Quote the algorithmic reduction instead. It is exact and reproducible, and it
 * is what the optimisation actually does.
 */
export const SPEED = {
  candidatesBefore: fig('12,548', true),
  candidatesAfter: fig('1,059', true),
  candidatesGain: fig('11.8x fewer candidates at p50', true),
  scoredBefore: fig('8,032', true),
  scoredAfter: fig('1,030', true),
  scoredGain: fig('7.8x fewer documents scored at p50', true),
  totalBefore: fig('159,328', true),
  totalAfter: fig('17,718', true),
  totalGain: fig('8.99x fewer over the whole 20-query set', true),

  /** latency is given as a range, never as a ratio */
  warmP50: fig('4 - 7 ms warm, across all six systems', true),
  coldStores: fig('1.2 - 1.5 s to memory-map the stores', true),
  coldFirstQuery: fig('about 1 s for the first query', true),
  latencyCaveat: fig(
    'wall clock moved run to run on unchanged code as the page cache warmed, so no '
      + 'end-to-end speed-up is claimed',
    true
  ),

  /**
   * Tier fidelity on the SHIPPED path - this is the one to quote. The harness
   * arm reports 12/20 and a 3/10 worst case because it builds candidates from
   * `scoring_terms`, so "appellate jurisdiction of the supreme court" collapses
   * to the single term `suprem` and tier 1 is one champion list. `search()`
   * builds from `free_terms`, so tier 1 is a union of six.
   */
  tierScored: fig('159,328 -> 17,718 documents scored, 8.99x fewer', true),
  tierLatency: fig('mean 10.4 -> 4.2 ms', true),
  tierIdentical: fig('top-10 identical 18/20, mean overlap 0.990', true),
  tierTop1: fig('rank 1 unchanged 20 of 20', true),
  tierFellThrough: fig('fell through to tier 2 on 0 of 20', true),
  tierWorst: fig('9/10, twice in twenty \u2014 worst is "unlawful activities prevention"', true),
  tierJudgedCost: fig(
    'on the judged set the champion lists cost exactly zero: nDCG@10, MAP and P@10 are '
      + 'identical to four decimals with tiers on and off, in all three authority modes',
    true
  ),
  /** the harness arm, kept so the difference is documented rather than hidden */
  tierHarness: fig('harness path: 78.7% fewer documents, 12/20 identical, worst case 3/10', true),

  eliminationAll: fig('16.5% fewer documents scored, 132,206 -> 110,396', true),
  eliminationFidelity: fig('top-10 identical 16/20, mean overlap 0.985', true),
  tierIsNonSafe: fig('champion lists are not guaranteed to hold the exhaustive top k', true),
};

// ---------------------------------------------------------------------------
// 7. Component accuracy
// ---------------------------------------------------------------------------
export const ACCURACY = {
  statutePrecision: fig('92.5%', true),
  statuteRecall: fig('95.9%', true),
  statuteSample: fig('151 real judgments', true),
  crosswalkRows: fig('120 rows verified against the official GoI tables', true),
  citationNameRung: fig('91.3% precision', true),
  /**
   * 37 of 40. An earlier working note said 97.5%; that belonged to a gold file
   * replaced when the gold keys gained a window fingerprint, and the two files
   * share zero keys. Four historical versions of classify() score 37/40 against
   * the current gold, so nothing regressed - the old number was simply measured
   * against different data.
   */
  treatmentAccuracy: fig('92.5%', true),
  treatmentSampleSize: fig('40', true),
  treatmentErrors: fig('3 of 40, all of them "mentioned" read as something stronger', true),
  treatmentPerLabel: fig(
    'precision: overruled 80%, distinguished 100%, followed 90%, mentioned 100% \u2014 mentioned recall 77%',
    true
  ),
  zoneTitle: fig('33/33', true),
  zoneBench: fig('33/33', true),
  zoneOrder: fig('25/33', true),
};

/** Authority examples the film puts on screen. Both legally verifiable. */
export const AUTHORITY_CASES = [
  {
    name: 'Pune Municipal Corporation',
    year: '2014',
    cited: fig('73', true),
    overruling: fig('17', true),
    gBefore: fig('0.377', true),
    gAfter: fig('0.087', true),
    by: 'overruled by Indore Development Authority (2020)',
  },
  {
    name: 'T. Devadasan',
    year: '1964',
    cited: fig('61', true),
    overruling: fig('3', true),
    gBefore: fig('0.460', true),
    gAfter: fig('0.115', true),
    by: 'overruled on the carry-forward rule',
  },
] as const;

export const AUTHORITY = {
  gDrop: fig('about 75% of g(d)', true),
  singleEdgeFlagged: fig('67 of the 91 rest on a single overruling citation', true),
  movedNodes: fig('11,316 judgments changed g between S3 and S4', true),
  /** both on-screen examples are free of the single-edge (!) marker */
  examplesAreClean: fig('neither Pune nor Devadasan carries the single-edge (!) marker', true),
  manekaCleared: fig(
    'Maneka Gandhi is no longer flagged: status ok, g 0.468, 0 overruling in-edges of 181 \u2014 '
      + 'it was a stale-store false positive and the rebuild cleared it',
    true
  ),
};

/**
 * Limitations the film and the report both state out loud.
 *
 * The last one is the most ordinary and the most likely to recur: two fixes were
 * correct in the code and had still not reached the ranking, because a derived
 * store was older than the corpus it came from.
 */
export const KNOWN_ISSUES = {
  benchReparse: fig(
    'the bench re-parse moved 488 documents: 258 up, 230 down, 184 of them to 1',
    true
  ),
  benchNetCorrecting: fig(
    '163 of the 258 upward corrections went 1 -> more, so the re-parse is net-correcting; '
      + 'the 184 downward are the residual cost',
    true
  ),
  benchFallback: fig(
    'those 184 fall back to the metadata single authoring judge, and g(d) reads them as bench 1',
    true
  ),
  /**
   * What a mis-read bench costs, which is what makes this a limitation rather
   * than a cosmetic defect. bench_norm = min(bench,13)/13 and G_BENCH_WEIGHT is
   * 0.25, so one missing seat is 0.25/13 = 0.0192 of g(d).
   */
  benchSeatCost: fig('each missing bench seat costs 0.0192 of g(d)', true),
  benchFiveAsOne: fig('a 5-judge bench read as 1 understates g by 0.077', true),
  benchThirteenAsOne: fig('a 13-judge bench read as 1 understates g by 0.231', true),
  benchVsOverruled: fig(
    'for scale, the whole overruled penalty on Pune Municipal Corporation is 0.289 \u2014 so one '
      + 'mis-parsed Constitution Bench moves authority by the same order as being overruled',
    true
  ),
  benchOnes: fig(
    '999 of 13,942 documents show bench 1, and roughly a quarter of those are parse failures '
      + 'rather than real single-judge benches',
    true
  ),
  manekaOcr: fig(
    'Maneka Gandhi is a seven-judge bench and prints bench 1: its CORAM OCRs as "M. H. BEG, c. J."',
    true
  ),
  staleStores: fig(
    'Puttaswamy served bench 1 from a parametric index older than the corpus, and Maneka '
      + 'Gandhi was flagged overruled from an authority store built 25 minutes before the '
      + 're-classification pass that fixed it',
    true
  ),
  staleAssertion: fig(
    'runs/efficiency.json still carried the note "S5 is NOT yet wired into search()" after '
      + 'that had become false \u2014 a stale claim inside the measurement itself, caught before it '
      + 'reached the report',
    true
  ),
  /**
   * A silent no-op, and the best bug in the project. The act-inheritance
   * ablation arm reported the shipped numbers to five decimals, which reads as
   * "inheritance does not matter" and actually meant the flag was never off.
   * Two independent causes: a `bool = INHERIT_ACT_DEFAULT` default argument is
   * bound at IMPORT, so reassigning the module constant changed nothing; and
   * the builder fans out over ProcessPoolExecutor, which on Windows spawns by
   * re-importing, so a parent-process flip never reaches a worker.
   */
  silentNoop: fig(
    'an ablation arm that printed the shipped numbers exactly \u2014 a default argument bound at '
      + 'import, plus ProcessPoolExecutor re-importing in each worker, so the flag was never off',
    true
  ),
  silentNoopFix: fig(
    'fixed with call-time resolution and an env var, and pinned by a test that runs a REAL '
      + 'child process, because an in-process test cannot see that class of bug',
    true
  ),
  /** the theme all four share, and the line the limitations beat lands on */
  theme: fig(
    'in a multi-stage index pipeline, a correct fix that never reaches the derived stores is '
      + 'indistinguishable from no fix at all',
    true
  ),
};

/**
 * Precedent Brief. It ships EXTRACTIVE and the film says so plainly.
 *
 * No LLM API key was available in this environment - ANTHROPIC_API_KEY and
 * OPENAI_API_KEY are both unset and the `anthropic` SDK is not installed - so
 * the abstractive path, though written and prompt-complete, HAS NEVER BEEN
 * EXECUTED. Every brief carries a note saying its output is verbatim from the
 * judgments. Showing a working extractive feature with its reach honestly
 * bounded is worth more than an LLM demo we cannot run.
 */
export const BRIEF = {
  composer: fig('extractive \u2014 the abstractive path is written but has never run', true),
  briefsRun: fig('6 real briefs', true),
  sentences: fig('6 sentences from 8 paragraphs across 4-5 judgments', true),
  citedPct: fig('100% \u2014 every sentence carries a [Pn] tag', true),
  flaggedPct: fig('0 flagged', true),
  latency: fig('200 - 1,000 ms', true),
  cosineThreshold: fig('0.20', true),
  topDocs: fig('5', true),
  maxParas: fig('8', true),

  /** the verifier is measured against deliberate fabrications, not asserted */
  realSentences: fig('36 real extractive sentences score cosine 0.9924 - 1.0000', true),
  fakeSentences: fig('18 deliberate fabrications score 0.0866 - 0.8451', true),
  caught: fig('at threshold 0.20 it catches 4 of 18 fabrications, with 0 false alarms on 36 real', true),
  /** the honest bound, and the reason the threshold was NOT raised */
  bound: fig(
    'it catches the mechanical failures - no citation, citation outside the pack - but not a '
      + 'confident on-topic falsehood: "privacy was expressly held subordinate to property" '
      + 'scores 0.8451 and passes',
    true
  ),
  whyNotStricter: fig(
    'a bag-of-words cosine is a topicality check, not an entailment check; raising the '
      + 'threshold to 0.90 would reject a genuine paraphrase, so it was left at 0.20',
    true
  ),
  pinnedByTest: fig('a test pins the limitation and says in its docstring that passing it means the docs need new numbers', true),
};

/** The limitation we demonstrate live. */
export const LIMITATION = {
  /** every error runs the same way: upward, into a label that demotes */
  errorDirection: fig('all 3 errors are "mentioned" read as something stronger', true),
  treatmentAccuracy: fig('92.5%', true),
  treatmentSampleSize: fig('40', true),
  sentence: fig('"We are unable to hold that the decision stands overruled."', true),
};

// ---------------------------------------------------------------------------
// 8. Fixed, non-measured copy. Real text, not results.
// ---------------------------------------------------------------------------
export const TEAM = ['Rakshit', 'Suvrat', 'Sumedha'] as const;
export type Person = (typeof TEAM)[number];

export const COURSE = {
  title: 'NyayaTrace',
  tagline: 'treatment-aware precedent search for Indian case law',
  course: 'CSD358 · Information Retrieval',
  event: 'Mid-term Hackathon',
  track: 'Track T6 — vertical search for law',
};

/** The three surface forms that collapse to one token. */
export const STATUTE_SURFACES = [
  'Section 302 IPC',
  'S. 302 of the Indian Penal Code, 1860',
  'u/s 302',
];
export const STATUTE_TOKEN = 'ipc§302';
export const BNS_TOKEN = 'bns§103';
export const BNS_DATE = '1 July 2024';

/** Example queries. These are exactly what the screen recordings type. */
export const DEMO_FREETEXT = 'right to privacy of personal data';
export const DEMO_QUERY = 'privacy /p "fundamental right" AND bench:>=5 year:>=2015';
export const DEMO_PROX_QUERY = 'privacy /p "fundamental right"';
export const DEMO_STATUTE_TEXT = 'convicted u/s 302 IPC read with S. 34';

/**
 * The real plan for DEMO_QUERY, printed by `--explain-plan`. The BooleanPlan
 * animation draws exactly these numbers, so the insert and the terminal agree.
 */
export const DEMO_PLAN = {
  operands: [
    {label: 'PROX_P', df: 126},
    {label: 'FILTER(bench>=5)', df: 1154},
    {label: 'FILTER(year>=2015)', df: 2139},
  ],
  /** running candidate count after each step, in the df-ordered plan */
  running: [126, 66, 16, 9],
  comparedOrdered: 3375,
  comparedAsWritten: 3449,
  postingsRead: 12500,
};

/**
 * What the headline free-text query prints, re-verified after the rebuild.
 * S5 is now wired into search(), so the candidate count is tier 1's 1,174 and
 * no longer the whole collection.
 */
export const DEMO_RESULT = {
  candidates: fig('1,174', true),
  collection: fig('13,878', true),
  s4Top: [
    'Puttaswamy 2018 \u00b7 bench 5',
    'Puttaswamy 2017 \u00b7 bench 9',
    'Electoral Bonds 2024 \u00b7 bench 6',
    'Puttaswamy 2015 \u00b7 bench 3',
  ],
  b0Top: fig('the 2015 three-judge reference, with no 2017 or 2018 Puttaswamy in the ten', true),
  /** from --why on rank 1 */
  whyTf: fig('237.914396', true),
  whyNet: fig('0.900210', true),
  whyG: fig('0.251052', true),
};

/**
 * The real /p walk inside doc 239, printed by `--merge --trace`. `privaci`
 * positions on the left, the head of each `fundament right` phrase hit on the
 * right, with the sentence and paragraph id the index stores for each.
 */
export const DEMO_WALK = {
  doc: 239,
  left: [
    {pos: 536, sent: 23, para: 8},
    {pos: 7027, sent: 252, para: 56},
  ],
  right: [
    {pos: 91, sent: 4, para: 4},
    {pos: 186, sent: 7, para: 4},
    {pos: 533, sent: 23, para: 8},
    {pos: 556, sent: 23, para: 8},
    {pos: 648, sent: 28, para: 9},
    {pos: 1474, sent: 60, para: 15},
    {pos: 7024, sent: 252, para: 56},
    {pos: 7044, sent: 252, para: 56},
  ],
  pairsAccepted: 8,
  smallestGap: 2,
};
