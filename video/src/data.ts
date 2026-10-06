/**
 * ============================================================================
 *  PLACEHOLDER - replaced with measured values before render
 * ============================================================================
 *
 *  EVERY number, score, timing and count that the video SPEAKS OR SHOWS lives
 *  in this file and nowhere else. No component may hard-code a figure.
 *
 *  Anything whose `measured` flag is false is a PLACEHOLDER: the renderer
 *  draws it with a dashed "pending measurement" underline so that an
 *  un-updated figure is impossible to miss in a review render.
 *
 *  Phase 2: the evaluation harness (Rakshit) and the efficiency harness
 *  (Suvrat) produce `runs/*.json`; paste the values in here, flip
 *  `MEASURED` to true, and re-render. Do not invent numbers.
 * ============================================================================
 */

/** Master switch. Flip to `true` only when every value below is real. */
export const MEASURED = false;

/** A figure that is still a placeholder renders with a dashed underline. */
export type Figure = {value: string; measured?: boolean};
export const fig = (value: string, measured = false): Figure => ({value, measured});

// ---------------------------------------------------------------------------
// 1. Corpus - known from the public release + our frozen corpus policy
//    (docs/CONTRACTS.md), so these are already true.
// ---------------------------------------------------------------------------
export const CORPUS = {
  /** Every English judgment in the AWS Open Data release, de-duplicated. */
  universe: fig('42,371', true),
  /** Stratified sample actually indexed, <= PER_YEAR_CAP per year. */
  indexed: fig('~14,000', true),
  perYearCap: fig('200', true),
  years: fig('1950 - 2025', true),
  seed: fig('358', true),
  licence: fig('AWS Open Data | CC-BY-4.0', true),

  // --- filled from data/processed/extract_log.csv after the build ---
  indexedExact: fig('[DOCS]'),
  paragraphs: fig('[PARAS]'),
  tokens: fig('[TOKENS]'),
  vocabulary: fig('[VOCAB]'),
  postingsMB: fig('[POSTINGS_MB]'),
  buildMinutes: fig('[BUILD_MIN]'),
  citationEdges: fig('[EDGES]'),
  resolvedPct: fig('[RESOLVED_PCT]'),
  overruledCases: fig('[N_OVERRULED]'),
  /** share of all indexed tokens that fall in the `body` zone */
  bodyTokenShare: fig('[BODY_SHARE_PCT]'),
};

// ---------------------------------------------------------------------------
// 2. Architecture constants - mirrored from nyaya/config.py. True today;
//    re-check if config.py changes (ZONE_WEIGHTS, ALPHA, BETA, BM25F_K1...).
// ---------------------------------------------------------------------------
export const CONFIG = {
  zones: ['title', 'parties', 'bench', 'statutes', 'order', 'body'] as const,
  zoneWeights: {title: 3.0, parties: 0.5, bench: 0.0, statutes: 2.0, order: 1.5, body: 1.0},
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
};

// ---------------------------------------------------------------------------
// 3. Evaluation - ALL PLACEHOLDERS. From runs/metrics.json.
// ---------------------------------------------------------------------------
export type SystemRow = {
  id: string;
  label: string;
  /** one-line description spoken over the chart */
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
    p5: fig('[B0_P5]'), p10: fig('[B0_P10]'), r20: fig('[B0_R20]'),
    map: fig('[B0_MAP]'), ndcg10: fig('[B0_NDCG]'),
  },
  {
    id: 'S1',
    label: 'S1',
    blurb: 'plus statute canonicalisation and a legal stop list',
    p5: fig('[S1_P5]'), p10: fig('[S1_P10]'), r20: fig('[S1_R20]'),
    map: fig('[S1_MAP]'), ndcg10: fig('[S1_NDCG]'),
  },
  {
    id: 'S2',
    label: 'S2',
    blurb: 'plus weighted zone scoring',
    p5: fig('[S2_P5]'), p10: fig('[S2_P10]'), r20: fig('[S2_R20]'),
    map: fig('[S2_MAP]'), ndcg10: fig('[S2_NDCG]'),
  },
  {
    id: 'S2p',
    label: 'S2′',
    blurb: 'BM25F instead of cosine',
    p5: fig('[S2P_P5]'), p10: fig('[S2P_P10]'), r20: fig('[S2P_R20]'),
    map: fig('[S2P_MAP]'), ndcg10: fig('[S2P_NDCG]'),
  },
  {
    id: 'S3',
    label: 'S3',
    blurb: 'plus authority from plain PageRank',
    p5: fig('[S3_P5]'), p10: fig('[S3_P10]'), r20: fig('[S3_R20]'),
    map: fig('[S3_MAP]'), ndcg10: fig('[S3_NDCG]'),
  },
  {
    id: 'S4',
    label: 'S4',
    blurb: 'plus treatment-aware g(d) and proximity - the full system',
    p5: fig('[S4_P5]'), p10: fig('[S4_P10]'), r20: fig('[S4_R20]'),
    map: fig('[S4_MAP]'), ndcg10: fig('[S4_NDCG]'),
  },
];

/**
 * Normalised bar heights for the nDCG@10 ladder, 0..1. These drive the CHART
 * GEOMETRY only; every printed label comes from SYSTEMS above. Until the run
 * exists these are a deliberately uniform ramp, and the chart is stamped
 * "shape only - awaiting measurement" so nobody mistakes it for a result.
 */
export const LADDER_HEIGHTS: Record<string, number> = {
  B0: 0.34, S1: 0.42, S2: 0.5, S2p: 0.58, S3: 0.66, S4: 0.76,
};

export const EVAL = {
  nQueries: fig('[N_QUERIES]'),
  nJudged: fig('[N_JUDGED]'),
  poolDepth: fig('[POOL_DEPTH]'),
  gradedScale: fig('0 - 3', true),
  /** headline deltas, S4 over B0 */
  deltaNdcg: fig('[DELTA_NDCG]'),
  deltaMap: fig('[DELTA_MAP]'),
  deltaP10: fig('[DELTA_P10]'),
};

/** Automatic track: hide a judgment's own citations, try to recover them. */
export const CITATION_RECOVERY = {
  nQueries: fig('[CR_N]'),
  recallAt20: fig('[CR_R20]'),
  mrr: fig('[CR_MRR]'),
  ceilingNote: fig('capped by the 200-per-year sample', true),
};

/** Operator correctness against a brute-force linear scan of the raw text. */
export const OPERATORS = {
  nCases: fig('[OP_N]'),
  agreement: fig('[OP_AGREE]'),
  families: ['phrase', '/n', '/s', '/p', 'AND OR NOT', 'root!', 'filters', 'zone:(...)'] as const,
};

/** Efficiency: tiered index plus champion lists plus df-ordered planning. */
export const SPEED = {
  fullScanMs: fig('[MS_FULL]'),
  tieredMs: fig('[MS_TIER]'),
  speedup: fig('[SPEEDUP]x'),
  recallLoss: fig('[TIER_RECALL_LOSS]'),
  skipMergeGain: fig('[SKIP_GAIN]'),
  medianQueryMs: fig('[MS_MEDIAN]'),
};

/** Precedent Brief verifier. */
export const BRIEF = {
  sentences: fig('[BR_SENT]'),
  citedPct: fig('[BR_CITED_PCT]'),
  flaggedPct: fig('[BR_FLAGGED_PCT]'),
  cosineThreshold: fig('0.20', true),
  topDocs: fig('5', true),
  maxParas: fig('8', true),
};

/** The limitation we demonstrate live. */
export const LIMITATION = {
  negationErrorRate: fig('[NEG_ERR]'),
  treatmentAccuracy: fig('[TREAT_ACC]'),
  treatmentSampleSize: fig('[TREAT_N]'),
};

// ---------------------------------------------------------------------------
// 4. Fixed, non-measured copy. Real text, not results - safe to show now.
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

/** Example queries. Must match what the demo recording actually types. */
export const DEMO_FREETEXT = 'right to privacy of personal data';
export const DEMO_QUERY = 'privacy /p "fundamental right" AND bench:>=5 year:>=2015';
export const DEMO_STATUTE_QUERY = 'act:IPC "common intention" /s murder year:>=2000';
