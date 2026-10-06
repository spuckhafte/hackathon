/**
 * NyayaTrace video — visual language.
 *
 * Calm, dark, 3Blue1Brown-flavoured. One deep background, one warm accent for
 * "our system", one cool accent for structure/maths, one alarm colour used
 * ONLY for `overruled`. Nothing else gets a colour.
 */

export const COLORS = {
  bg: '#0d1117',
  bgDeep: '#080b10',
  panel: '#151b24',
  panelEdge: '#242e3c',

  ink: '#e8edf4', // primary text
  inkDim: '#9aa7b8', // secondary text
  inkFaint: '#5b6676', // tertiary / grid lines

  accent: '#f0b26b', // NyayaTrace warm gold — the system itself
  accentSoft: '#8a6433',
  cool: '#6fb3e0', // structure, indexes, maths
  coolSoft: '#2f5872',
  green: '#7bc98c', // followed / good / verified
  violet: '#b79af0', // authority, PageRank
  alarm: '#e0604f', // overruled — used nowhere else
  amber: '#d8a23a', // distinguished
} as const;

export const FONTS = {
  // Loaded from @remotion/google-fonts in src/fonts.ts
  sans: 'Inter, system-ui, sans-serif',
  mono: '"JetBrains Mono", "Cascadia Mono", Consolas, monospace',
  serif: '"Source Serif 4", Georgia, serif',
  math: '"Latin Modern Math", "Cambria Math", "Source Serif 4", Georgia, serif',
} as const;

export const SIZES = {
  width: 1920,
  height: 1080,
  fps: 30,
  /** Generous whitespace: nothing is drawn outside this box. */
  margin: 150,
  h1: 96,
  h2: 62,
  h3: 44,
  body: 34,
  caption: 27,
  code: 28,
} as const;

/** Standard spring for things that enter. */
export const SPRING = {damping: 200, mass: 0.9, stiffness: 110} as const;
/** Snappier spring for things that move within a scene. */
export const SPRING_FAST = {damping: 200, mass: 0.6, stiffness: 160} as const;
