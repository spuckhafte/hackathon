import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {CORPUS} from '../data';

/**
 * Act 4, beat 0 - the map of the whole system, drawn once so that every later
 * scene can be located on it. Colour = owner, so the camera segments that
 * follow slot straight into the diagram.
 */

type Box = {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  owner: 'R' | 'V' | 'M';
};

const OWNER_COLOR = {R: COLORS.accent, V: COLORS.violet, M: COLORS.green} as const;
const OWNER_NAME = {R: 'Rakshit', V: 'Suvrat', M: 'Sumedha'} as const;

const ROW1 = 312;
const ROW2 = 470;
const ROW3 = 640;
const ROW4 = 800;
const H = 76;

const BOXES: Box[] = [
  {id: 'pdf', label: 'PDFs + metadata', sub: 'AWS Open Data', x: 150, y: ROW1, w: 250, owner: 'R'},
  {id: 'zone', label: 'ingest → zones', sub: 'title…body, paragraphs', x: 430, y: ROW1, w: 260, owner: 'R'},
  {id: 'an', label: 'analyze()', sub: 'canonicalise · stop · stem', x: 720, y: ROW1, w: 280, owner: 'M'},
  {id: 'spimi', label: 'SPIMI blocks', sub: 'memmapped int32', x: 1030, y: ROW1, w: 250, owner: 'R'},
  {id: 'cite', label: 'citation extract', sub: 'resolve → treatment', x: 1310, y: ROW1, w: 270, owner: 'V' },

  {id: 's3', label: 'S3 positional index', sub: 'zoned · skips', x: 150, y: ROW2, w: 300, owner: 'R'},
  {id: 's2', label: 'S2 parametric', sub: 'year · bench · judge · act', x: 480, y: ROW2, w: 300, owner: 'V'},
  {id: 's1', label: 'S1 paragraphs', sub: 'evidence units', x: 810, y: ROW2, w: 250, owner: 'M'},
  {id: 's4', label: 'S4 authority', sub: 'weighted PageRank · g(d)', x: 1090, y: ROW2, w: 320, owner: 'V'},
  {id: 's5', label: 'S5 tiers', sub: 'champion lists', x: 1440, y: ROW2, w: 220, owner: 'V'},

  {id: 'q', label: 'query parser', sub: 'AST · filters · root!', x: 150, y: ROW3, w: 280, owner: 'M'},
  {id: 'plan', label: 'df-ordered planner', sub: 'rarest first · skip merges', x: 460, y: ROW3, w: 330, owner: 'V'},
  {id: 'prox', label: 'proximity', sub: '/n · /s · /p', x: 820, y: ROW3, w: 220, owner: 'R'},
  {id: 'score', label: 'BM25F + zones', sub: 'tf-idf baseline behind it', x: 1070, y: ROW3, w: 300, owner: 'M'},
  {id: 'net', label: 'net score + heap', sub: 'α · text + g(d) + β · prox', x: 1400, y: ROW3, w: 300, owner: 'V'},

  {id: 'why', label: 'why this result', sub: 'score decomposition', x: 560, y: ROW4, w: 290, owner: 'M'},
  {id: 'snip', label: 'snippets + badge', sub: 'treatment evidence', x: 880, y: ROW4, w: 290, owner: 'R'},
  {id: 'brief', label: 'Precedent Brief', sub: 'vectorless RAG + verifier', x: 1200, y: ROW4, w: 320, owner: 'M'},
];

const EDGES: [string, string][] = [
  ['pdf', 'zone'], ['zone', 'an'], ['an', 'spimi'], ['zone', 'cite'],
  ['spimi', 's3'], ['an', 's1'], ['zone', 's2'], ['cite', 's4'], ['s3', 's5'],
  ['q', 'plan'], ['plan', 'prox'], ['prox', 'score'], ['score', 'net'],
  ['s3', 'plan'], ['s2', 'plan'], ['s4', 'net'], ['s5', 'plan'],
  ['net', 'why'], ['net', 'snip'], ['net', 'brief'], ['s1', 'brief'],
];

const byId = (id: string) => BOXES.find((b) => b.id === id)!;

export const PipelineMap: React.FC = () => {
  const frame = useCurrentFrame();
  const n = BOXES.length;

  return (
    <Scene act="the pipeline" heading="Three people, one index, one score.">
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {EDGES.map(([a, b], i) => {
          const A = byId(a);
          const B = byId(b);
          const x1 = A.x + A.w / 2;
          const y1 = A.y + H;
          const x2 = B.x + B.w / 2;
          const y2 = B.y;
          const sameRow = A.y === B.y;
          const sx = sameRow ? A.x + A.w : x1;
          const sy = sameRow ? A.y + H / 2 : y1;
          const tx = sameRow ? B.x : x2;
          const ty = sameRow ? B.y + H / 2 : y2;
          const t = ramp(frame, 26 + i * 3, 26 + i * 3 + 16);
          const d = sameRow
            ? `M ${sx} ${sy} L ${tx} ${ty}`
            : `M ${sx} ${sy} C ${sx} ${sy + 42}, ${tx} ${ty - 42}, ${tx} ${ty}`;
          return (
            <path
              key={i}
              d={d}
              stroke={COLORS.panelEdge}
              strokeWidth={1.6}
              fill="none"
              opacity={t}
              strokeDasharray={1000}
              strokeDashoffset={1000 * (1 - t)}
            />
          );
        })}

        {BOXES.map((b, i) => {
          const t = ramp(frame, 8 + i * 5, 8 + i * 5 + 18);
          const c = OWNER_COLOR[b.owner];
          return (
            <g key={b.id} opacity={t} transform={`translate(0, ${(1 - t) * 10})`}>
              <rect
                x={b.x}
                y={b.y}
                width={b.w}
                height={H}
                rx={8}
                fill={COLORS.panel}
                stroke={c}
                strokeWidth={1.4}
                opacity={0.95}
              />
              <rect x={b.x} y={b.y} width={4} height={H} rx={2} fill={c} />
              <text
                x={b.x + 20}
                y={b.y + 32}
                fontFamily={FONT_MONO}
                fontSize={24}
                fill={COLORS.ink}
              >
                {b.label}
              </text>
              {b.sub ? (
                <text x={b.x + 20} y={b.y + 58} fontSize={19} fill={COLORS.inkDim}>
                  {b.sub}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>

      {/* owner legend */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 232,
          display: 'flex',
          gap: 40,
          opacity: ramp(frame, 8 + n * 5, 8 + n * 5 + 20),
          fontFamily: FONT_MONO,
          fontSize: 23,
        }}
      >
        {(['R', 'V', 'M'] as const).map((k) => (
          <div key={k} style={{display: 'flex', alignItems: 'center', gap: 10}}>
            <div style={{width: 16, height: 16, borderRadius: 3, background: OWNER_COLOR[k]}} />
            <span style={{color: COLORS.inkDim}}>{OWNER_NAME[k]}</span>
          </div>
        ))}
        <div style={{color: COLORS.inkFaint, marginLeft: 20}}>
          <Fig f={CORPUS.indexed} /> judgments indexed &#183;{' '}
          <Fig f={CORPUS.universe} /> in the citation graph
        </div>
      </div>

      <Caption delay={210}>
        Top row builds; middle row is what sits on disk; third row is what happens when
        you type. Everything below the fold is explanation, not retrieval.
      </Caption>
    </Scene>
  );
};
