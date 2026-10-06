import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {CONFIG} from '../data';

/**
 * Act 4 - the novelty, Suvrat's component (over edges Rakshit labelled).
 *
 * A small citation graph. First the edges are plain grey arrows and A's bar is
 * driven by in-degree. Then a cue-phrase classifier colours each edge with its
 * treatment, the edge weights change, power iteration runs visibly (mass flows
 * along the edges, the bars move, the iteration counter ticks) and finally the
 * status penalty multiplies the winner down.
 */

type NodeT = {id: string; x: number; y: number; r: number};
type EdgeT = {from: string; to: string; treat: 'followed' | 'distinguished' | 'mentioned' | 'overruled'};

const NODES: NodeT[] = [
  {id: 'A', x: 1000, y: 560, r: 52}, // the heavily cited case
  {id: 'B', x: 760, y: 372, r: 30},
  {id: 'C', x: 726, y: 596, r: 30},
  {id: 'D', x: 790, y: 790, r: 30},
  {id: 'E', x: 1236, y: 358, r: 36}, // the court that overrules it
  {id: 'F', x: 1250, y: 752, r: 30},
  {id: 'G', x: 642, y: 470, r: 24},
  {id: 'H', x: 650, y: 712, r: 24},
];

const EDGES: EdgeT[] = [
  {from: 'B', to: 'A', treat: 'followed'},
  {from: 'C', to: 'A', treat: 'followed'},
  {from: 'D', to: 'A', treat: 'mentioned'},
  {from: 'F', to: 'A', treat: 'distinguished'},
  {from: 'E', to: 'A', treat: 'overruled'},
  {from: 'G', to: 'B', treat: 'followed'},
  {from: 'H', to: 'C', treat: 'followed'},
  {from: 'G', to: 'C', treat: 'mentioned'},
  {from: 'D', to: 'F', treat: 'followed'},
];

const TREAT_COLOR = {
  followed: COLORS.green,
  distinguished: COLORS.amber,
  mentioned: COLORS.inkFaint,
  overruled: COLORS.alarm,
} as const;

const CUES: Record<EdgeT['treat'], string> = {
  followed: '"we respectfully follow"',
  distinguished: '"is distinguishable on facts"',
  mentioned: '"see also"',
  overruled: '"stands overruled"',
};

const byId = (id: string) => NODES.find((n) => n.id === id)!;

const BAR_X = 1470;
const BAR_W = 42;
const BAR_GAP = 54;
const BAR_BASE = 800;
const BAR_H = 390;

/** Geometry only - the printed constants all come from CONFIG. */
const PR_BEFORE: Record<string, number> = {A: 0.95, B: 0.35, C: 0.4, D: 0.2, E: 0.5, F: 0.3, G: 0.12, H: 0.12};
const PR_AFTER: Record<string, number> = {A: 0.62, B: 0.38, C: 0.44, D: 0.2, E: 0.6, F: 0.26, G: 0.12, H: 0.12};

export const CitationGraph: React.FC = () => {
  const frame = useCurrentFrame();

  const graphIn = ramp(frame, 8, 70);
  const labelsIn = ramp(frame, 110, 190); // the cue classifier colours the edges
  const iterIn = ramp(frame, 240, 410); // power iteration
  const penaltyIn = ramp(frame, 450, 520);

  const iteration = Math.round(iterIn * Number(CONFIG.pagerankIterations.value));

  const barValue = (id: string) => {
    const mid = PR_BEFORE[id] + (PR_AFTER[id] - PR_BEFORE[id]) * iterIn;
    if (id !== 'A') return mid;
    return mid * (1 - (1 - CONFIG.treatmentPenalty.overruled) * penaltyIn);
  };

  return (
    <Scene act="the pipeline · authority" heading="An edge is a verdict, not a vote.">
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {EDGES.map((e, i) => {
          const a = byId(e.from);
          const b = byId(e.to);
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len = Math.hypot(dx, dy);
          const ux = dx / len;
          const uy = dy / len;
          const x1 = a.x + ux * (a.r + 4);
          const y1 = a.y + uy * (a.r + 4);
          const x2 = b.x - ux * (b.r + 12);
          const y2 = b.y - uy * (b.r + 12);
          const t = ramp(frame, 12 + i * 6, 12 + i * 6 + 22);
          const col = `rgb(${lerpCol(COLORS.inkFaint, TREAT_COLOR[e.treat], labelsIn)})`;
          const weight = CONFIG.treatmentWeights[e.treat];
          const sw = 1.6 + labelsIn * weight * 3.4;
          const pulseT = (frame / 44 + i * 0.37) % 1;
          const showPulse = iterIn > 0.02 && iterIn < 0.999 && weight > 0;
          return (
            <g key={i} opacity={t * graphIn}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={sw} opacity={0.85} />
              <polygon
                points={`${x2},${y2} ${x2 - ux * 15 - uy * 6},${y2 - uy * 15 + ux * 6} ${
                  x2 - ux * 15 + uy * 6
                },${y2 - uy * 15 - ux * 6}`}
                fill={col}
              />
              {showPulse ? (
                <circle
                  cx={x2 + (x1 - x2) * pulseT}
                  cy={y2 + (y1 - y2) * pulseT}
                  r={4.5}
                  fill={col}
                  opacity={0.9 * (1 - Math.abs(pulseT - 0.5) * 1.2)}
                />
              ) : null}
            </g>
          );
        })}

        {/* call out the one edge that changes everything */}
        <g opacity={labelsIn}>
          <text x={byId('E').x + 54} y={byId('E').y - 6} fontFamily={FONT_MONO} fontSize={25} fill={COLORS.alarm}>
            overruled
          </text>
          <text x={byId('E').x + 54} y={byId('E').y + 24} fontFamily={FONT_MONO} fontSize={19} fill={COLORS.inkDim}>
            weight {CONFIG.treatmentWeights.overruled.toFixed(1)}
          </text>
        </g>

        {NODES.map((n, i) => {
          const t = ramp(frame, 8 + i * 7, 8 + i * 7 + 20);
          const isA = n.id === 'A';
          const dead = isA && penaltyIn > 0.4;
          return (
            <g key={n.id} opacity={t}>
              <circle
                cx={n.x}
                cy={n.y}
                r={n.r}
                fill={dead ? 'rgba(224,96,79,0.16)' : 'rgba(111,179,224,0.1)'}
                stroke={dead ? COLORS.alarm : isA ? COLORS.cool : COLORS.coolSoft}
                strokeWidth={isA ? 2.6 : 1.6}
              />
              <text
                x={n.x}
                y={n.y + 9}
                textAnchor="middle"
                fontFamily={FONT_MONO}
                fontSize={isA ? 29 : 22}
                fill={dead ? COLORS.alarm : COLORS.ink}
              >
                {n.id}
              </text>
            </g>
          );
        })}

        {/* authority bars */}
        <text x={BAR_X + 3 * BAR_GAP} y={BAR_BASE - BAR_H - 44} fontFamily={FONT_MONO} fontSize={21} letterSpacing={2.5} fill={COLORS.inkFaint} opacity={graphIn}>
          g(d)
        </text>
        {NODES.map((n, i) => {
          const h = barValue(n.id) * BAR_H;
          const isA = n.id === 'A';
          const dead = isA && penaltyIn > 0.4;
          return (
            <g key={n.id} opacity={graphIn}>
              <rect
                x={BAR_X + i * BAR_GAP}
                y={BAR_BASE - h}
                width={BAR_W}
                height={h}
                rx={4}
                fill={dead ? COLORS.alarm : isA ? COLORS.violet : COLORS.coolSoft}
                opacity={0.85}
              />
              <text
                x={BAR_X + i * BAR_GAP + BAR_W / 2}
                y={BAR_BASE + 28}
                textAnchor="middle"
                fontFamily={FONT_MONO}
                fontSize={20}
                fill={dead ? COLORS.alarm : COLORS.inkDim}
              >
                {n.id}
              </text>
            </g>
          );
        })}
        <line x1={BAR_X - 14} y1={BAR_BASE} x2={BAR_X + NODES.length * BAR_GAP} y2={BAR_BASE} stroke={COLORS.inkFaint} strokeWidth={1.2} opacity={graphIn} />
      </svg>

      {/* left column: the classifier, the iteration, the penalty */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 258,
          width: 400,
          opacity: labelsIn,
          fontFamily: FONT_MONO,
          fontSize: 22,
        }}
      >
        <div style={{color: COLORS.inkFaint, letterSpacing: 2.5, marginBottom: 16}}>
          CUE PHRASE &#8594; EDGE WEIGHT
        </div>
        {(Object.keys(TREAT_COLOR) as EdgeT['treat'][]).map((k) => (
          <div key={k} style={{marginBottom: 13}}>
            <div style={{display: 'flex', gap: 12}}>
              <span style={{color: TREAT_COLOR[k], width: 190}}>{k}</span>
              <span style={{color: COLORS.ink}}>{CONFIG.treatmentWeights[k].toFixed(1)}</span>
            </div>
            <div style={{color: COLORS.inkFaint, fontSize: 19}}>{CUES[k]}</div>
          </div>
        ))}
      </div>

      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 606,
          width: 470,
          opacity: ramp(frame, 240, 290),
          fontFamily: FONT_MONO,
          fontSize: 22,
          color: COLORS.inkDim,
          lineHeight: 1.6,
        }}
      >
        <div style={{color: COLORS.violet}}>
          power iteration {iteration} / <Fig f={CONFIG.pagerankIterations} />
        </div>
        <div>
          &#960; &#8592; &#955;&#183;P&#7488;&#960; + (1&#8722;&#955;)/N, &#955; ={' '}
          <Fig f={CONFIG.pagerankLambda} />
        </div>
        <div style={{color: COLORS.inkFaint, fontSize: 18, lineHeight: 1.4, marginTop: 8}}>
          P is row-normalised over <em>treatment</em> weights &#8212; an overruling edge
          transfers nothing.
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 752,
          width: 580,
          opacity: penaltyIn,
          fontFamily: FONT_MONO,
          fontSize: 22,
          lineHeight: 1.6,
        }}
      >
        <div style={{color: COLORS.alarm}}>then the status penalty, which is not PageRank:</div>
        <div style={{color: COLORS.ink}}>
          g(d) = <Fig f={CONFIG.gPagerankWeight} />&#183;pr&#770; +{' '}
          <Fig f={CONFIG.gBenchWeight} />&#183;bench&#770; &nbsp;&#215;&nbsp; penalty
        </div>
        <div style={{color: COLORS.inkDim, fontSize: 20}}>
          overruled &#215; {CONFIG.treatmentPenalty.overruled} &#183; distinguished &#215;{' '}
          {CONFIG.treatmentPenalty.distinguished} &#183; ok &#215;{' '}
          {CONFIG.treatmentPenalty.ok}
        </div>
      </div>

      <Caption delay={530}>
        A is still the most cited node here. After one red edge it is no longer the most
        authoritative &#8212; and the result card names who overruled it, and where.
      </Caption>
    </Scene>
  );
};

/** Linear interpolation between two hex colours, returned as "r,g,b". */
function lerpCol(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const A = p(a);
  const B = p(b);
  return A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',');
}
