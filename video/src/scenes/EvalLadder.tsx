import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Fig, PendingStamp, ramp, Scene} from '../components/primitives';
import {EVAL, LADDER_HEIGHTS, MEASURED, SYSTEMS} from '../data';

/**
 * Act 5 - the ladder. Six systems, one code path, one flag apart.
 *
 * Bars rise in sequence while the metric table underneath fills in. Geometry
 * comes from LADDER_HEIGHTS and every printed number from SYSTEMS, so until the
 * run exists the chart is stamped shape-only and all five metrics read as
 * bracketed placeholders.
 */

const X0 = 230;
const GAP = 170;
const BAR_W = 100;
const BASE = 720;
const MAXH = 300;
const BLURB_Y = 296;
const BLURB_W = 164;

const METRICS: {key: 'p5' | 'p10' | 'r20' | 'map' | 'ndcg10'; label: string}[] = [
  {key: 'p5', label: 'P@5'},
  {key: 'p10', label: 'P@10'},
  {key: 'r20', label: 'R@20'},
  {key: 'map', label: 'MAP'},
  {key: 'ndcg10', label: 'nDCG@10'},
];

const COL_W = 190;

export const EvalLadder: React.FC = () => {
  const frame = useCurrentFrame();
  const axisIn = ramp(frame, 8, 36);
  const tableIn = ramp(frame, 220, 290);
  const noteIn = ramp(frame, 60, 110);
  const readIn = ramp(frame, 330, 380);

  const barTop = (id: string) => BASE - (LADDER_HEIGHTS[id] ?? 0.4) * MAXH;

  return (
    <Scene act="evaluation" heading="Six systems. One code path. One flag apart.">
      {MEASURED ? null : <PendingStamp />}

      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        <line
          x1={X0 - 60}
          y1={BASE}
          x2={X0 + (SYSTEMS.length - 1) * GAP + BAR_W + 50}
          y2={BASE}
          stroke={COLORS.inkFaint}
          strokeWidth={1.5}
          opacity={axisIn}
        />
        {[0.33, 0.66, 1].map((g) => (
          <line
            key={g}
            x1={X0 - 60}
            y1={BASE - g * MAXH}
            x2={X0 + (SYSTEMS.length - 1) * GAP + BAR_W + 50}
            y2={BASE - g * MAXH}
            stroke={COLORS.panelEdge}
            strokeWidth={1}
            strokeDasharray="4 10"
            opacity={axisIn * 0.8}
          />
        ))}
        <text
          x={X0 - 66}
          y={BASE - MAXH - 16}
          textAnchor="end"
          fontFamily={FONT_MONO}
          fontSize={21}
          fill={COLORS.inkFaint}
          opacity={axisIn}
        >
          nDCG@10
        </text>

        {SYSTEMS.map((s, i) => {
          const t = ramp(frame, 30 + i * 24, 30 + i * 24 + 30);
          const h = (LADDER_HEIGHTS[s.id] ?? 0.4) * MAXH * t;
          const isFull = s.id === 'S4';
          return (
            <g key={s.id}>
              {i > 0 ? (
                <path
                  d={`M ${X0 + (i - 1) * GAP + BAR_W} ${barTop(SYSTEMS[i - 1].id)} L ${
                    X0 + i * GAP
                  } ${barTop(s.id)}`}
                  stroke={COLORS.inkFaint}
                  strokeWidth={1.4}
                  strokeDasharray="5 6"
                  fill="none"
                  opacity={t * 0.75}
                />
              ) : null}
              <rect
                x={X0 + i * GAP}
                y={BASE - h}
                width={BAR_W}
                height={h}
                rx={5}
                fill={isFull ? COLORS.accent : COLORS.coolSoft}
                opacity={isFull ? 0.92 : 0.62}
              />
              <text
                x={X0 + i * GAP + BAR_W / 2}
                y={BASE + 46}
                textAnchor="middle"
                fontFamily={FONT_MONO}
                fontSize={32}
                fill={isFull ? COLORS.accent : COLORS.ink}
                opacity={t}
              >
                {s.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* one-line description of the single change each system makes */}
      {SYSTEMS.map((s, i) => {
        const t = ramp(frame, 36 + i * 24, 36 + i * 24 + 26);
        return (
          <div
            key={s.id}
            style={{
              position: 'absolute',
              left: X0 + i * GAP + BAR_W / 2 - BLURB_W / 2,
              width: BLURB_W,
              top: BLURB_Y,
              textAlign: 'center',
              fontSize: 19,
              lineHeight: 1.35,
              color: s.id === 'S4' ? COLORS.accent : COLORS.inkDim,
              opacity: t,
            }}
          >
            {s.blurb}
          </div>
        );
      })}

      {/* right column: how the judgments were made, and how to read the chart */}
      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 256,
          width: 420,
          fontSize: 25,
          lineHeight: 1.55,
          color: COLORS.inkDim,
          opacity: noteIn,
        }}
      >
        <Fig f={EVAL.nQueries} /> hand-written queries, pooled to depth{' '}
        <Fig f={EVAL.poolDepth} />, <Fig f={EVAL.nJudged} /> judgments graded{' '}
        <Fig f={EVAL.gradedScale} /> by a law-reading human who could not see which system
        retrieved what.
      </div>

      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 524,
          width: 420,
          fontSize: 25,
          lineHeight: 1.55,
          color: COLORS.inkDim,
          opacity: readIn,
          borderLeft: `2px solid ${COLORS.accentSoft}`,
          paddingLeft: 20,
        }}
      >
        Read the shape, not any single bar. Each step is one isolated change, so when a metric
        fails to move we know exactly which idea did not pay for itself.
      </div>

      {/* metric table */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 812,
          opacity: tableIn,
          fontFamily: FONT_MONO,
          fontSize: 22,
        }}
      >
        <div
          style={{
            display: 'flex',
            borderBottom: `1px solid ${COLORS.panelEdge}`,
            paddingBottom: 8,
          }}
        >
          <div style={{width: 150, color: COLORS.inkFaint}}>metric</div>
          {SYSTEMS.map((s) => (
            <div key={s.id} style={{width: COL_W, color: COLORS.inkFaint}}>
              {s.label}
            </div>
          ))}
        </div>
        {METRICS.map((m, r) => (
          <div
            key={m.key}
            style={{
              display: 'flex',
              paddingTop: 9,
              opacity: ramp(frame, 230 + r * 10, 230 + r * 10 + 16),
            }}
          >
            <div style={{width: 150, color: COLORS.inkDim}}>{m.label}</div>
            {SYSTEMS.map((s) => (
              <div key={s.id} style={{width: COL_W}}>
                <Fig f={s[m.key]} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </Scene>
  );
};
