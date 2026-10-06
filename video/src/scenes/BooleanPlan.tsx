import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';

/**
 * Act 4 - the Boolean planner, Suvrat's component.
 *
 * Three operands of an AND arrive in the order the user typed them. Their bars
 * are sized by document frequency. The bars then physically re-sort so the
 * rarest goes first, and the candidate counter shows why: the intersection
 * cannot grow, so starting small keeps every later merge small.
 */

type Operand = {term: string; df: number};

const OPERANDS: Operand[] = [
  {term: 'right', df: 9120},
  {term: 'privaci', df: 410},
  {term: 'const§21', df: 1260},
];

const sorted = [...OPERANDS].sort((a, b) => a.df - b.df);
const MAXDF = Math.max(...OPERANDS.map((o) => o.df));

/** Candidate set after each intersection, in each order. Illustrative shape. */
const NAIVE_PATH = [9120, 980, 190];
const PLANNED_PATH = [410, 160, 120];

const BAR_X = 480;
const BAR_W = 760;
const ROW_H = 112;
const TOP = 330;

export const BooleanPlan: React.FC = () => {
  const frame = useCurrentFrame();
  const barsIn = ramp(frame, 10, 60);
  const resort = ramp(frame, 120, 190);
  const pathsIn = ramp(frame, 210, 280);

  return (
    <Scene act="the pipeline · planner" heading="Process the rarest operand first.">
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 250,
          fontFamily: FONT_MONO,
          fontSize: 32,
          color: COLORS.inkDim,
          opacity: barsIn,
        }}
      >
        <span style={{color: COLORS.ink}}>right</span> AND{' '}
        <span style={{color: COLORS.ink}}>privacy</span> AND{' '}
        <span style={{color: COLORS.ink}}>const&#167;21</span>
      </div>

      {OPERANDS.map((o, i) => {
        const target = sorted.findIndex((s) => s.term === o.term);
        const y = TOP + interpolate(resort, [0, 1], [i, target]) * ROW_H;
        const w = (o.df / MAXDF) * BAR_W * barsIn;
        const isFirst = resort > 0.5 && target === 0;
        return (
          <div key={o.term} style={{position: 'absolute', left: 0, top: y, right: 0}}>
            <div
              style={{
                position: 'absolute',
                left: BAR_X - 24,
                width: 300,
                textAlign: 'right',
                transform: 'translateX(-300px)',
                fontFamily: FONT_MONO,
                fontSize: 30,
                color: isFirst ? COLORS.accent : COLORS.ink,
                opacity: barsIn,
              }}
            >
              {o.term}
            </div>
            <div
              style={{
                position: 'absolute',
                left: BAR_X,
                width: w,
                height: 54,
                borderRadius: 5,
                background: isFirst ? COLORS.accent : COLORS.coolSoft,
                opacity: isFirst ? 0.9 : 0.6,
              }}
            />
            <div
              style={{
                position: 'absolute',
                left: BAR_X + w + 20,
                top: 12,
                width: 160,
                fontFamily: FONT_MONO,
                fontSize: 27,
                color: COLORS.inkDim,
                opacity: barsIn,
                whiteSpace: 'nowrap',
              }}
            >
              df {o.df.toLocaleString('en-US')}
            </div>
            {isFirst ? (
              <div
                style={{
                  position: 'absolute',
                  left: BAR_X + w + 190,
                  top: 10,
                  fontFamily: FONT_MONO,
                  fontSize: 24,
                  color: COLORS.accent,
                  border: `1.3px solid ${COLORS.accent}`,
                  borderRadius: 5,
                  padding: '4px 14px',
                  whiteSpace: 'nowrap',
                }}
              >
                start here
              </div>
            ) : null}
          </div>
        );
      })}

      {/* candidate-set paths */}
      <svg
        width={SIZES.width}
        height={SIZES.height}
        style={{position: 'absolute', inset: 0, opacity: pathsIn}}
      >
        {[
          {path: NAIVE_PATH, color: COLORS.inkFaint, label: 'as typed'},
          {path: PLANNED_PATH, color: COLORS.accent, label: 'df-ordered'},
        ].map(({path, color, label}, k) => {
          const x0 = 560;
          const dx = 290;
          const yBase = 790;
          const scale = (v: number) => yBase - (Math.log10(v) / Math.log10(MAXDF)) * 140;
          const t = ramp(frame, 230 + k * 40, 230 + k * 40 + 60);
          const pts = path.map((v, i) => [x0 + i * dx, scale(v)] as const);
          const shown = pts.slice(0, Math.max(1, Math.ceil(pts.length * t)));
          return (
            <g key={label}>
              <polyline
                points={shown.map(([x, y]) => `${x},${y}`).join(' ')}
                fill="none"
                stroke={color}
                strokeWidth={2.5}
              />
              {shown.map(([x, y], i) => (
                <g key={i}>
                  <circle cx={x} cy={y} r={6} fill={color} />
                  <text
                    x={x}
                    y={y - 18}
                    textAnchor="middle"
                    fontFamily={FONT_MONO}
                    fontSize={22}
                    fill={color}
                  >
                    {path[i].toLocaleString('en-US')}
                  </text>
                </g>
              ))}
              <text
                x={x0 - 20}
                y={scale(path[0]) + 8}
                textAnchor="end"
                fontFamily={FONT_MONO}
                fontSize={23}
                fill={color}
              >
                {label}
              </text>
            </g>
          );
        })}
        <text
          x={560}
          y={840}
          fontFamily={FONT_MONO}
          fontSize={21}
          fill={COLORS.inkFaint}
          opacity={pathsIn}
        >
          documents still in play after each intersection
        </text>
      </svg>

      <Caption delay={300}>
        The planner also folds the parametric filters in first when they are selective, and
        every merge uses the skip pointers, so a long list is never read end to end just to
        find out it has nothing in common with a short one.
      </Caption>
    </Scene>
  );
};
