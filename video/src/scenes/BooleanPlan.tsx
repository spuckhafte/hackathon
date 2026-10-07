import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';
import {DEMO_PLAN, DEMO_QUERY} from '../data';

/**
 * INSERT over `--explain-plan` / `python -m nyaya.query.planner --compare`.
 *
 * The three operands of the AND arrive in the order the user typed them, each
 * bar sized by the document frequency the terminal has just printed. The bars
 * then physically re-sort so the rarest goes first, and the running candidate
 * count walks down the real numbers from the plan: 126 -> 66 -> 15 -> 8.
 *
 * Nothing here is illustrative. Every df, every running count and the two
 * comparison totals come from DEMO_PLAN in src/data.ts, which is the measured
 * output of that command on DEMO_QUERY.
 */

type Operand = {term: string; df: number};

const OPERANDS: Operand[] = DEMO_PLAN.operands.map((o) => ({term: o.label, df: o.df}));

const sorted = [...OPERANDS].sort((a, b) => a.df - b.df);
const MAXDF = Math.max(...OPERANDS.map((o) => o.df));

/** The real running candidate count after each step of the df-ordered plan. */
const PLANNED_PATH = DEMO_PLAN.running;

const BAR_X = 480;
const BAR_W = 760;
const ROW_H = 96;
const TOP = 300;

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
        <span style={{color: COLORS.ink}}>{DEMO_QUERY}</span>
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
        {[{path: PLANNED_PATH, color: COLORS.accent, label: 'df-ordered'}].map(
          ({path, color, label}, k) => {
          const x0 = 600;
          const dx = 215;
          const yBase = 782;
          // normalised to the path's own range, so the descent reads at a glance
          const lo = Math.log10(Math.min(...path));
          const hi = Math.log10(Math.max(...path));
          const scale = (v: number) => yBase - ((Math.log10(v) - lo) / (hi - lo)) * 150;
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
        }
        )}
        <text
          x={600}
          y={814}
          fontFamily={FONT_MONO}
          fontSize={21}
          fill={COLORS.inkFaint}
          opacity={pathsIn}
        >
          documents still in play after each step — an intersection can only shrink
        </text>
        <text
          x={600}
          y={850}
          fontFamily={FONT_MONO}
          fontSize={21}
          fill={COLORS.inkFaint}
          opacity={ramp(frame, 330, 380)}
        >
          entries compared: {DEMO_PLAN.comparedOrdered.toLocaleString('en-US')} df-ordered vs{' '}
          {DEMO_PLAN.comparedAsWritten.toLocaleString('en-US')} as written, on this query
        </text>
      </svg>

      <Caption delay={300}>
        Across the query set that ordering does 2.10&#215; less comparison work, and 327&#215;
        less in the best case. Every merge uses the skip pointers too, so a long list is never
        read end to end just to find out it has nothing in common with a short one.
      </Caption>
    </Scene>
  );
};
