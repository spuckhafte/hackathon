import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';
import {AUTHORITY_CASES} from '../data';

/**
 * INSERT over `python -m nyaya.authority.pagerank compare`.
 *
 * The row the terminal is showing, animated. Pune Municipal Corporation (2014)
 * is cited 73 times, so its authority bar fills. Then one red `overruled` edge
 * arrives - Indore Development Authority, 2020 - and g(d) drains from 0.377 to
 * 0.087, which is the drop the compare table prints. Popularity and reliability
 * come apart, and no amount of text relevance can tell you which you are
 * looking at.
 *
 * Every number drawn here comes from AUTHORITY_CASES in src/data.ts, which is
 * the measured output of that command.
 */
const CASE = AUTHORITY_CASES[0];

const CX = 960;
const CY = 590;
const R = 76;
/** Arcs drawn, not a count: the real in-degree is CASE.cited and is printed. */
const N_IN = 11;

const BAR_X = 1500;
const BAR_W = 72;
const BAR_H = 400;
const BAR_TOP = 390;

export const OverruledTrap: React.FC = () => {
  const frame = useCurrentFrame();
  const nodeIn = ramp(frame, 8, 34);
  const barFill = ramp(frame, 50, 160);
  const killerIn = ramp(frame, 186, 232);
  const drain = ramp(frame, 232, 304);
  const stamp = ramp(frame, 256, 300);

  // citing nodes sit on the left half only, so the authority bar stays clear
  const citers = new Array(N_IN).fill(0).map((_, i) => {
    const a = Math.PI * 0.42 + (i / (N_IN - 1)) * Math.PI * 1.16;
    return {x: CX + Math.cos(a) * 330, y: CY + Math.sin(a) * 275};
  });

  /** 0.087 / 0.377 = 0.231 survives, so 76.9% of the bar drains away. */
  const DRAIN_FRACTION = 1 - 0.087 / 0.377;
  const filled = BAR_H * (0.86 * barFill) * (1 - DRAIN_FRACTION * drain);
  const overCount = killerIn > 0.5 ? 1 : 0;

  return (
    <Scene act="the problem" heading="Most cited is not the same as safe to cite.">
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {citers.map((c, i) => {
          const t = ramp(frame, 30 + i * 9, 30 + i * 9 + 20);
          const dx = CX - c.x;
          const dy = CY - c.y;
          const len = Math.hypot(dx, dy);
          const ex = c.x + (dx / len) * (len - R - 10) * t;
          const ey = c.y + (dy / len) * (len - R - 10) * t;
          return (
            <g key={i}>
              <line x1={c.x} y1={c.y} x2={ex} y2={ey} stroke={COLORS.green} strokeWidth={1.6} opacity={0.5 * t} />
              <circle cx={c.x} cy={c.y} r={7} fill={COLORS.green} opacity={0.55 * t} />
            </g>
          );
        })}

        {/* the overruling edge, arriving from above right */}
        <g opacity={killerIn}>
          <line
            x1={CX + 300}
            y1={CY - 320}
            x2={CX + 300 - (300 - R * 0.72) * killerIn}
            y2={CY - 320 + (320 - R * 0.72) * killerIn}
            stroke={COLORS.alarm}
            strokeWidth={4}
          />
          <circle cx={CX + 300} cy={CY - 320} r={11} fill={COLORS.alarm} />
          <text x={CX + 322} y={CY - 326} fontFamily={FONT_MONO} fontSize={27} fill={COLORS.alarm}>
            overruled
          </text>
          <text x={CX + 322} y={CY - 294} fontSize={21} fill={COLORS.inkDim}>
            {CASE.by}
          </text>
        </g>

        {/* the node */}
        <g opacity={nodeIn}>
          <circle
            cx={CX}
            cy={CY}
            r={R}
            fill={drain > 0.1 ? `rgba(224,96,79,${0.14 + 0.1 * drain})` : 'rgba(123,201,140,0.14)'}
            stroke={drain > 0.1 ? COLORS.alarm : COLORS.green}
            strokeWidth={2.5}
          />
          <text x={CX} y={CY + 10} textAnchor="middle" fontFamily={FONT_MONO} fontSize={30} fill={COLORS.ink}>
            d
          </text>
          <text x={CX} y={CY + R + 44} textAnchor="middle" fontSize={30} fill={COLORS.ink}>
            {CASE.name}
          </text>
          <text
            x={CX}
            y={CY + R + 78}
            textAnchor="middle"
            fontFamily={FONT_MONO}
            fontSize={24}
            fill={COLORS.inkFaint}
          >
            {CASE.year}
          </text>
        </g>

        {/* authority bar */}
        <g>
          <rect x={BAR_X} y={BAR_TOP} width={BAR_W} height={BAR_H} rx={6} fill="none" stroke={COLORS.panelEdge} strokeWidth={1.5} />
          <rect
            x={BAR_X}
            y={BAR_TOP + (BAR_H - filled)}
            width={BAR_W}
            height={filled}
            rx={6}
            fill={drain > 0.1 ? COLORS.alarm : COLORS.green}
            opacity={0.75}
          />
          <text x={BAR_X + BAR_W / 2} y={BAR_TOP + BAR_H + 42} textAnchor="middle" fontFamily={FONT_MONO} fontSize={26} fill={COLORS.inkDim}>
            g(d)
          </text>
          <text
            x={BAR_X + BAR_W / 2}
            y={BAR_TOP + BAR_H + 80}
            textAnchor="middle"
            fontFamily={FONT_MONO}
            fontSize={27}
            fill={drain > 0.1 ? COLORS.alarm : COLORS.green}
          >
            {drain > 0.05
              ? `${CASE.gBefore.value} → ${CASE.gAfter.value}`
              : CASE.gBefore.value}
          </text>
          <text x={BAR_X + BAR_W / 2} y={BAR_TOP - 22} textAnchor="middle" fontFamily={FONT_MONO} fontSize={21} letterSpacing={2} fill={COLORS.inkFaint}>
            AUTHORITY
          </text>
        </g>
      </svg>

      {/* left column */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 268,
          width: 470,
          fontSize: 29,
          lineHeight: 1.5,
          color: COLORS.inkDim,
          opacity: ramp(frame, 52, 100),
        }}
      >
        Citation count is a vote, and it never gets retracted. Every case that cited{' '}
        <span style={{fontFamily: FONT_MONO, color: COLORS.ink}}>d</span> before it fell still
        points at it. Plain PageRank has no way to subtract them.
      </div>

      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 528,
          fontFamily: FONT_MONO,
          fontSize: 29,
          color: COLORS.inkDim,
          opacity: ramp(frame, 104, 148),
        }}
      >
        cited {CASE.cited.value} times
        <span style={{color: COLORS.inkFaint, fontSize: 24}}> &#8212; and still rising</span>
        <div style={{marginTop: 8, color: overCount ? COLORS.alarm : COLORS.inkFaint}}>
          {CASE.overruling.value} of them say <em>overruled</em>
        </div>
      </div>

      {/* the stamp */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 666,
          opacity: stamp,
          transform: `rotate(-3deg) scale(${0.85 + 0.15 * stamp})`,
          transformOrigin: 'left top',
          fontFamily: FONT_MONO,
          fontSize: 34,
          letterSpacing: 3,
          color: COLORS.alarm,
          border: `3px solid ${COLORS.alarm}`,
          borderRadius: 6,
          padding: '10px 22px',
          background: 'rgba(224,96,79,0.07)',
          display: 'inline-block',
        }}
      >
        NO LONGER GOOD LAW
      </div>

      <Caption delay={306}>
        That is the gap NyayaTrace is built around: an edge is not just a link, it carries a{' '}
        <em>treatment</em> &#8212; followed, distinguished, mentioned or overruled &#8212; and
        authority has to be computed from the labels, not the arrows.
      </Caption>
    </Scene>
  );
};
