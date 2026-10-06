import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';
import {STATUTE_SURFACES, STATUTE_TOKEN} from '../data';

/**
 * Act 2, beat 1 - the vocabulary problem.
 *
 * Three real surface forms of the same section sit on three rows. A plain
 * tokenizer cuts each into terms that share almost nothing, so each lands in
 * its own postings list. The three rows then slide together and dissolve into
 * one canonical token.
 */

const ROW_Y = [330, 470, 610];
const TEXT_X = 230;
const LIST_X = 1310;
const CELL = 36;

export const ThreeWays: React.FC = () => {
  const frame = useCurrentFrame();

  const show = (i: number) => ramp(frame, 24 + i * 24, 24 + i * 24 + 20);
  const split = ramp(frame, 110, 165); // the tokenizer cuts them up
  const lists = ramp(frame, 175, 240); // three separate postings lists
  const converge = ramp(frame, 320, 390); // rows slide together and dissolve
  const fuse = ramp(frame, 358, 410);

  return (
    <Scene act="the problem" heading="One section of the Penal Code. Three ways to write it.">
      {/* the three surface forms */}
      {STATUTE_SURFACES.map((s, i) => {
        const e = show(i);
        const dy = converge * (ROW_Y[1] - ROW_Y[i]);
        const alive = e * (1 - converge);
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: TEXT_X,
              top: ROW_Y[i],
              opacity: alive,
              transform: `translateY(${dy}px) scale(${1 - 0.14 * converge})`,
              transformOrigin: 'left center',
              display: 'flex',
              gap: 2 + split * 8,
              alignItems: 'center',
            }}
          >
            {s.split(' ').map((tok, j) => (
              <span
                key={j}
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 38,
                  color: COLORS.ink,
                  padding: `${6 + split * 4}px ${8 + split * 7}px`,
                  borderRadius: 5,
                  background: `rgba(111,179,224,${0.13 * split})`,
                  border: `1px solid rgba(111,179,224,${0.42 * split})`,
                  transform: `translateY(${Math.sin(j * 1.9) * 9 * lists}px)`,
                }}
              >
                {tok}
              </span>
            ))}
          </div>
        );
      })}

      {/* three divergent postings lists */}
      <svg
        width={SIZES.width}
        height={SIZES.height}
        style={{position: 'absolute', inset: 0, opacity: lists * (1 - converge)}}
      >
        <text
          x={LIST_X}
          y={ROW_Y[0] - 56}
          fontFamily={FONT_MONO}
          fontSize={21}
          letterSpacing={2.5}
          fill={COLORS.inkFaint}
        >
          THREE POSTINGS LISTS, NO OVERLAP
        </text>
        {ROW_Y.map((y, i) => (
          <g key={i}>
            <path
              d={`M ${LIST_X - 120} ${y + 24} C ${LIST_X - 70} ${y + 24}, ${
                LIST_X - 60
              } ${y + 24}, ${LIST_X - 14} ${y + 24}`}
              stroke={COLORS.inkFaint}
              strokeWidth={1.4}
              fill="none"
            />
            {[0, 1, 2, 3, 4].map((k) => (
              <g key={k}>
                <rect
                  x={LIST_X + k * (CELL + 10)}
                  y={y + 24 - CELL / 2}
                  width={CELL}
                  height={CELL}
                  rx={4}
                  fill="rgba(111,179,224,0.06)"
                  stroke={COLORS.coolSoft}
                  strokeWidth={1.4}
                />
                {k < 4 ? (
                  <line
                    x1={LIST_X + k * (CELL + 10) + CELL}
                    y1={y + 24}
                    x2={LIST_X + (k + 1) * (CELL + 10)}
                    y2={y + 24}
                    stroke={COLORS.coolSoft}
                    strokeWidth={1.2}
                  />
                ) : null}
              </g>
            ))}
            <text
              x={LIST_X + 5 * (CELL + 10) + 10}
              y={y + 32}
              fontFamily={FONT_MONO}
              fontSize={22}
              fill={COLORS.inkFaint}
            >
              &#8230;
            </text>
          </g>
        ))}
      </svg>

      {/* the fused canonical token */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: ROW_Y[1] - 46,
          display: 'flex',
          justifyContent: 'center',
          opacity: fuse,
          transform: `scale(${0.82 + 0.18 * fuse})`,
        }}
      >
        <div
          style={{
            fontFamily: FONT_MONO,
            fontSize: 96,
            fontWeight: 700,
            color: COLORS.accent,
            border: `2px solid ${COLORS.accent}`,
            borderRadius: 14,
            padding: '20px 54px',
            background: 'rgba(240,178,107,0.08)',
            boxShadow: `0 0 ${70 * fuse}px rgba(240,178,107,0.2)`,
          }}
        >
          {STATUTE_TOKEN}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: ROW_Y[1] + 120,
          textAlign: 'center',
          fontSize: 32,
          color: COLORS.inkDim,
          opacity: ramp(frame, 400, 440),
        }}
      >
        one term, one postings list, one place to look
      </div>

      <Caption delay={130}>
        A general-purpose tokenizer is not wrong here, it is simply blind: these three
        strings share almost no terms, so the same provision ends up in three different
        places in the index.
      </Caption>
    </Scene>
  );
};
