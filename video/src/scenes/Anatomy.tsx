import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {CONFIG, CORPUS} from '../data';

/**
 * Act 2, beat 2 - a judgment is not a bag of words, it is a document with
 * parts. A page of running text separates into six bands; the band that carries
 * the holding is tiny next to the body, which is why an unweighted model drowns
 * in recitals.
 */

type Band = {zone: string; share: number; sample: string};

const BANDS: Band[] = [
  {zone: 'title', share: 0.035, sample: 'K.S. Puttaswamy v. Union of India'},
  {zone: 'parties', share: 0.045, sample: 'petitioner / respondent / advocates'},
  {zone: 'bench', share: 0.04, sample: 'nine judges, names and seniority'},
  {zone: 'statutes', share: 0.06, sample: 'const§21  const§368  it-act§43a'},
  {zone: 'order', share: 0.1, sample: 'the operative holding and directions'},
  {zone: 'body', share: 0.72, sample: 'facts, submissions, precedent, reasoning'},
];

const TOP = 300;
const COL_X = 300;
const COL_W = 470;
const COL_H = 560;

export const Anatomy: React.FC = () => {
  const frame = useCurrentFrame();
  const pageIn = ramp(frame, 10, 44);
  const separate = ramp(frame, 70, 150);
  const weightsIn = ramp(frame, 180, 250);

  let acc = 0;
  const geom = BANDS.map((b) => {
    const h = b.share * COL_H;
    const y = TOP + acc;
    acc += h;
    return {...b, y, h};
  });

  const zoneW = CONFIG.zoneWeights as Record<string, number>;
  const maxW = Math.max(...Object.values(zoneW));

  return (
    <Scene act="the problem" heading="A judgment has parts, and they are not equally informative.">
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {/* running text, before separation */}
        <g opacity={pageIn * (1 - separate)}>
          {new Array(26).fill(0).map((_, i) => (
            <rect
              key={i}
              x={COL_X}
              y={TOP + i * 21}
              width={COL_W * (i % 7 === 6 ? 0.55 : 0.97)}
              height={8}
              rx={4}
              fill={COLORS.inkFaint}
              opacity={0.35}
            />
          ))}
        </g>

        {geom.map((b, i) => {
          const gap = separate * 11 * i;
          const w = zoneW[b.zone] ?? 1;
          const filterOnly = w === 0;
          const color = filterOnly ? COLORS.inkFaint : i === 5 ? COLORS.coolSoft : COLORS.cool;
          return (
            <g key={b.zone} opacity={separate}>
              <rect
                x={COL_X}
                y={b.y + gap}
                width={COL_W}
                height={Math.max(10, b.h - 4)}
                rx={5}
                fill={`${color}1f`}
                stroke={color}
                strokeWidth={1.4}
              />
              <text
                x={COL_X - 18}
                y={b.y + gap + Math.max(10, b.h - 4) / 2 + 8}
                textAnchor="end"
                fontFamily={FONT_MONO}
                fontSize={25}
                fill={filterOnly ? COLORS.inkFaint : COLORS.ink}
              >
                {b.zone}
              </text>
              <text
                x={COL_X + 16}
                y={b.y + gap + Math.min(32, Math.max(10, b.h - 4) / 2 + 8)}
                fontSize={20}
                fill={COLORS.inkDim}
                fontFamily={FONT_MONO}
              >
                {b.sample}
              </text>

              {/* zone weight bar */}
              <g opacity={weightsIn}>
                <line
                  x1={COL_X + COL_W + 14}
                  y1={b.y + gap + Math.max(10, b.h - 4) / 2}
                  x2={COL_X + COL_W + 60}
                  y2={b.y + gap + Math.max(10, b.h - 4) / 2}
                  stroke={COLORS.panelEdge}
                  strokeWidth={1}
                />
                <rect
                  x={COL_X + COL_W + 66}
                  y={b.y + gap + Math.max(10, b.h - 4) / 2 - 11}
                  width={Math.max(3, (w / maxW) * 300 * weightsIn)}
                  height={22}
                  rx={3}
                  fill={filterOnly ? COLORS.inkFaint : COLORS.accent}
                  opacity={filterOnly ? 0.3 : 0.85}
                />
                <text
                  x={COL_X + COL_W + 66 + Math.max(3, (w / maxW) * 300) + 14}
                  y={b.y + gap + Math.max(10, b.h - 4) / 2 + 7}
                  fontFamily={FONT_MONO}
                  fontSize={23}
                  fill={filterOnly ? COLORS.inkFaint : COLORS.accent}
                >
                  {filterOnly ? 'filter only' : w.toFixed(1)}
                </text>
              </g>
            </g>
          );
        })}

        <text
          x={COL_X + COL_W + 66}
          y={TOP - 26}
          fontFamily={FONT_MONO}
          fontSize={21}
          letterSpacing={2.5}
          fill={COLORS.inkFaint}
          opacity={weightsIn}
        >
          ZONE WEIGHT
        </text>
      </svg>

      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: TOP + 420,
          width: 430,
          fontSize: 28,
          lineHeight: 1.55,
          color: COLORS.inkDim,
          opacity: ramp(frame, 270, 320),
        }}
      >
        <Fig f={CORPUS.bodyTokenShare} /> of all indexed tokens fall in the body zone.
        The holding lives in a band a fraction of that size, and the bench band is
        never scored at all &#8212; it exists so that{' '}
        <span style={{fontFamily: FONT_MONO, color: COLORS.cool}}>bench:&gt;=5</span>{' '}
        can be a filter.
      </div>

      <Caption delay={210}>
        So we index six zones separately and keep a per-zone term frequency in every
        posting. The same word is worth more in a title than in the twentieth
        paragraph of the facts.
      </Caption>
    </Scene>
  );
};
