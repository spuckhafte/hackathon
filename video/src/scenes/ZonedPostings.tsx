import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {CONFIG} from '../data';

/**
 * Act 4 - the index itself, Rakshit's component.
 *
 * The dictionary row slides open into a postings list; each cell grows a
 * per-zone tf column and a positions tail; skip arcs are drawn over every
 * sqrt(df)-th cell. The point is that one posting carries three things at once:
 * where, how often per zone, and exactly where in the text.
 */

const DOC_IDS = [112, 907, 1840, 2211, 3078, 4190, 5533, 6102, 7765, 8142, 9310, 11004];
const SKIP = 4; // sqrt(df) rounded, for df = 12..16

const ZONE_TF = [
  [0, 1, 4], [0, 0, 2], [1, 2, 9], [0, 0, 1], [0, 1, 3], [0, 0, 5],
  [1, 0, 2], [0, 2, 6], [0, 0, 1], [1, 3, 14], [0, 0, 2], [0, 1, 4],
];
const POSITIONS = ['3, 41, 88', '210', '7, 9, 64', '1902', '77, 310', '15', '2, 5',
  '44, 51', '880', '1, 12, 19', '501', '66, 140'];

const LEFT = 230;
const CELL_W = 118;
const TOP = 420;

export const ZonedPostings: React.FC = () => {
  const frame = useCurrentFrame();
  const dictIn = ramp(frame, 6, 32);
  const cellsIn = (i: number) => ramp(frame, 46 + i * 7, 46 + i * 7 + 16);
  const zonesIn = ramp(frame, 180, 250);
  const posIn = ramp(frame, 250, 320);
  const skipsIn = ramp(frame, 340, 420);

  return (
    <Scene act="the pipeline · index" heading="One posting, three facts.">
      {/* dictionary entry */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 262,
          display: 'flex',
          alignItems: 'center',
          gap: 22,
          opacity: dictIn,
          fontFamily: FONT_MONO,
          fontSize: 30,
        }}
      >
        <span style={{color: COLORS.inkFaint, fontSize: 22, letterSpacing: 2}}>
          DICTIONARY
        </span>
        <span
          style={{
            color: COLORS.accent,
            border: `1.5px solid ${COLORS.accent}`,
            borderRadius: 6,
            padding: '8px 18px',
          }}
        >
          privaci
        </span>
        <span style={{color: COLORS.inkDim}}>&#8594;</span>
        <span style={{color: COLORS.inkDim}}>df {DOC_IDS.length}</span>
        <span style={{color: COLORS.panelEdge}}>|</span>
        <span style={{color: COLORS.inkDim}}>offset 0x1f4a20</span>
        <span style={{color: COLORS.panelEdge}}>|</span>
        <span style={{color: COLORS.inkDim}}>skip stride {SKIP}</span>
      </div>

      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {/* skip arcs */}
        {DOC_IDS.map((_, i) => {
          if (i % SKIP !== 0 || i + SKIP >= DOC_IDS.length) return null;
          const x1 = LEFT + i * CELL_W + CELL_W / 2;
          const x2 = LEFT + (i + SKIP) * CELL_W + CELL_W / 2;
          const t = ramp(frame, 340 + i * 8, 340 + i * 8 + 24);
          return (
            <g key={i} opacity={t * skipsIn}>
              <path
                d={`M ${x1} ${TOP - 10} C ${x1} ${TOP - 90}, ${x2} ${TOP - 90}, ${x2} ${
                  TOP - 10
                }`}
                stroke={COLORS.violet}
                strokeWidth={2}
                fill="none"
              />
              <polygon
                points={`${x2},${TOP - 10} ${x2 - 6},${TOP - 24} ${x2 + 6},${TOP - 24}`}
                fill={COLORS.violet}
              />
            </g>
          );
        })}
        <text
          x={LEFT}
          y={TOP - 108}
          fontFamily={FONT_MONO}
          fontSize={21}
          letterSpacing={2.5}
          fill={COLORS.violet}
          opacity={skipsIn}
        >
          SKIP POINTERS &#8212; STRIDE {SKIP} &#8776; &#8730;df
        </text>
      </svg>

      {/* postings cells */}
      {DOC_IDS.map((d, i) => {
        const t = cellsIn(i);
        return (
          <div
            key={d}
            style={{
              position: 'absolute',
              left: LEFT + i * CELL_W,
              top: TOP,
              width: CELL_W - 10,
              opacity: t,
              transform: `translateX(${(1 - t) * -24}px)`,
            }}
          >
            <div
              style={{
                height: 58,
                border: `1.5px solid ${i === 9 ? COLORS.accent : COLORS.coolSoft}`,
                background: i === 9 ? 'rgba(240,178,107,0.1)' : 'rgba(111,179,224,0.06)',
                borderRadius: 6,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: FONT_MONO,
                fontSize: 25,
                color: i === 9 ? COLORS.accent : COLORS.ink,
              }}
            >
              {d}
            </div>

            {/* per-zone tf: statutes / order / body */}
            <div
              style={{
                marginTop: 10,
                display: 'flex',
                gap: 4,
                alignItems: 'flex-end',
                height: 86,
                opacity: zonesIn,
              }}
            >
              {ZONE_TF[i].map((v, z) => (
                <div
                  key={z}
                  style={{
                    width: 30,
                    height: Math.max(3, (v / 14) * 80 * zonesIn),
                    background: [COLORS.accent, COLORS.cool, COLORS.coolSoft][z],
                    opacity: 0.8,
                    borderRadius: 2,
                  }}
                />
              ))}
            </div>

            <div
              style={{
                marginTop: 12,
                fontFamily: FONT_MONO,
                fontSize: 18,
                color: COLORS.inkDim,
                opacity: posIn,
                lineHeight: 1.35,
              }}
            >
              {POSITIONS[i]}
            </div>
          </div>
        );
      })}

      {/* legends */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin - 76,
          top: TOP + 72,
          width: 290,
          textAlign: 'right',
          fontFamily: FONT_MONO,
          fontSize: 20,
          color: COLORS.inkDim,
          opacity: zonesIn,
          lineHeight: 1.5,
        }}
      >
        <div style={{color: COLORS.accent}}>statutes tf</div>
        <div style={{color: COLORS.cool}}>order tf</div>
        <div style={{color: COLORS.coolSoft}}>body tf</div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin - 76,
          top: TOP + 168,
          width: 290,
          textAlign: 'right',
          fontFamily: FONT_MONO,
          fontSize: 20,
          color: COLORS.inkFaint,
          opacity: posIn,
        }}
      >
        positions
      </div>

      <Caption delay={430}>
        Built with SPIMI: <Fig f={CONFIG.spimiBlockDocs} /> documents inverted in memory,
        flushed to a block,
        then a single k-way merge into one memory-mapped int32 array. Nothing holds the
        whole index in RAM, and nothing is a Python dictionary at query time.
      </Caption>
    </Scene>
  );
};
