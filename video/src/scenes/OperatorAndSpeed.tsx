import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, PendingStamp, ramp, Scene} from '../components/primitives';
import {CITATION_RECOVERY, CONFIG, MEASURED, OPERATORS, SPEED} from '../data';

/**
 * Act 5, second half - the two measurements that do not need a human judge.
 *
 * Left: correctness. Two sets slide together; the index result and a
 * brute-force linear scan of the raw text must coincide exactly, operator by
 * operator. Right: speed. Two bars, full postings scan against champion lists
 * plus the tier-one index. Below: the automatic citation-recovery track.
 */

const CX_L = 560;
const CY = 520;
const R = 150;

export const OperatorAndSpeed: React.FC = () => {
  const frame = useCurrentFrame();
  const setsIn = ramp(frame, 10, 60);
  const overlap = ramp(frame, 80, 170);
  const famIn = ramp(frame, 170, 250);
  const speedIn = ramp(frame, 180, 290);
  const recIn = ramp(frame, 280, 350);

  const sep = (1 - overlap) * 118;

  const barFull = 420;
  const barTier = 110; // geometry only; printed values come from SPEED

  return (
    <Scene act="evaluation" heading="Two things a human judge is not needed for.">
      {MEASURED ? null : <PendingStamp />}

      {/* correctness: index vs brute force */}
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        <g opacity={setsIn}>
          <circle cx={CX_L - sep} cy={CY} r={R} fill="rgba(111,179,224,0.12)" stroke={COLORS.cool} strokeWidth={2} />
          <circle cx={CX_L + sep} cy={CY} r={R} fill="rgba(240,178,107,0.12)" stroke={COLORS.accent} strokeWidth={2} />
          <text x={CX_L - sep - R - 16} y={CY - R - 24} textAnchor="start" fontFamily={FONT_MONO} fontSize={23} fill={COLORS.cool}>
            index + skip merges
          </text>
          <text x={CX_L + sep + R + 16} y={CY + R + 44} textAnchor="end" fontFamily={FONT_MONO} fontSize={23} fill={COLORS.accent}>
            brute-force scan of the raw text
          </text>
        </g>
      </svg>

      <div
        style={{
          position: 'absolute',
          left: CX_L - 160,
          top: CY - 46,
          width: 320,
          textAlign: 'center',
          opacity: overlap,
          fontFamily: FONT_MONO,
        }}
      >
        <div style={{fontSize: 58, color: COLORS.green}}>
          <Fig f={OPERATORS.agreement} style={{color: COLORS.green, borderBottomColor: COLORS.amber}} />
        </div>
        <div style={{fontSize: 23, color: COLORS.inkDim, marginTop: 8}}>
          set agreement over <Fig f={OPERATORS.nCases} /> cases
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: CX_L - 300,
          top: CY + 230,
          width: 600,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 10,
          justifyContent: 'center',
          opacity: famIn,
        }}
      >
        {OPERATORS.families.map((f, i) => (
          <span
            key={f}
            style={{
              fontFamily: FONT_MONO,
              fontSize: 22,
              color: COLORS.inkDim,
              border: `1px solid ${COLORS.panelEdge}`,
              borderRadius: 4,
              padding: '6px 13px',
              opacity: ramp(frame, 170 + i * 8, 170 + i * 8 + 14),
            }}
          >
            {f}
          </span>
        ))}
      </div>

      {/* speed */}
      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 300,
          width: 640,
          opacity: speedIn,
        }}
      >
        <div style={{fontFamily: FONT_MONO, fontSize: 21, letterSpacing: 2.5, color: COLORS.inkFaint, marginBottom: 22}}>
          LATENCY, FREE-TEXT FAST PATH
        </div>
        {[
          {label: 'score every posting', w: barFull, color: COLORS.coolSoft, f: SPEED.fullScanMs},
          {
            label: `champion lists (r = ${CONFIG.championListSize.value}) + tier 1`,
            w: barTier,
            color: COLORS.accent,
            f: SPEED.tieredMs,
          },
        ].map((b, i) => {
          const t = ramp(frame, 190 + i * 34, 190 + i * 34 + 36);
          return (
            <div key={b.label} style={{marginBottom: 30}}>
              <div style={{fontFamily: FONT_MONO, fontSize: 23, color: COLORS.inkDim, marginBottom: 8}}>
                {b.label}
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
                <div
                  style={{
                    width: b.w * t,
                    height: 44,
                    background: b.color,
                    borderRadius: 4,
                    opacity: 0.85,
                  }}
                />
                <span style={{fontFamily: FONT_MONO, fontSize: 26}}>
                  <Fig f={b.f} /> <span style={{color: COLORS.inkDim}}>ms</span>
                </span>
              </div>
            </div>
          );
        })}
        <div style={{fontFamily: FONT_MONO, fontSize: 25, color: COLORS.ink, marginTop: 6}}>
          <Fig f={SPEED.speedup} /> faster, <Fig f={SPEED.recallLoss} /> of top-10 recall
          given up
        </div>
        <div style={{fontSize: 22, color: COLORS.inkFaint, marginTop: 10, lineHeight: 1.5}}>
          and separately, skip-pointer merges against a linear walk of the same lists:{' '}
          <Fig f={SPEED.skipMergeGain} /> fewer postings touched
        </div>
      </div>

      {/* citation recovery */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          right: SIZES.margin,
          top: 892,
          opacity: recIn,
          display: 'flex',
          gap: 56,
          alignItems: 'baseline',
          borderTop: `1px solid ${COLORS.panelEdge}`,
          paddingTop: 20,
        }}
      >
        <div style={{fontFamily: FONT_MONO, fontSize: 22, letterSpacing: 2.5, color: COLORS.inkFaint, width: 290}}>
          CITATION RECOVERY
        </div>
        <div style={{fontSize: 24, color: COLORS.inkDim, flex: 1, lineHeight: 1.5}}>
          take a judgment, hide the cases it cites, feed its own holding back as the query:
          R@20 <Fig f={CITATION_RECOVERY.recallAt20} />, MRR{' '}
          <Fig f={CITATION_RECOVERY.mrr} /> over <Fig f={CITATION_RECOVERY.nQueries} />{' '}
          queries &#8212; <Fig f={CITATION_RECOVERY.ceilingNote} />
        </div>
      </div>

      <Caption delay={360}>
        Operator correctness is pass or fail, so it is the one number in this project that is
        allowed to be boring.
      </Caption>
    </Scene>
  );
};
