import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {LIMITATION} from '../data';

/**
 * Act 6 - where it breaks, shown rather than confessed.
 *
 * The cue-phrase classifier is a surface matcher, so a sentence that *denies* an
 * overruling still contains the cue for one. We highlight the cue, let the wrong
 * label land, then draw the negation scope the classifier never sees, and end on
 * the roadmap drawn as a path with three stations - not a bulleted list.
 */

const SENTENCE =
  'We are unable to hold that the decision in Rajesh v. State stands overruled by the later judgment.';

const NEG = 'unable to hold that';
const CUE = 'stands overruled';

const FONT_SIZE = 27;
const CW = FONT_SIZE * 0.6; // monospace advance width
const SENT_X = SIZES.margin;
const SENT_Y = 282;

const at = (i: number) => SENT_X + i * CW;
const negStart = SENTENCE.indexOf(NEG);
const cueStart = SENTENCE.indexOf(CUE);

type Piece = {text: string; kind: 'plain' | 'cue' | 'neg'};

const PIECES: Piece[] = [
  {text: SENTENCE.slice(0, negStart), kind: 'plain'},
  {text: NEG, kind: 'neg'},
  {text: SENTENCE.slice(negStart + NEG.length, cueStart), kind: 'plain'},
  {text: CUE, kind: 'cue'},
  {text: SENTENCE.slice(cueStart + CUE.length), kind: 'plain'},
];

const STATIONS = [
  {
    when: 'next',
    what: 'negation and modality scope',
    how: 'a dependency window around the cue, so "unable to hold" cancels the match',
  },
  {
    when: 'then',
    what: 'learned treatment classifier',
    how: 'the hand-labelled set becomes training data instead of only a test set',
  },
  {
    when: 'after',
    what: 'High Courts and full-corpus text',
    how: 'drop the per-year cap, which is what currently caps citation recall',
  },
];

const ROAD_W = SIZES.width - SIZES.margin * 2;
const GAP = 56;
const COL_W = (ROAD_W - GAP * (STATIONS.length - 1)) / STATIONS.length;
const colCentre = (i: number) => i * (COL_W + GAP) + COL_W / 2;

export const Limitation: React.FC = () => {
  const frame = useCurrentFrame();
  const sentIn = ramp(frame, 8, 44);
  const cueIn = ramp(frame, 66, 106);
  const wrongIn = ramp(frame, 126, 172);
  const negIn = ramp(frame, 220, 278);
  const statsIn = ramp(frame, 286, 330);
  const roadIn = ramp(frame, 352, 430);

  return (
    <Scene act="limitations" heading="A surface cue cannot read a denial.">
      <div
        style={{
          position: 'absolute',
          left: SENT_X,
          top: SENT_Y,
          fontSize: FONT_SIZE,
          fontFamily: FONT_MONO,
          whiteSpace: 'pre',
          letterSpacing: 0,
          opacity: sentIn,
        }}
      >
        {PIECES.map((p, i) => {
          const hl = p.kind === 'cue' ? cueIn : p.kind === 'neg' ? negIn : 0;
          const color =
            p.kind === 'cue' ? COLORS.alarm : p.kind === 'neg' ? COLORS.green : COLORS.ink;
          return (
            <span
              key={i}
              style={{
                color: hl > 0.2 ? color : COLORS.ink,
                background: hl > 0.05 ? `${color}26` : 'transparent',
                borderBottom: hl > 0.2 ? `2px solid ${color}` : '2px solid transparent',
              }}
            >
              {p.text}
            </span>
          );
        })}
      </div>

      {/* the scope the classifier does not see */}
      <svg
        width={SIZES.width}
        height={SIZES.height}
        style={{position: 'absolute', inset: 0, opacity: negIn}}
      >
        <path
          d={`M ${at(negStart) + 4} 332 C ${at(negStart) + 4} 410, ${
            at(cueStart + CUE.length) - 4
          } 410, ${at(cueStart + CUE.length) - 4} 332`}
          stroke={COLORS.green}
          strokeWidth={2}
          fill="none"
          strokeDasharray="7 8"
        />
        <text
          x={(at(negStart) + at(cueStart + CUE.length)) / 2}
          y={434}
          fontFamily={FONT_MONO}
          fontSize={23}
          fill={COLORS.green}
          textAnchor="middle"
        >
          the scope that actually governs the cue
        </text>
      </svg>

      {/* the wrong label */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          right: SIZES.margin,
          top: 486,
          display: 'flex',
          alignItems: 'center',
          gap: 22,
          opacity: wrongIn,
        }}
      >
        <span style={{fontFamily: FONT_MONO, fontSize: 23, color: COLORS.inkFaint}}>
          classifier &#8594;
        </span>
        <span
          style={{
            fontFamily: FONT_MONO,
            fontSize: 29,
            color: COLORS.alarm,
            border: `1.5px solid ${COLORS.alarm}`,
            borderRadius: 6,
            padding: '7px 17px',
          }}
        >
          overruled
        </span>
        <span style={{fontSize: 27, color: COLORS.inkDim}}>
          so a case that was explicitly <em>upheld</em> loses three quarters of its g(d)
        </span>
      </div>

      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          right: SIZES.margin,
          top: 562,
          fontSize: 26,
          lineHeight: 1.5,
          color: COLORS.inkDim,
          opacity: statsIn,
          maxWidth: 1400,
        }}
      >
        On our hand-labelled sample of <Fig f={LIMITATION.treatmentSampleSize} /> edges the
        classifier is right <Fig f={LIMITATION.treatmentAccuracy} /> of the time, and{' '}
        <Fig f={LIMITATION.negationErrorRate} /> of its errors are negated or hypothetical cues
        like this one.
      </div>

      {/* roadmap as a path, not a list */}
      <div style={{position: 'absolute', left: SIZES.margin, right: SIZES.margin, top: 672}}>
        <svg width={ROAD_W} height={34} style={{display: 'block'}}>
          <line
            x1={colCentre(0)}
            y1={17}
            x2={colCentre(0) + (colCentre(STATIONS.length - 1) - colCentre(0)) * roadIn}
            y2={17}
            stroke={COLORS.accent}
            strokeWidth={2}
            opacity={0.6}
          />
          {STATIONS.map((_, i) => {
            const t = ramp(frame, 358 + i * 32, 358 + i * 32 + 24);
            return <circle key={i} cx={colCentre(i)} cy={17} r={9 * t} fill={COLORS.accent} />;
          })}
        </svg>
        <div style={{display: 'flex', gap: GAP, marginTop: 16}}>
          {STATIONS.map((s, i) => {
            const t = ramp(frame, 366 + i * 32, 366 + i * 32 + 28);
            return (
              <div
                key={s.what}
                style={{
                  width: COL_W,
                  opacity: t,
                  transform: `translateY(${(1 - t) * 14}px)`,
                }}
              >
                <div
                  style={{
                    fontFamily: FONT_MONO,
                    fontSize: 20,
                    color: COLORS.accent,
                    letterSpacing: 2.5,
                  }}
                >
                  {s.when.toUpperCase()}
                </div>
                <div style={{fontSize: 30, marginTop: 10, lineHeight: 1.2}}>{s.what}</div>
                <div style={{fontSize: 22, color: COLORS.inkDim, marginTop: 10, lineHeight: 1.5}}>
                  {s.how}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <Caption delay={440}>
        We are not fixing this by hand-patching the phrase list. The fix is a scope, and until it
        lands the flag is visible in the interface with the sentence that caused it.
      </Caption>
    </Scene>
  );
};
