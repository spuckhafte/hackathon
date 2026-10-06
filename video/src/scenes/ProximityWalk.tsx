import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';

/**
 * Act 4 - the proximity operators, Rakshit's component.
 *
 * Two position lists for one document, walked by two pointers in lock-step.
 * Because every token carries its paragraph and sentence id as well as its
 * offset, the same walk answers /n, /s and /p: only the comparison changes.
 * The pointers really move, one step per beat, and the matching pair lights.
 */

type Hit = {pos: number; para: number; sent: number};

const LEFT_HITS: Hit[] = [
  {pos: 118, para: 7, sent: 22},
  {pos: 341, para: 14, sent: 61},
  {pos: 902, para: 31, sent: 150},
  {pos: 1204, para: 41, sent: 206},
];
const RIGHT_HITS: Hit[] = [
  {pos: 95, para: 6, sent: 19},
  {pos: 360, para: 14, sent: 63},
  {pos: 1190, para: 41, sent: 205},
  {pos: 1588, para: 55, sent: 271},
];

/** The merge, pre-computed so the animation is the algorithm, not a cartoon. */
type Step = {i: number; j: number; verdict: 'advance-left' | 'advance-right' | 'match'};
const STEPS: Step[] = (() => {
  const out: Step[] = [];
  let i = 0;
  let j = 0;
  while (i < LEFT_HITS.length && j < RIGHT_HITS.length) {
    const a = LEFT_HITS[i];
    const b = RIGHT_HITS[j];
    if (a.para === b.para) {
      out.push({i, j, verdict: 'match'});
      i++;
      j++;
    } else if (a.para < b.para) {
      out.push({i, j, verdict: 'advance-left'});
      i++;
    } else {
      out.push({i, j, verdict: 'advance-right'});
      j++;
    }
  }
  return out;
})();

const FRAMES_PER_STEP = 42;
const START = 96;
const COL_X = 430;
const COL_W = 236;
const ROW_L = 380;
const ROW_R = 620;

const Cell: React.FC<{h: Hit; x: number; y: number; active: boolean; matched: boolean; color: string}> = ({
  h,
  x,
  y,
  active,
  matched,
  color,
}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width: COL_W - 22,
      border: `${active || matched ? 2.2 : 1.3}px solid ${
        matched ? COLORS.green : active ? color : COLORS.panelEdge
      }`,
      background: matched
        ? 'rgba(123,201,140,0.12)'
        : active
        ? `${color}1a`
        : 'rgba(21,27,36,0.6)',
      borderRadius: 7,
      padding: '12px 14px',
      fontFamily: FONT_MONO,
      transition: 'none',
    }}
  >
    <div style={{fontSize: 27, color: matched ? COLORS.green : COLORS.ink}}>
      pos {h.pos}
    </div>
    <div style={{fontSize: 19, color: COLORS.inkDim, marginTop: 4}}>
      &#182;{h.para} &#183; sent {h.sent}
    </div>
  </div>
);

export const ProximityWalk: React.FC = () => {
  const frame = useCurrentFrame();
  const stepIdx = Math.max(
    0,
    Math.min(STEPS.length - 1, Math.floor((frame - START) / FRAMES_PER_STEP))
  );
  const started = frame >= START;
  const step = STEPS[stepIdx];
  const matchedPairs = STEPS.slice(0, stepIdx + 1).filter((s) => s.verdict === 'match');

  const matchedLeft = new Set(matchedPairs.map((s) => s.i));
  const matchedRight = new Set(matchedPairs.map((s) => s.j));

  const queryIn = ramp(frame, 4, 28);
  const listsIn = ramp(frame, 30, 70);

  return (
    <Scene act="the pipeline · proximity" heading="Two pointers, one pass, three operators.">
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          top: 258,
          fontFamily: FONT_MONO,
          fontSize: 36,
          opacity: queryIn,
        }}
      >
        <span style={{color: COLORS.cool}}>privacy</span>{' '}
        <span style={{color: COLORS.accent}}>/p</span>{' '}
        <span style={{color: COLORS.cool}}>&#8220;fundamental right&#8221;</span>
        <span style={{color: COLORS.inkFaint, fontSize: 25, marginLeft: 26}}>
          inside doc 8142
        </span>
      </div>

      {/* the two lists */}
      <div
        style={{
          position: 'absolute',
          left: COL_X - 220,
          top: ROW_L + 18,
          width: 200,
          textAlign: 'right',
          fontFamily: FONT_MONO,
          fontSize: 24,
          color: COLORS.cool,
          opacity: listsIn,
        }}
      >
        privaci
      </div>
      <div
        style={{
          position: 'absolute',
          left: COL_X - 220,
          top: ROW_R + 18,
          width: 200,
          textAlign: 'right',
          fontFamily: FONT_MONO,
          fontSize: 24,
          color: COLORS.cool,
          opacity: listsIn,
        }}
      >
        fundament
      </div>

      <div style={{opacity: listsIn}}>
        {LEFT_HITS.map((h, k) => (
          <Cell
            key={k}
            h={h}
            x={COL_X + k * COL_W}
            y={ROW_L}
            active={started && step.i === k && step.verdict !== 'match'}
            matched={matchedLeft.has(k)}
            color={COLORS.accent}
          />
        ))}
        {RIGHT_HITS.map((h, k) => (
          <Cell
            key={k}
            h={h}
            x={COL_X + k * COL_W}
            y={ROW_R}
            active={started && step.j === k && step.verdict !== 'match'}
            matched={matchedRight.has(k)}
            color={COLORS.violet}
          />
        ))}
      </div>

      {/* pointers */}
      {started ? (
        <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
          <g>
            <polygon
              points={`${COL_X + step.i * COL_W + (COL_W - 22) / 2},${ROW_L - 14} ${
                COL_X + step.i * COL_W + (COL_W - 22) / 2 - 11
              },${ROW_L - 40} ${COL_X + step.i * COL_W + (COL_W - 22) / 2 + 11},${ROW_L - 40}`}
              fill={COLORS.accent}
            />
            <text
              x={COL_X + step.i * COL_W + (COL_W - 22) / 2}
              y={ROW_L - 50}
              textAnchor="middle"
              fontFamily={FONT_MONO}
              fontSize={22}
              fill={COLORS.accent}
            >
              i
            </text>
            <polygon
              points={`${COL_X + step.j * COL_W + (COL_W - 22) / 2},${ROW_R + 110} ${
                COL_X + step.j * COL_W + (COL_W - 22) / 2 - 11
              },${ROW_R + 136} ${COL_X + step.j * COL_W + (COL_W - 22) / 2 + 11},${ROW_R + 136}`}
              fill={COLORS.violet}
            />
            <text
              x={COL_X + step.j * COL_W + (COL_W - 22) / 2}
              y={ROW_R + 164}
              textAnchor="middle"
              fontFamily={FONT_MONO}
              fontSize={22}
              fill={COLORS.violet}
            >
              j
            </text>
          </g>
          {step.verdict === 'match' ? (
            <line
              x1={COL_X + step.i * COL_W + (COL_W - 22) / 2}
              y1={ROW_L + 88}
              x2={COL_X + step.j * COL_W + (COL_W - 22) / 2}
              y2={ROW_R - 6}
              stroke={COLORS.green}
              strokeWidth={3}
            />
          ) : null}
        </svg>
      ) : null}

      {/* verdict readout */}
      {started ? (
        <div
          style={{
            position: 'absolute',
            left: SIZES.margin,
            top: 830,
            fontFamily: FONT_MONO,
            fontSize: 28,
            color:
              step.verdict === 'match'
                ? COLORS.green
                : step.verdict === 'advance-left'
                ? COLORS.accent
                : COLORS.violet,
          }}
        >
          {step.verdict === 'match'
            ? `¶${LEFT_HITS[step.i].para} == ¶${RIGHT_HITS[step.j].para}  →  emit, advance both`
            : step.verdict === 'advance-left'
            ? `¶${LEFT_HITS[step.i].para} < ¶${RIGHT_HITS[step.j].para}  →  i++`
            : `¶${LEFT_HITS[step.i].para} > ¶${RIGHT_HITS[step.j].para}  →  j++`}
        </div>
      ) : null}

      {/* the three comparisons */}
      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 258,
          width: 400,
          fontFamily: FONT_MONO,
          fontSize: 24,
          lineHeight: 1.9,
          color: COLORS.inkDim,
          opacity: ramp(frame, 60, 100),
        }}
      >
        <div>
          <span style={{color: COLORS.accent}}>/n k</span> &nbsp;|pos<sub>a</sub> &#8722; pos
          <sub>b</sub>| &#8804; k
        </div>
        <div>
          <span style={{color: COLORS.accent}}>/s</span> &nbsp;&nbsp;&nbsp;sent<sub>a</sub> ==
          sent<sub>b</sub>
        </div>
        <div>
          <span style={{color: COLORS.accent}}>/p</span> &nbsp;&nbsp;&nbsp;para<sub>a</sub> ==
          para<sub>b</sub>
        </div>
        <div style={{marginTop: 18, fontSize: 21, color: COLORS.inkFaint, lineHeight: 1.5}}>
          one merge, three predicates &#8212; the ids are in the posting, so none of this
          needs the text
        </div>
      </div>

      <Caption delay={380}>
        A phrase is just <span style={{fontFamily: FONT_MONO}}>/n 1</span> with the order
        kept, so the same routine serves quoted phrases too. Linear in the two lists, never
        quadratic.
      </Caption>
    </Scene>
  );
};
