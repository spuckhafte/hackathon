import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Backdrop, Fig, ramp} from '../components/primitives';
import {CORPUS, COURSE, TEAM} from '../data';

/** The close: the mark, the names, the licence, nothing else. */
export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const markIn = ramp(frame, 6, 40);
  const namesIn = ramp(frame, 50, 100);
  const metaIn = ramp(frame, 100, 150);
  const out = interpolate(frame, [180, 210], [1, 0], {extrapolateLeft: 'clamp'});

  return (
    <Backdrop>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: out,
        }}
      >
        <div
          style={{
            fontSize: 120,
            fontWeight: 700,
            letterSpacing: -4,
            opacity: markIn,
            transform: `translateY(${(1 - markIn) * 20}px)`,
          }}
        >
          <span style={{color: COLORS.accent}}>Nyaya</span>
          <span style={{color: COLORS.ink}}>Trace</span>
        </div>
        <div
          style={{
            width: 420 * markIn,
            height: 2,
            background: `linear-gradient(90deg, transparent, ${COLORS.accent}, transparent)`,
            marginTop: 22,
          }}
        />
        <div
          style={{
            marginTop: 40,
            display: 'flex',
            gap: 70,
            opacity: namesIn,
            fontSize: 38,
            fontWeight: 600,
          }}
        >
          {TEAM.map((n, i) => (
            <span key={n} style={{color: [COLORS.accent, COLORS.violet, COLORS.green][i]}}>
              {n}
            </span>
          ))}
        </div>
        <div
          style={{
            marginTop: 54,
            fontFamily: FONT_MONO,
            fontSize: 24,
            color: COLORS.inkFaint,
            textAlign: 'center',
            lineHeight: 1.9,
            opacity: metaIn,
          }}
        >
          <div>
            {COURSE.course} &#183; {COURSE.event} &#183; {COURSE.track}
          </div>
          <div>
            <Fig f={CORPUS.indexed} /> judgments indexed from{' '}
            <Fig f={CORPUS.universe} /> &#183; <Fig f={CORPUS.years} />
          </div>
          <div style={{color: COLORS.inkFaint}}>{CORPUS.licence.value}</div>
        </div>
      </div>
    </Backdrop>
  );
};
