import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO, FONT_SANS} from '../fonts';
import {Backdrop, ramp, useEnter} from '../components/primitives';
import {COURSE, TEAM} from '../data';

/** Deterministic hash-based pseudo-random, so the field is identical per render. */
const rnd = (i: number, salt: number) => {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

const N_NODES = 46;

/**
 * A slowly drifting citation constellation. It is not decoration: it is the
 * object the whole film is about, and it returns in the authority scene.
 */
const Constellation: React.FC<{opacity: number}> = ({opacity}) => {
  const frame = useCurrentFrame();
  const nodes = React.useMemo(
    () =>
      new Array(N_NODES).fill(0).map((_, i) => ({
        x: 120 + rnd(i, 1) * (SIZES.width - 240),
        y: 90 + rnd(i, 2) * (SIZES.height - 180),
        r: 2 + rnd(i, 3) * 4,
        phase: rnd(i, 4) * Math.PI * 2,
        amp: 6 + rnd(i, 5) * 16,
      })),
    []
  );
  const pos = nodes.map((n) => ({
    ...n,
    px: n.x + Math.sin(frame / 90 + n.phase) * n.amp,
    py: n.y + Math.cos(frame / 110 + n.phase) * n.amp * 0.7,
  }));
  const edges: [number, number][] = [];
  for (let i = 0; i < N_NODES; i++) {
    for (let j = i + 1; j < N_NODES; j++) {
      const d = Math.hypot(pos[i].px - pos[j].px, pos[i].py - pos[j].py);
      if (d < 230) edges.push([i, j]);
    }
  }
  return (
    <svg
      width={SIZES.width}
      height={SIZES.height}
      style={{position: 'absolute', inset: 0, opacity}}
    >
      {edges.map(([i, j], k) => {
        const d = Math.hypot(pos[i].px - pos[j].px, pos[i].py - pos[j].py);
        return (
          <line
            key={k}
            x1={pos[i].px}
            y1={pos[i].py}
            x2={pos[j].px}
            y2={pos[j].py}
            stroke={COLORS.cool}
            strokeWidth={0.8}
            opacity={0.1 * (1 - d / 230)}
          />
        );
      })}
      {pos.map((n, i) => (
        <circle
          key={i}
          cx={n.px}
          cy={n.py}
          r={n.r}
          fill={i % 11 === 0 ? COLORS.accent : COLORS.cool}
          opacity={i % 11 === 0 ? 0.5 : 0.25}
        />
      ))}
    </svg>
  );
};

const LETTERS = COURSE.title.split('');

export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const sub = useEnter(44);
  const meta = useEnter(66);
  const rule = ramp(frame, 34, 64);
  const fieldIn = ramp(frame, 0, 60);
  const out = interpolate(frame, [240, 270], [1, 0], {extrapolateLeft: 'clamp'});

  return (
    <Backdrop>
      <Constellation opacity={0.85 * fieldIn * out} />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          paddingLeft: SIZES.margin,
          paddingRight: SIZES.margin,
          opacity: out,
        }}
      >
        <div style={{display: 'flex', alignItems: 'baseline'}}>
          {LETTERS.map((ch, i) => {
            const e = ramp(frame, 6 + i * 3.5, 6 + i * 3.5 + 20);
            return (
              <span
                key={i}
                style={{
                  fontSize: 150,
                  fontWeight: 700,
                  letterSpacing: -5,
                  color: i < 5 ? COLORS.accent : COLORS.ink,
                  opacity: e,
                  transform: `translateY(${(1 - e) * 44}px)`,
                  display: 'inline-block',
                }}
              >
                {ch}
              </span>
            );
          })}
        </div>

        <div
          style={{
            height: 2,
            width: `${rule * 58}%`,
            background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.coolSoft})`,
            marginTop: 10,
            marginBottom: 30,
          }}
        />

        <div
          style={{
            fontSize: 42,
            fontWeight: 300,
            color: COLORS.inkDim,
            opacity: sub,
            transform: `translateY(${(1 - sub) * 12}px)`,
          }}
        >
          {COURSE.tagline}
        </div>

        <div
          style={{
            marginTop: 46,
            display: 'flex',
            gap: 58,
            alignItems: 'center',
            fontFamily: FONT_MONO,
            fontSize: 25,
            letterSpacing: 1.5,
            color: COLORS.inkFaint,
            opacity: meta,
          }}
        >
          <span>{COURSE.course}</span>
          <span style={{color: COLORS.panelEdge}}>|</span>
          <span>{COURSE.event}</span>
          <span style={{color: COLORS.panelEdge}}>|</span>
          <span style={{color: COLORS.cool}}>{COURSE.track}</span>
        </div>

        <div style={{marginTop: 72, display: 'flex', gap: 96, fontFamily: FONT_SANS}}>
          {TEAM.map((name, i) => {
            const e = ramp(frame, 112 + i * 18, 112 + i * 18 + 22);
            return (
              <div
                key={name}
                style={{
                  opacity: e,
                  transform: `translateY(${(1 - e) * 18}px)`,
                }}
              >
                <div
                  style={{
                    width: 46,
                    height: 2,
                    background: [COLORS.accent, COLORS.violet, COLORS.green][i],
                    marginBottom: 14,
                  }}
                />
                <div style={{fontSize: 40, fontWeight: 600, letterSpacing: -0.5}}>{name}</div>
              </div>
            );
          })}
        </div>
      </div>
    </Backdrop>
  );
};
