/**
 * Shared visual primitives. Everything in `scenes/` is built from these so the
 * whole film has one grammar: one background, one caption position, one way a
 * formula is set, one way a still-unmeasured figure is marked.
 */
import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {COLORS, SIZES, SPRING} from '../theme';
import {FONT_MONO, FONT_SANS, FONT_SERIF} from '../fonts';
import type {Figure} from '../data';

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** 0 -> 1 spring that starts at `delay` frames into the current sequence. */
export const useEnter = (delay = 0, cfg = SPRING) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: cfg, durationInFrames: 28});
};

/** Fades a scene in over `inF` frames and out over the last `outF` frames. */
export const useSceneFade = (inF = 7, outF = 12) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return Math.min(
    interpolate(frame, [0, inF], [0, 1], {extrapolateRight: 'clamp'}),
    interpolate(frame, [durationInFrames - outF, durationInFrames], [1, 0], {
      extrapolateLeft: 'clamp',
    })
  );
};

/** Linear 0..1 ramp between two frames, clamped. */
export const ramp = (frame: number, a: number, b: number) =>
  interpolate(frame, [a, b], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

// ---------------------------------------------------------------------------
// Scene chrome
// ---------------------------------------------------------------------------

/** The one and only background: a deep near-black with a soft radial lift. */
export const Backdrop: React.FC<{children?: React.ReactNode}> = ({children}) => (
  <AbsoluteFill
    style={{
      backgroundColor: COLORS.bg,
      backgroundImage: `radial-gradient(1400px 900px at 50% 32%, #141d28 0%, ${COLORS.bg} 55%, ${COLORS.bgDeep} 100%)`,
      fontFamily: FONT_SANS,
      color: COLORS.ink,
    }}
  >
    {children}
  </AbsoluteFill>
);

/**
 * Every explained segment sits in a Scene: a small act label top-left, a
 * heading, and a body area with generous margins. No bullets anywhere.
 */
export const Scene: React.FC<{
  act?: string;
  heading?: string;
  children?: React.ReactNode;
  /** Suppress the automatic fade (used by the title card). */
  noFade?: boolean;
}> = ({act, heading, children, noFade}) => {
  const fade = useSceneFade();
  const head = useEnter(2);
  return (
    <Backdrop>
      <AbsoluteFill style={{opacity: noFade ? 1 : fade}}>
        {act ? (
          <div
            style={{
              position: 'absolute',
              left: SIZES.margin,
              top: 72,
              fontFamily: FONT_MONO,
              fontSize: 22,
              letterSpacing: 4,
              textTransform: 'uppercase',
              color: COLORS.inkFaint,
              opacity: head,
            }}
          >
            {act}
          </div>
        ) : null}
        {heading ? (
          <div
            style={{
              position: 'absolute',
              left: SIZES.margin,
              top: 124,
              right: SIZES.margin,
              fontSize: SIZES.h2,
              fontWeight: 300,
              lineHeight: 1.12,
              letterSpacing: -1,
              color: COLORS.ink,
              opacity: head,
              transform: `translateY(${(1 - head) * 14}px)`,
            }}
          >
            {heading}
          </div>
        ) : null}
        {children}
      </AbsoluteFill>
    </Backdrop>
  );
};

/** A single line of commentary, always at the same place: bottom-left. */
export const Caption: React.FC<{children: React.ReactNode; delay?: number}> = ({
  children,
  delay = 0,
}) => {
  const e = useEnter(delay);
  return (
    <div
      style={{
        position: 'absolute',
        left: SIZES.margin,
        right: SIZES.margin,
        bottom: 88,
        fontSize: SIZES.caption,
        color: COLORS.inkDim,
        fontWeight: 300,
        lineHeight: 1.45,
        maxWidth: 1180,
        opacity: e,
        transform: `translateY(${(1 - e) * 10}px)`,
      }}
    >
      {children}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Figures (the only place a number ever reaches the screen)
// ---------------------------------------------------------------------------

/**
 * Prints a {@link Figure}. Unmeasured figures get a dashed amber underline and
 * a tiny caret, so a placeholder can never slip into a final render unnoticed.
 */
export const Fig: React.FC<{f: Figure; style?: React.CSSProperties}> = ({f, style}) => (
  <span
    style={{
      fontFamily: FONT_MONO,
      color: f.measured ? COLORS.ink : COLORS.amber,
      borderBottom: f.measured ? 'none' : `2px dashed ${COLORS.amber}`,
      paddingBottom: f.measured ? 0 : 2,
      whiteSpace: 'nowrap',
      ...style,
    }}
    title={f.measured ? undefined : 'placeholder - awaiting measurement'}
  >
    {f.value}
  </span>
);

/** Stamp drawn on any chart whose geometry is not yet real. */
export const PendingStamp: React.FC<{label?: string}> = ({
  label = 'shape only — awaiting measurement',
}) => (
  <div
    style={{
      position: 'absolute',
      right: SIZES.margin,
      top: 84,
      fontFamily: FONT_MONO,
      fontSize: 20,
      letterSpacing: 2,
      color: COLORS.amber,
      border: `1px dashed ${COLORS.amber}`,
      borderRadius: 4,
      padding: '6px 14px',
      opacity: 0.85,
    }}
  >
    {label}
  </div>
);

// ---------------------------------------------------------------------------
// Typeset maths
// ---------------------------------------------------------------------------

/**
 * Light-weight mathematical typesetting: serif italics for variables, upright
 * mono for operators, real fractions. Enough for the four formulas we show,
 * without dragging in a TeX engine.
 */
export const MathRow: React.FC<{
  children: React.ReactNode;
  size?: number;
  style?: React.CSSProperties;
}> = ({children, size = 52, style}) => (
  <div
    style={{
      fontFamily: FONT_SERIF,
      fontSize: size,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 10,
      color: COLORS.ink,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Var: React.FC<{children: React.ReactNode; color?: string}> = ({
  children,
  color,
}) => <span style={{fontStyle: 'italic', color: color ?? COLORS.ink}}>{children}</span>;

export const Op: React.FC<{children?: React.ReactNode}> = ({children}) => (
  <span style={{color: COLORS.inkDim, padding: '0 6px', fontStyle: 'normal'}}>
    {children}
  </span>
);

export const Frac: React.FC<{
  num: React.ReactNode;
  den: React.ReactNode;
  color?: string;
}> = ({num, den, color}) => (
  <span
    style={{
      display: 'inline-flex',
      flexDirection: 'column',
      alignItems: 'center',
      verticalAlign: 'middle',
      lineHeight: 1.15,
      fontSize: '0.84em',
      color,
    }}
  >
    <span style={{padding: '0 10px'}}>{num}</span>
    <span
      style={{
        height: 2,
        alignSelf: 'stretch',
        background: COLORS.inkFaint,
        margin: '6px 0',
      }}
    />
    <span style={{padding: '0 10px'}}>{den}</span>
  </span>
);

export const Sub: React.FC<{children: React.ReactNode}> = ({children}) => (
  <sub style={{fontSize: '0.56em', fontStyle: 'normal', color: COLORS.inkDim}}>
    {children}
  </sub>
);

// ---------------------------------------------------------------------------
// Small reusable shapes
// ---------------------------------------------------------------------------

/** A code / terminal panel. Used for analyzer output, plans, postings. */
export const CodePanel: React.FC<{
  lines: React.ReactNode[];
  width?: number;
  title?: string;
  /** Reveal one line every `stagger` frames starting at `delay`. */
  delay?: number;
  stagger?: number;
  style?: React.CSSProperties;
  fontSize?: number;
}> = ({lines, width = 820, title, delay = 0, stagger = 6, style, fontSize = SIZES.code}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        width,
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelEdge}`,
        borderRadius: 10,
        padding: '22px 26px',
        fontFamily: FONT_MONO,
        fontSize,
        lineHeight: 1.6,
        boxShadow: '0 24px 60px rgba(0,0,0,0.45)',
        ...style,
      }}
    >
      {title ? (
        <div
          style={{
            color: COLORS.inkFaint,
            fontSize: fontSize - 6,
            letterSpacing: 2,
            textTransform: 'uppercase',
            marginBottom: 14,
          }}
        >
          {title}
        </div>
      ) : null}
      {lines.map((l, i) => {
        const o = ramp(frame, delay + i * stagger, delay + i * stagger + 8);
        return (
          <div key={i} style={{opacity: o, transform: `translateX(${(1 - o) * -8}px)`}}>
            {l}
          </div>
        );
      })}
    </div>
  );
};

/** A rounded chip, the atom of most diagrams. */
export const Chip: React.FC<{
  children: React.ReactNode;
  color?: string;
  fill?: string;
  w?: number;
  h?: number;
  fontSize?: number;
  mono?: boolean;
  style?: React.CSSProperties;
}> = ({children, color = COLORS.cool, fill, w, h = 58, fontSize = 28, mono, style}) => (
  <div
    style={{
      width: w,
      height: h,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '0 20px',
      borderRadius: 8,
      border: `1.5px solid ${color}`,
      background: fill ?? 'rgba(111,179,224,0.07)',
      color: COLORS.ink,
      fontFamily: mono ? FONT_MONO : FONT_SANS,
      fontSize,
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {children}
  </div>
);
