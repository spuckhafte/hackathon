import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {CONFIG} from '../data';

/**
 * Act 4, close - the three streams meet.
 *
 * Each term of the net score flies in from the side of the frame where its
 * scene lived, so the formula is assembled out of the film rather than printed.
 * Then a worked decomposition: the same numbers the "why this result" panel
 * shows in the live demo.
 */

type Part = {
  key: string;
  label: string;
  from: 'left' | 'right' | 'bottom';
  color: string;
  weight: string;
  provenance: string;
};

const PARTS: Part[] = [
  {
    key: 'text',
    label: 'text̂',
    from: 'left',
    color: COLORS.cool,
    weight: CONFIG.alpha.value,
    provenance: 'BM25F over six zones, min-max normalised within the candidate set',
  },
  {
    key: 'g',
    label: 'g(d)',
    from: 'right',
    color: COLORS.violet,
    weight: `1 − ${CONFIG.alpha.value}`,
    provenance: 'treatment-weighted PageRank × bench size × status penalty',
  },
  {
    key: 'prox',
    label: 'prox',
    from: 'bottom',
    color: COLORS.accent,
    weight: CONFIG.beta.value,
    provenance: 'tightest window the operators found, inverted',
  },
];

export const NetScore: React.FC = () => {
  const frame = useCurrentFrame();
  const inAt = (i: number) => ramp(frame, 20 + i * 34, 20 + i * 34 + 30);
  const formulaIn = ramp(frame, 150, 200);
  const breakdownIn = ramp(frame, 230, 300);

  return (
    <Scene act="the pipeline · net score" heading="One number, and you can take it apart.">
      {/* the three incoming terms */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          right: SIZES.margin,
          top: 300,
          display: 'flex',
          justifyContent: 'space-between',
          gap: 40,
        }}
      >
        {PARTS.map((p, i) => {
          const t = inAt(i);
          const dx = p.from === 'left' ? -110 : p.from === 'right' ? 110 : 0;
          const dy = p.from === 'bottom' ? 130 : 0;
          return (
            <div
              key={p.key}
              style={{
                flex: 1,
                opacity: t,
                transform: `translate(${dx * (1 - t)}px, ${dy * (1 - t)}px)`,
                border: `1.5px solid ${p.color}`,
                background: `${p.color}12`,
                borderRadius: 10,
                padding: '22px 24px',
              }}
            >
              <div style={{display: 'flex', alignItems: 'baseline', gap: 16}}>
                <span style={{fontFamily: FONT_MONO, fontSize: 44, color: p.color}}>
                  {p.label}
                </span>
                <span style={{fontFamily: FONT_MONO, fontSize: 24, color: COLORS.inkDim}}>
                  &#215; {p.weight}
                </span>
              </div>
              <div style={{marginTop: 14, fontSize: 23, color: COLORS.inkDim, lineHeight: 1.5}}>
                {p.provenance}
              </div>
            </div>
          );
        })}
      </div>

      {/* the formula */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 560,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 14,
          fontFamily: FONT_MONO,
          fontSize: 54,
          opacity: formulaIn,
          transform: `scale(${0.94 + 0.06 * formulaIn})`,
        }}
      >
        <span style={{color: COLORS.ink}}>net</span>
        <span style={{color: COLORS.inkDim}}>=</span>
        <span style={{color: COLORS.cool}}>&#945;</span>
        <span style={{color: COLORS.inkDim}}>&#183;</span>
        <span style={{color: COLORS.cool}}>text&#770;</span>
        <span style={{color: COLORS.inkDim}}>+</span>
        <span style={{color: COLORS.violet}}>(1&#8722;&#945;)</span>
        <span style={{color: COLORS.inkDim}}>&#183;</span>
        <span style={{color: COLORS.violet}}>g(d)</span>
        <span style={{color: COLORS.inkDim}}>+</span>
        <span style={{color: COLORS.accent}}>&#946;</span>
        <span style={{color: COLORS.inkDim}}>&#183;</span>
        <span style={{color: COLORS.accent}}>prox</span>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 646,
          textAlign: 'center',
          fontFamily: FONT_MONO,
          fontSize: 28,
          color: COLORS.inkDim,
          opacity: formulaIn,
        }}
      >
        &#945; = <Fig f={CONFIG.alpha} /> &nbsp;&nbsp; &#946; = <Fig f={CONFIG.beta} />
      </div>

      {/* why-this-result preview */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 736,
          display: 'flex',
          justifyContent: 'center',
          opacity: breakdownIn,
        }}
      >
        <div
          style={{
            width: 1180,
            border: `1px solid ${COLORS.panelEdge}`,
            background: COLORS.panel,
            borderRadius: 10,
            padding: '20px 26px',
            fontFamily: FONT_MONO,
            fontSize: 24,
            color: COLORS.inkDim,
          }}
        >
          <div style={{color: COLORS.inkFaint, fontSize: 20, letterSpacing: 2.5, marginBottom: 14}}>
            EVERY RESULT SHIPS ITS OWN DECOMPOSITION
          </div>
          <div style={{display: 'flex', gap: 34, flexWrap: 'wrap', color: COLORS.ink}}>
            <span>text&#770;</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>per-zone contribution</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>top terms by idf&#183;tf</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>prox window</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>pagerank&#770;</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>bench&#770;</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span style={{color: COLORS.alarm}}>penalty</span>
            <span style={{color: COLORS.inkFaint}}>|</span>
            <span>net</span>
          </div>
          <div style={{marginTop: 12, fontSize: 21, color: COLORS.inkFaint}}>
            ScoreExplanation in nyaya/contracts.py &#8212; the panel is a dataclass, not a
            text template, so it cannot drift from the scorer
          </div>
        </div>
      </div>

      <Caption delay={330}>
        Switch &#945; to one and the authority layer is gone; switch the scorer and you are
        back on the cosine baseline. That is how the six systems in the evaluation are built
        from one code path.
      </Caption>
    </Scene>
  );
};
