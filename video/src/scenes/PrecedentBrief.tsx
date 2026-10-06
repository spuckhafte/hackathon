import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, ramp, Scene} from '../components/primitives';
import {BRIEF} from '../data';

/**
 * Act 7 - the stretch feature, and the reason it is not a vector database.
 *
 * Paragraph-level BM25 over the index we already built selects the evidence;
 * the model may only write sentences that carry a paragraph id; a tf-idf cosine
 * verifier then re-reads each sentence against the paragraph it claims and
 * flags the one that is not supported. No embeddings anywhere.
 */

type Sent = {text: string; cite: string; cos: number; ok: boolean};

/** Cosine values here are illustrative geometry; the headline rates are Figures. */
const SENTS: Sent[] = [
  {text: 'Privacy is protected as an intrinsic part of Article 21.', cite: '8142 ¶ 24', cos: 0.61, ok: true},
  {text: 'A restriction must satisfy legality, necessity and proportionality.', cite: '8142 ¶ 31', cos: 0.54, ok: true},
  {text: 'The earlier contrary holding no longer binds later benches.', cite: '7765 ¶ 9', cos: 0.38, ok: true},
  {text: 'Courts have since applied this to biometric identification.', cite: '5533 ¶ 12', cos: 0.11, ok: false},
];

const PARAS = ['8142 ¶24', '8142 ¶31', '7765 ¶9', '5533 ¶12', '3078 ¶47', '2211 ¶3'];

export const PrecedentBrief: React.FC = () => {
  const frame = useCurrentFrame();
  const retrIn = ramp(frame, 8, 60);
  const genIn = ramp(frame, 90, 180);
  const verifyIn = ramp(frame, 210, 330);
  const flagIn = ramp(frame, 330, 380);

  return (
    <Scene act="precedent brief" heading="A brief it cannot write without citing a paragraph.">
      {/* stage 1: paragraph retrieval */}
      <div style={{position: 'absolute', left: SIZES.margin, top: 268, width: 380, opacity: retrIn}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 21, letterSpacing: 2.5, color: COLORS.cool}}>
          1 &#183; PARAGRAPH BM25
        </div>
        <div style={{fontSize: 23, color: COLORS.inkDim, marginTop: 12, lineHeight: 1.5}}>
          over the same positional index &#8212; <Fig f={BRIEF.topDocs} /> documents,{' '}
          <Fig f={BRIEF.maxParas} /> paragraphs, at most two from any one judgment
        </div>
        <div style={{marginTop: 22, display: 'flex', flexDirection: 'column', gap: 9}}>
          {PARAS.map((p, i) => {
            const t = ramp(frame, 20 + i * 9, 20 + i * 9 + 18);
            return (
              <div
                key={p}
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 23,
                  color: COLORS.ink,
                  border: `1px solid ${COLORS.coolSoft}`,
                  background: 'rgba(111,179,224,0.07)',
                  borderRadius: 5,
                  padding: '9px 14px',
                  opacity: t,
                  transform: `translateX(${(1 - t) * -16}px)`,
                }}
              >
                {p}
              </div>
            );
          })}
        </div>
        <div style={{marginTop: 24, fontFamily: FONT_MONO, fontSize: 21, color: COLORS.inkFaint, lineHeight: 1.5}}>
          no embeddings, no vector store &#8212; the retrieval is the system we already
          measured
        </div>
      </div>

      {/* stage 2: generation with forced citations */}
      <div style={{position: 'absolute', left: 640, top: 268, width: 700, opacity: genIn}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 21, letterSpacing: 2.5, color: COLORS.accent}}>
          2 &#183; ONE PARAGRAPH ID PER SENTENCE
        </div>
        <div style={{marginTop: 22, display: 'flex', flexDirection: 'column', gap: 18}}>
          {SENTS.map((s, i) => {
            const t = ramp(frame, 100 + i * 22, 100 + i * 22 + 24);
            const checked = ramp(frame, 220 + i * 28, 220 + i * 28 + 26);
            const bad = !s.ok && flagIn > 0.3;
            return (
              <div
                key={i}
                style={{
                  opacity: t,
                  border: `1.3px solid ${bad ? COLORS.amber : checked > 0.5 ? COLORS.green : COLORS.panelEdge}`,
                  background: bad ? 'rgba(216,162,58,0.1)' : COLORS.panel,
                  borderRadius: 7,
                  padding: '14px 18px',
                }}
              >
                <div style={{fontSize: 26, lineHeight: 1.4}}>
                  {s.text}{' '}
                  <span style={{fontFamily: FONT_MONO, fontSize: 21, color: COLORS.cool}}>
                    [{s.cite}]
                  </span>
                </div>
                <div
                  style={{
                    marginTop: 10,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 14,
                    opacity: checked,
                  }}
                >
                  <div style={{width: 300, height: 8, background: COLORS.panelEdge, borderRadius: 4}}>
                    <div
                      style={{
                        width: `${s.cos * 100 * checked}%`,
                        height: 8,
                        borderRadius: 4,
                        background: bad ? COLORS.amber : COLORS.green,
                      }}
                    />
                  </div>
                  <span style={{fontFamily: FONT_MONO, fontSize: 20, color: bad ? COLORS.amber : COLORS.inkDim}}>
                    cos {s.cos.toFixed(2)}
                  </span>
                  {bad ? (
                    <span style={{fontFamily: FONT_MONO, fontSize: 20, color: COLORS.amber}}>
                      unsupported &#8212; flagged, not deleted
                    </span>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* stage 3: the verifier */}
      <div style={{position: 'absolute', right: SIZES.margin, top: 268, width: 300, opacity: verifyIn}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 21, letterSpacing: 2.5, color: COLORS.green}}>
          3 &#183; COSINE VERIFIER
        </div>
        <div style={{fontSize: 23, color: COLORS.inkDim, marginTop: 12, lineHeight: 1.55}}>
          each sentence is re-tokenised through <span style={{fontFamily: FONT_MONO}}>analyze()</span>{' '}
          and scored against the paragraph it cites. Below{' '}
          <Fig f={BRIEF.cosineThreshold} /> it is marked unsupported.
        </div>
        <div style={{marginTop: 28, fontFamily: FONT_MONO, fontSize: 23, lineHeight: 1.8}}>
          <div>
            sentences <Fig f={BRIEF.sentences} />
          </div>
          <div>
            carrying a cite <Fig f={BRIEF.citedPct} />
          </div>
          <div>
            flagged <Fig f={BRIEF.flaggedPct} />
          </div>
        </div>
      </div>

      <Caption delay={390}>
        The brief is the only part of NyayaTrace that uses a language model, and it is wrapped
        so that the model cannot assert anything the index does not already hold.
      </Caption>
    </Scene>
  );
};
