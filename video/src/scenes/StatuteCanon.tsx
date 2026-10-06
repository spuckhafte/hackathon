import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, ramp, Scene} from '../components/primitives';
import {BNS_DATE, BNS_TOKEN, STATUTE_TOKEN} from '../data';

/**
 * Act 4 - the analyzer, Sumedha's component.
 *
 * One clause of real judgment text walks through four stages. Tokens visibly
 * change, disappear or fuse at each stage, so the viewer sees that document and
 * query go through the *same* function. The band underneath is the crosswalk:
 * at query time the old Penal Code token also fires the Sanhita token, because
 * the statute was replaced in the middle of our corpus.
 */

const RAW = 'convicted u/s 302 IPC read with S. 34';

type Stage = {name: string; note: string; tokens: {t: string; kind?: string}[]};

const STAGES: Stage[] = [
  {
    name: 'fold + tokenise',
    note: 'case folding and punctuation, no stemming yet',
    tokens: [
      {t: 'convicted'}, {t: 'u/s'}, {t: '302'}, {t: 'ipc'},
      {t: 'read'}, {t: 'with'}, {t: 's'}, {t: '34'},
    ],
  },
  {
    name: 'canonicalise statutes',
    note: 'a grammar over citation forms, not a word list',
    tokens: [
      {t: 'convicted'}, {t: STATUTE_TOKEN, kind: 'statute'}, {t: 'read'},
      {t: 'with'}, {t: 'ipc§34', kind: 'statute'},
    ],
  },
  {
    name: 'legal stop list',
    note: 'drops the words that appear in every judgment',
    tokens: [
      {t: 'convicted'}, {t: STATUTE_TOKEN, kind: 'statute'},
      {t: 'ipc§34', kind: 'statute'},
    ],
  },
  {
    name: 'Porter stem',
    note: 'statute tokens are exempt - never stem an identifier',
    tokens: [
      {t: 'convict', kind: 'stem'}, {t: STATUTE_TOKEN, kind: 'statute'},
      {t: 'ipc§34', kind: 'statute'},
    ],
  },
];

const colorOf = (kind?: string) =>
  kind === 'statute' ? COLORS.accent : kind === 'stem' ? COLORS.cool : COLORS.ink;

const TOP = 316;
const ROW = 104;
const LABEL_W = 300;

export const StatuteCanon: React.FC = () => {
  const frame = useCurrentFrame();
  const rawIn = ramp(frame, 6, 30);
  const crosswalk = ramp(frame, 300, 360);

  return (
    <Scene act="the pipeline · analyzer" heading="Documents and queries go through one function.">
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin + LABEL_W + 42,
          top: 248,
          fontFamily: FONT_MONO,
          fontSize: 29,
          color: COLORS.inkDim,
          opacity: rawIn,
        }}
      >
        &#8220;{RAW}&#8221;
      </div>

      {STAGES.map((s, i) => {
        const t = ramp(frame, 40 + i * 46, 40 + i * 46 + 24);
        const y = TOP + i * ROW;
        return (
          <div
            key={s.name}
            style={{
              position: 'absolute',
              left: SIZES.margin,
              top: y,
              right: SIZES.margin,
              opacity: t,
              transform: `translateY(${(1 - t) * 14}px)`,
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', gap: 20}}>
              <div
                style={{
                  width: LABEL_W,
                  fontFamily: FONT_MONO,
                  fontSize: 24,
                  color: COLORS.cool,
                  textAlign: 'right',
                }}
              >
                {s.name}
              </div>
              <div style={{width: 1, height: 52, background: COLORS.panelEdge}} />
              <div style={{display: 'flex', gap: 9}}>
                {s.tokens.map((tk, j) => {
                  const te = ramp(frame, 40 + i * 46 + j * 4, 40 + i * 46 + j * 4 + 14);
                  return (
                    <span
                      key={j}
                      style={{
                        fontFamily: FONT_MONO,
                        fontSize: 27,
                        color: colorOf(tk.kind),
                        border: `1.2px solid ${tk.kind ? colorOf(tk.kind) : COLORS.panelEdge}`,
                        background: tk.kind ? `${colorOf(tk.kind)}18` : 'rgba(21,27,36,0.6)',
                        borderRadius: 5,
                        padding: '6px 13px',
                        opacity: te,
                        transform: `scale(${0.9 + 0.1 * te})`,
                      }}
                    >
                      {tk.t}
                    </span>
                  );
                })}
              </div>
            </div>
            <div
              style={{
                marginLeft: LABEL_W + 42,
                marginTop: 7,
                fontSize: 21,
                color: COLORS.inkFaint,
              }}
            >
              {s.note}
            </div>
          </div>
        );
      })}

      {/* IPC <-> BNS crosswalk, as a band under the stages */}
      <div
        style={{
          position: 'absolute',
          left: SIZES.margin,
          right: SIZES.margin,
          top: TOP + STAGES.length * ROW + 26,
          opacity: crosswalk,
          transform: `translateY(${(1 - crosswalk) * 16}px)`,
          borderTop: `1px solid ${COLORS.panelEdge}`,
          paddingTop: 26,
          display: 'flex',
          alignItems: 'center',
          gap: 44,
        }}
      >
        <div
          style={{
            width: LABEL_W,
            textAlign: 'right',
            fontFamily: FONT_MONO,
            fontSize: 21,
            letterSpacing: 2.5,
            color: COLORS.inkFaint,
          }}
        >
          AT QUERY TIME
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 16}}>
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: 32,
              color: COLORS.accent,
              border: `1.5px solid ${COLORS.accent}`,
              borderRadius: 6,
              padding: '9px 15px',
            }}
          >
            {STATUTE_TOKEN}
          </span>
          <span style={{color: COLORS.inkDim, fontSize: 32}}>&#8644;</span>
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: 32,
              color: COLORS.green,
              border: `1.5px solid ${COLORS.green}`,
              borderRadius: 6,
              padding: '9px 15px',
            }}
          >
            {BNS_TOKEN}
          </span>
        </div>
        <div style={{flex: 1, fontSize: 25, color: COLORS.inkDim, lineHeight: 1.45}}>
          The Penal Code was replaced by the Bharatiya Nyaya Sanhita on{' '}
          <span style={{color: COLORS.ink, fontFamily: FONT_MONO}}>{BNS_DATE}</span>, and the
          section numbers moved. A corpus that spans the change needs that OR to happen in the
          parser, not in the user&#8217;s head.
        </div>
      </div>

      <Caption delay={380}>
        Canonicalisation happens once at build time, so it costs nothing at query time, and the
        crosswalk is a query-side expansion we can switch off for the ablation.
      </Caption>
    </Scene>
  );
};
