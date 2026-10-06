import React from 'react';
import {useCurrentFrame} from 'remotion';
import {COLORS, SIZES} from '../theme';
import {FONT_MONO} from '../fonts';
import {Caption, Fig, MathRow, Op, ramp, Scene, Sub, Var} from '../components/primitives';
import {CONFIG} from '../data';

/**
 * Act 4 - the scorers, Sumedha's component.
 *
 * Left: the saturation curve. A straight line (raw tf) runs off the top while
 * the BM25F curve bends over; the twentieth occurrence of a word is drawn
 * almost exactly on top of the nineteenth. Right: the zones are combined
 * *before* saturation, which is what makes it BM25F rather than BM25 on a
 * concatenated document.
 */

const K1 = CONFIG.bm25fK1;
const K1V = 1.2; // geometry only; the printed value comes from CONFIG.bm25fK1

const PLOT_X = 230;
const PLOT_Y = 760;
const PLOT_W = 700;
const PLOT_H = 420;
const TF_MAX = 20;

const satur = (tf: number) => (tf * (K1V + 1)) / (tf + K1V);
const Y_MAX = satur(TF_MAX) * 1.1;
/** tf at which the straight raw-tf line leaves the top of the plot. */
const RAW_EXIT = Y_MAX;

const px = (tf: number) => PLOT_X + (tf / TF_MAX) * PLOT_W;
const py = (v: number) => PLOT_Y - (v / Y_MAX) * PLOT_H;

export const Bm25f: React.FC = () => {
  const frame = useCurrentFrame();
  const axesIn = ramp(frame, 8, 40);
  const lineIn = ramp(frame, 50, 130);
  const curveIn = ramp(frame, 100, 210);
  const zonesIn = ramp(frame, 230, 300);

  const curvePts = new Array(121).fill(0).map((_, i) => {
    const tf = (i / 120) * TF_MAX;
    return [px(tf), py(satur(tf))] as const;
  });
  const shownCurve = curvePts.slice(0, Math.max(2, Math.ceil(curvePts.length * curveIn)));

  const zoneEntries = Object.entries(CONFIG.zoneWeights).filter(([, w]) => w > 0);
  const totalW = zoneEntries.reduce((a, [, w]) => a + w, 0);

  return (
    <Scene act="the pipeline · ranking" heading="The twentieth mention is not twenty times the evidence.">
      <svg width={SIZES.width} height={SIZES.height} style={{position: 'absolute', inset: 0}}>
        {/* axes */}
        <g opacity={axesIn}>
          <line x1={PLOT_X} y1={PLOT_Y} x2={PLOT_X + PLOT_W + 40} y2={PLOT_Y} stroke={COLORS.inkFaint} strokeWidth={1.5} />
          <line x1={PLOT_X} y1={PLOT_Y} x2={PLOT_X} y2={PLOT_Y - PLOT_H - 40} stroke={COLORS.inkFaint} strokeWidth={1.5} />
          {[5, 10, 15, 20].map((t) => (
            <g key={t}>
              <line x1={px(t)} y1={PLOT_Y} x2={px(t)} y2={PLOT_Y + 8} stroke={COLORS.inkFaint} />
              <text x={px(t)} y={PLOT_Y + 34} textAnchor="middle" fontFamily={FONT_MONO} fontSize={21} fill={COLORS.inkFaint}>
                {t}
              </text>
            </g>
          ))}
          <text x={PLOT_X + PLOT_W / 2} y={PLOT_Y + 74} textAnchor="middle" fontFamily={FONT_MONO} fontSize={22} fill={COLORS.inkDim}>
            term frequency in the document
          </text>
          <text x={PLOT_X - 22} y={PLOT_Y - PLOT_H - 20} textAnchor="end" fontFamily={FONT_MONO} fontSize={22} fill={COLORS.inkDim}>
            weight
          </text>
        </g>

        {/* raw tf: leaves the top of the chart almost immediately */}
        <g opacity={lineIn}>
          <line
            x1={px(0)}
            y1={py(0)}
            x2={px(RAW_EXIT * lineIn)}
            y2={py(Y_MAX * lineIn)}
            stroke={COLORS.inkFaint}
            strokeWidth={2}
            strokeDasharray="8 7"
          />
          <text
            x={px(RAW_EXIT) + 16}
            y={py(Y_MAX) + 26}
            fontFamily={FONT_MONO}
            fontSize={23}
            fill={COLORS.inkFaint}
          >
            raw tf, off the top
          </text>
        </g>

        {/* saturating curve */}
        <polyline
          points={shownCurve.map(([x, y]) => `${x},${y}`).join(' ')}
          fill="none"
          stroke={COLORS.accent}
          strokeWidth={3.5}
        />
        {/* k1 marker */}
        <g opacity={ramp(frame, 180, 220)}>
          <line
            x1={px(K1V)}
            y1={PLOT_Y}
            x2={px(K1V)}
            y2={py(satur(K1V))}
            stroke={COLORS.cool}
            strokeWidth={1.5}
            strokeDasharray="5 6"
          />
          <circle cx={px(K1V)} cy={py(satur(K1V))} r={6} fill={COLORS.cool} />
          <text x={px(K1V) + 14} y={py(satur(K1V)) - 14} fontFamily={FONT_MONO} fontSize={22} fill={COLORS.cool}>
            tf = k&#8321;: half of the ceiling
          </text>
        </g>
      </svg>

      {/* the formula */}
      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 268,
          width: 700,
          opacity: ramp(frame, 60, 110),
        }}
      >
        <MathRow size={46} style={{justifyContent: 'flex-start'}}>
          <Var color={COLORS.accent}>
            tf&#773;<Sub>d</Sub>
          </Var>
          <Op>=</Op>
          <span style={{fontSize: '1.1em'}}>&#931;</span>
          <Sub>z</Sub>
          <Op />
          <Var color={COLORS.cool}>w</Var>
          <Sub>z</Sub>
          <Op>&#183;</Op>
          <span style={{display: 'inline-flex', flexDirection: 'column', alignItems: 'center', fontSize: '0.8em'}}>
            <span>tf<Sub>z,d</Sub></span>
            <span style={{height: 2, alignSelf: 'stretch', background: COLORS.inkFaint, margin: '5px 0'}} />
            <span>
              1 + b<Sub>z</Sub>(len<Sub>z,d</Sub>/len&#773;<Sub>z</Sub> &#8722; 1)
            </span>
          </span>
        </MathRow>
        <div style={{height: 26}} />
        <MathRow size={46} style={{justifyContent: 'flex-start'}}>
          <Var>score</Var>
          <Op>=</Op>
          <span style={{fontSize: '1.1em'}}>&#931;</span>
          <Sub>t&#8712;q</Sub>
          <Op />
          <Var color={COLORS.violet}>idf</Var>
          <Sub>t</Sub>
          <Op>&#183;</Op>
          <span style={{display: 'inline-flex', flexDirection: 'column', alignItems: 'center', fontSize: '0.84em'}}>
            <span>
              tf&#773;<Sub>d</Sub>
            </span>
            <span style={{height: 2, alignSelf: 'stretch', background: COLORS.inkFaint, margin: '5px 0'}} />
            <span>
              k&#8321; + tf&#773;<Sub>d</Sub>
            </span>
          </span>
        </MathRow>
        <div style={{marginTop: 30, fontSize: 25, color: COLORS.inkDim, lineHeight: 1.5}}>
          The zones are summed <em>before</em> saturation, each with its own length
          normaliser b<sub>z</sub>: saturating per zone and then adding would let a long body
          out-shout a title. k&#8321; = <Fig f={K1} />.
        </div>
      </div>

      {/* zone contribution bar */}
      <div
        style={{
          position: 'absolute',
          right: SIZES.margin,
          top: 748,
          width: 700,
          opacity: zonesIn,
        }}
      >
        <div style={{fontFamily: FONT_MONO, fontSize: 21, letterSpacing: 2.5, color: COLORS.inkFaint, marginBottom: 14}}>
          w&#8338; &#8212; ONE TERM, SIX PLACES IT COULD BE
        </div>
        <div style={{display: 'flex', height: 46, borderRadius: 5, overflow: 'hidden'}}>
          {zoneEntries.map(([z, w], i) => (
            <div
              key={z}
              title={z}
              style={{
                width: `${(w / totalW) * 100 * zonesIn}%`,
                background: [COLORS.accent, COLORS.cool, COLORS.violet, COLORS.green, COLORS.coolSoft][i % 5],
                opacity: 0.8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: FONT_MONO,
                fontSize: 20,
                color: COLORS.bgDeep,
                fontWeight: 600,
              }}
            >
              {w / totalW > 0.1 ? z : ''}
            </div>
          ))}
        </div>
        <div style={{marginTop: 12, fontFamily: FONT_MONO, fontSize: 20, color: COLORS.inkFaint, lineHeight: 1.45}}>
          the sliver is <span style={{color: COLORS.inkDim}}>parties</span>; bench is weight 0,
          a filter, so scoring it would be double counting
        </div>
      </div>

      <Caption delay={320}>
        Behind all of this the plain lnc.ltc cosine baseline is still wired in, because the
        ablation has to be a flag rather than a rewrite.
      </Caption>
    </Scene>
  );
};
