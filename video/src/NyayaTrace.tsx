import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {CamSlot, ScreenSlot} from './components/Slots';
import {LAYOUT} from './timeline';
import {COLORS} from './theme';

import {Title} from './scenes/Title';
import {ThreeWays} from './scenes/ThreeWays';
import {Anatomy} from './scenes/Anatomy';
import {OverruledTrap} from './scenes/OverruledTrap';
import {PipelineMap} from './scenes/PipelineMap';
import {StatuteCanon} from './scenes/StatuteCanon';
import {ZonedPostings} from './scenes/ZonedPostings';
import {ProximityWalk} from './scenes/ProximityWalk';
import {BooleanPlan} from './scenes/BooleanPlan';
import {Bm25f} from './scenes/Bm25f';
import {CitationGraph} from './scenes/CitationGraph';
import {NetScore} from './scenes/NetScore';
import {EvalLadder} from './scenes/EvalLadder';
import {OperatorAndSpeed} from './scenes/OperatorAndSpeed';
import {Limitation} from './scenes/Limitation';
import {PrecedentBrief} from './scenes/PrecedentBrief';
import {Outro} from './scenes/Outro';

/** id -> animated scene. Every `explain` segment in the timeline needs an entry. */
const SCENES: Record<string, React.FC> = {
  title: Title,
  'three-ways': ThreeWays,
  anatomy: Anatomy,
  'overruled-trap': OverruledTrap,
  'pipeline-map': PipelineMap,
  'statute-canon': StatuteCanon,
  'zoned-postings': ZonedPostings,
  'proximity-walk': ProximityWalk,
  'boolean-plan': BooleanPlan,
  bm25f: Bm25f,
  'citation-graph': CitationGraph,
  'net-score': NetScore,
  'eval-ladder': EvalLadder,
  'operator-speed': OperatorAndSpeed,
  limitation: Limitation,
  'precedent-brief': PrecedentBrief,
  outro: Outro,
};

const Missing: React.FC<{id: string}> = ({id}) => (
  <AbsoluteFill
    style={{
      background: COLORS.bgDeep,
      color: COLORS.alarm,
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: 48,
      fontFamily: 'monospace',
    }}
  >
    no scene registered for &quot;{id}&quot;
  </AbsoluteFill>
);

/**
 * An animated scene lingers this many frames past its slot so that its fade-out
 * happens *under* the next segment: a dissolve rather than a dip to black.
 * Slots keep their exact length, since footage must not run past its cut.
 */
const OVERLAP = 12;

export const NyayaTrace: React.FC = () => (
  <AbsoluteFill style={{backgroundColor: COLORS.bg}}>
    {LAYOUT.map((seg, i) => {
      const last = i === LAYOUT.length - 1;
      const common = {
        key: seg.id,
        from: seg.from,
        durationInFrames:
          seg.durationInFrames + (seg.kind === 'explain' && !last ? OVERLAP : 0),
        name: `${seg.kind}:${seg.id}`,
      };
      if (seg.kind === 'explain') {
        const Comp = SCENES[seg.id];
        return (
          <Sequence {...common}>
            {Comp ? <Comp /> : <Missing id={seg.id} />}
          </Sequence>
        );
      }
      if (seg.kind === 'screen') {
        return (
          <Sequence {...common}>
            <ScreenSlot
              name={seg.id}
              seconds={seg.seconds}
              capture={seg.capture}
              commands={seg.commands}
              mustShow={seg.mustShow}
            />
          </Sequence>
        );
      }
      return (
        <Sequence {...common}>
          <CamSlot
            person={seg.person}
            slug={seg.id}
            seconds={seg.seconds}
            topic={seg.topic}
            beats={seg.beats}
          />
        </Sequence>
      );
    })}
  </AbsoluteFill>
);

/**
 * One segment, alone, starting at frame 0 - used by the per-segment
 * compositions in Root.tsx so a scene can be stilled or previewed without
 * scrubbing through the whole film.
 */
export const SegmentOnly: React.FC<{id: string}> = ({id}) => {
  const seg = LAYOUT.find((s) => s.id === id);
  if (!seg) return <Missing id={id} />;
  if (seg.kind === 'explain') {
    const Comp = SCENES[seg.id];
    return (
      <AbsoluteFill style={{backgroundColor: COLORS.bg}}>
        {Comp ? <Comp /> : <Missing id={seg.id} />}
      </AbsoluteFill>
    );
  }
  if (seg.kind === 'screen') {
    return (
      <AbsoluteFill style={{backgroundColor: COLORS.bg}}>
        <ScreenSlot
          name={seg.id}
          seconds={seg.seconds}
          capture={seg.capture}
          commands={seg.commands}
          mustShow={seg.mustShow}
        />
      </AbsoluteFill>
    );
  }
  return (
    <AbsoluteFill style={{backgroundColor: COLORS.bg}}>
      <CamSlot
        person={seg.person}
        slug={seg.id}
        seconds={seg.seconds}
        topic={seg.topic}
        beats={seg.beats}
      />
    </AbsoluteFill>
  );
};
