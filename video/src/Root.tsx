import React from 'react';
import {Composition} from 'remotion';
import {NyayaTrace, SegmentOnly} from './NyayaTrace';
import {FPS, LAYOUT, TOTAL_FRAMES} from './timeline';
import {SIZES} from './theme';

/**
 * One main composition, plus a `seg-<id>` composition for every segment so a
 * single animation or slot can be previewed, stilled or re-rendered on its own.
 * The per-segment compositions reuse the real durations, so what you check in
 * isolation is exactly what lands in the edit.
 */
export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="NyayaTrace"
      component={NyayaTrace}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={SIZES.width}
      height={SIZES.height}
    />
    {LAYOUT.map((s) => (
      <Composition
        key={s.id}
        id={`seg-${s.id}`}
        component={SegmentOnly}
        defaultProps={{id: s.id}}
        durationInFrames={s.durationInFrames}
        fps={FPS}
        width={SIZES.width}
        height={SIZES.height}
      />
    ))}
  </>
);
