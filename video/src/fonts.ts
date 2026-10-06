/**
 * Fonts are loaded best-effort: we never `await waitUntilDone()`, so a machine
 * with no network still renders (the CSS fallback stack in theme.ts takes over)
 * instead of hanging on `delayRender`.
 */
import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import {loadFont as loadSerif} from '@remotion/google-fonts/SourceSerif4';

let sans = 'Inter';
let mono = 'JetBrains Mono';
let serif = 'Source Serif 4';

try {
  sans = loadInter('normal', {weights: ['300', '400', '600', '700'], subsets: ['latin']}).fontFamily;
  mono = loadMono('normal', {weights: ['400', '500', '700'], subsets: ['latin']}).fontFamily;
  serif = loadSerif('normal', {weights: ['400', '600'], subsets: ['latin']}).fontFamily;
} catch {
  // keep the plain family names; the fallback stack handles it
}

export const FONT_SANS = `${sans}, system-ui, -apple-system, sans-serif`;
export const FONT_MONO = `${mono}, "Cascadia Mono", Consolas, monospace`;
export const FONT_SERIF = `${serif}, Georgia, serif`;
