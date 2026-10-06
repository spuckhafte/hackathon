/**
 * Prints the edit as a table, straight from src/timeline.ts, so the script and
 * the speaker map can be checked against the actual composition length.
 *   node scripts/timeline.mjs
 */
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, '..', 'src', 'timeline.ts'), 'utf8');

const segs = [];
const re = /kind:\s*'(explain|screen|cam)',\s*\n\s*id:\s*'([^']+)',\s*\n\s*seconds:\s*(\d+)/g;
let m;
while ((m = re.exec(src))) segs.push({kind: m[1], id: m[2], seconds: Number(m[3])});

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

let at = 0;
console.log('  #  start   end    len  kind     id');
segs.forEach((s, i) => {
  const from = at;
  at += s.seconds;
  console.log(
    `${String(i + 1).padStart(3)}  ${mmss(from).padStart(5)}  ${mmss(at).padStart(5)}  ` +
      `${String(s.seconds).padStart(3)}s  ${s.kind.padEnd(8)} ${s.id}`
  );
});

const byKind = (k) => segs.filter((s) => s.kind === k).reduce((a, s) => a + s.seconds, 0);
console.log('');
console.log(`total        ${mmss(at)}  (${at}s, ${at * 30} frames at 30fps)`);
console.log(`  animation  ${mmss(byKind('explain'))}  ${segs.filter((s) => s.kind === 'explain').length} segments`);
console.log(`  screen     ${mmss(byKind('screen'))}  ${segs.filter((s) => s.kind === 'screen').length} slots`);
console.log(`  webcam     ${mmss(byKind('cam'))}  ${segs.filter((s) => s.kind === 'cam').length} slots`);
if (at < 300 || at > 480) console.log('\n!! outside the 5:00-8:00 limit');
else if (at < 420 || at > 450) console.log('\n!  outside the 7:00-7:30 target (still legal)');
