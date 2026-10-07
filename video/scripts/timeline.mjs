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
const n = (k) => segs.filter((s) => s.kind === k).length;
const pct = (v) => `${((100 * v) / at).toFixed(1)}%`;
// title + outro are animation, but they are chrome rather than concept inserts
const CHROME = ['title', 'outro'];
const chrome = segs.filter((s) => CHROME.includes(s.id)).reduce((a, s) => a + s.seconds, 0);
const inserts = byKind('explain') - chrome;

console.log('');
console.log(`total        ${mmss(at)}  (${at}s, ${at * 30} frames at 30fps)`);
console.log(`  screen     ${mmss(byKind('screen'))}  ${pct(byKind('screen')).padStart(6)}  ${n('screen')} clips`);
console.log(`  animation  ${mmss(byKind('explain'))}  ${pct(byKind('explain')).padStart(6)}  ${n('explain')} scenes`);
console.log(`    inserts  ${mmss(inserts)}  ${pct(inserts).padStart(6)}  ${n('explain') - CHROME.length} concept inserts`);
console.log(`    chrome   ${mmss(chrome)}  ${pct(chrome).padStart(6)}  title + outro`);
console.log(`  webcam     ${mmss(byKind('cam'))}  ${pct(byKind('cam')).padStart(6)}  ${n('cam')} clips`);

const long = segs.filter((s) => s.kind === 'explain' && !CHROME.includes(s.id) && (s.seconds < 12 || s.seconds > 25));
if (long.length) console.log(`\n!  insert outside the 12-25s window: ${long.map((s) => s.id).join(', ')}`);
const screenShare = (100 * byKind('screen')) / at;
if (screenShare < 55 || screenShare > 65) console.log(`\n!  screen share ${screenShare.toFixed(1)}% is outside the 55-65% target`);
if (at < 300 || at > 480) console.log('\n!! outside the 5:00-8:00 limit');
else if (at < 435 || at > 470) console.log('\n!  outside the 7:15-7:50 target (still legal)');
