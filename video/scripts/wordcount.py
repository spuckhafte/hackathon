"""Check every voice-over passage against the slot it has to fit in.

    cd video
    python scripts/wordcount.py script/voiceover.md src/timeline.ts

Prints a word count, a speech duration at 150 wpm and a fill percentage for all
24 passages, then flags any that exceed 90% of its slot. Run it after editing
either the script or the timeline: a passage that overruns cannot be fixed in
the edit, because the slot lengths are what set the running time.


A passage is the block of '> ' lines under a '### ' heading. Lines that are
editor notes rather than speech (they start with a warning glyph) are skipped.
Cue markers like [ANIM: id ->] are not spoken, so they are stripped.
"""
import json, pathlib, re, sys

vo = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
tl = pathlib.Path(sys.argv[2]).read_text(encoding='utf-8')

# slot seconds straight out of timeline.ts
slots = {}
for m in re.finditer(r"kind:\s*'(explain|screen|cam)',\s*\n\s*id:\s*'([^']+)',\s*\n\s*seconds:\s*(\d+)", tl):
    kind, sid, sec = m.group(1), m.group(2), int(m.group(3))
    slots[('cam-' + sid) if kind == 'cam' else sid] = (kind, sec)

WPS = 2.5  # 150 words per minute

blocks, cur, head = [], [], None
for line in vo.splitlines():
    if line.startswith('### '):
        if head:
            blocks.append((head, cur))
        head, cur = line, []
    elif head is not None and line.startswith('>'):
        cur.append(line[1:].strip())
if head:
    blocks.append((head, cur))

rows, bad = [], []
for head, lines in blocks:
    hid = re.search(r'`(?:\[(?:SCREEN|CAM): )?([a-z0-9-]+)\]?`', head)
    if not hid:
        continue
    sid = hid.group(1)
    if sid not in slots and ('cam-' + sid) in slots:
        sid = 'cam-' + sid
    if sid not in slots:
        bad.append('no slot for ' + sid)
        continue
    kind, sec = slots[sid]
    text = ' '.join(l for l in lines if l and not l.startswith('⚠'))
    text = re.sub(r'\*\*\[(?:ANIM|CAM)[^\]]*\]\*\*', ' ', text)   # cue markers
    text = re.sub(r'\*\*|\*|`|_', ' ', text)
    words = [w for w in text.split() if re.search(r'[0-9A-Za-zÀ-ɏ]', w)]
    n = len(words)
    secs = n / WPS
    fill = secs / sec
    rows.append((sid, kind, sec, n, secs, fill))
    if fill > 0.90:
        bad.append('OVER 90%%: %-18s %2ds  %3d words  %.1fs  %.0f%%  (max %d words)'
                   % (sid, sec, n, secs, 100 * fill, int(sec * WPS * 0.9)))

print('%-20s %-8s %5s %7s %8s %6s' % ('passage', 'kind', 'slot', 'words', 'speech', 'fill'))
for sid, kind, sec, n, secs, fill in rows:
    flag = '  <<<' if fill > 0.90 else ''
    print('%-20s %-8s %4ds %6d %7.1fs %5.0f%%%s' % (sid, kind, sec, n, secs, 100 * fill, flag))

tot_words = sum(r[3] for r in rows)
tot_slot = sum(r[2] for r in rows)
print()
print('passages        : %d of %d segments' % (len(rows), len(slots)))
print('total words     : %d' % tot_words)
print('total speech    : %.0fs (%d:%02d)' % (tot_words / WPS, (tot_words / WPS) // 60, (tot_words / WPS) % 60))
print('total film      : %ds (%d:%02d)' % (tot_slot, tot_slot // 60, tot_slot % 60))
print()
if bad:
    print('PROBLEMS:')
    for b in bad:
        print('  ' + b)
else:
    print('all passages are inside 90% of their slot')

# by speaker
spk = {}
for head, _ in blocks:
    m = re.search(r'\*\*([A-Z][a-z]+)\*\*', head)
    hid = re.search(r'`(?:\[(?:SCREEN|CAM): )?([a-z0-9-]+)\]?`', head)
    if m and hid:
        sid = hid.group(1)
        if sid not in slots and ('cam-' + sid) in slots:
            sid = 'cam-' + sid
        spk.setdefault(m.group(1), []).append(sid)
print()
by = {k: sum(dict((r[0], r[3]) for r in rows).get(i, 0) for i in v) for k, v in spk.items()}
secs_by = {k: sum(dict((r[0], r[2]) for r in rows).get(i, 0) for i in v) for k, v in spk.items()}
t = sum(by.values()) or 1
for k in sorted(by):
    print('%-9s %4d words (%4.1f%%)   %3ds of film (%4.1f%%)'
          % (k, by[k], 100 * by[k] / t, secs_by[k], 100 * secs_by[k] / tot_slot))
