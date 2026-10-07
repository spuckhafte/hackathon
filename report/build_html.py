"""Build report/report.html from report/report.md.

The HTML used to be maintained by hand beside the Markdown, which meant the two
could disagree about a measured number -- exactly the failure this report spends
a section warning about. Now there is one source of prose (`report.md`) and three
fixed fragments that Markdown cannot express:

    _tpl_head.html   doctype, print CSS, @page rules, masthead
    _tpl_fig1.html   Figure 1, the pipeline diagram, as inline SVG
    _tpl_fig2.html   Figure 2, the ablation ladder, as inline SVG

`report.md` marks where the figures go with a line of its own:

    **[FIGURE 1 - pipeline diagram]**
    **[FIGURE 2 - ...]**

Everything else is converted straight through, including the tables.

    python report/build_html.py        # -> report/report.html
    python report/build_pdf.py         # -> report/NyayaTrace-report.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "report.md"
OUT = HERE / "report.html"


def _fragment(name: str) -> str:
    p = HERE / name
    if not p.exists():
        sys.exit(f"missing fragment {p} - it holds hand-written SVG, do not delete it")
    return p.read_text(encoding="utf-8")


def build() -> str:
    try:
        import markdown
    except ImportError:
        sys.exit("pip install markdown")

    md = MD.read_text(encoding="utf-8")

    # The first blockquote in the Markdown becomes the framed notice box.
    notice = ""
    m = re.search(r"^((?:> .*\n)+)", md, re.M)
    if m:
        raw = " ".join(line[1:].strip() for line in m.group(1).splitlines())
        notice = markdown.markdown(raw, output_format="html5")
        md = md.replace(m.group(1), "", 1)

    # Drop the H1 and the two byline lines; the masthead already carries them.
    md = re.sub(r"\A# .*?\n+\*\*CSD358.*?\n.*?\n+---\n", "", md, flags=re.S)

    body = markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )

    # Figure placeholders -> the inline SVG fragments.
    body = re.sub(r"<p><strong>\[FIGURE 1[^<]*</strong></p>", _fragment("_tpl_fig1.html"), body, count=1)
    body = re.sub(r"<p><strong>\[FIGURE 2[^<]*</strong></p>", _fragment("_tpl_fig2.html"), body, count=1)

    # Typographic tidying the Markdown source leaves to the typesetter.
    body = body.replace("<hr />", '<hr class="rule" />')
    body = re.sub(r"<h2>(AI-use declaration)</h2>", r'<div class="declaration"><h2>\1</h2>', body, count=1)
    body = re.sub(r"<h2>(References)</h2>", r'</div>\n<h2 class="page-break">\1</h2>', body, count=1)
    body = re.sub(r"(<h2[^>]*>References</h2>\s*)<ol>", r'\1<ol class="refs">', body, count=1)

    head = _fragment("_tpl_head.html")
    notice_html = f'<div class="notice">{notice}</div>\n' if notice else ""

    return head + notice_html + body + "\n</body>\n</html>\n"


def main() -> int:
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html):,} bytes)")
    leftover = re.findall(r"\[[A-Z][A-Z0-9_]+\]", html)
    if leftover:
        print("WARNING: unfilled placeholders still in the report:", sorted(set(leftover)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
