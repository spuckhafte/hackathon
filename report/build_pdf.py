"""Render report/report.html to report/NyayaTrace-report.pdf.

Chromium via Playwright is the primary path, because the report's typography and
the two inline SVG figures are CSS. If Chromium is unavailable, pass --fallback
to produce a plain reportlab PDF of report.md so there is always *something*
submittable; the fallback is deliberately ugly so nobody mistakes it for the
real artefact.

    python report/build_pdf.py
    python report/build_pdf.py --fallback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "report.html"
PDF = HERE / "NyayaTrace-report.pdf"


def render_with_chromium() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(HTML.as_uri(), wait_until="networkidle")
        # Google Fonts may be absent offline; the CSS stack falls back cleanly.
        page.emulate_media(media="print")
        page.pdf(
            path=str(PDF),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "16mm", "right": "17mm", "bottom": "15mm", "left": "17mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;font-family:Arial,sans-serif;font-size:7.5pt;'
                'color:#8a929c;padding:0 17mm;display:flex;justify-content:space-between;">'
                "<span>NyayaTrace &middot; CSD358 Information Retrieval &middot; Track T6</span>"
                '<span class="pageNumber"></span>'
                "</div>"
            ),
        )
        browser.close()


def render_fallback() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    md = (HERE / "report.md").read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12, spaceBefore=10)

    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    flow = []
    for block in md.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        esc = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if block.startswith("#"):
            flow.append(Paragraph(esc.lstrip("# "), h))
        else:
            flow.append(Paragraph(esc.replace("\n", " "), body))
            flow.append(Spacer(1, 3))
    doc.build(flow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", action="store_true",
                    help="skip Chromium and use the plain reportlab path")
    args = ap.parse_args()

    if not args.fallback:
        try:
            render_with_chromium()
            print(f"wrote {PDF}")
            return 0
        except Exception as exc:  # noqa: BLE001 - we want the message, then the fallback
            print(f"chromium path failed: {exc}", file=sys.stderr)
            print("falling back to reportlab (run: python -m playwright install chromium)",
                  file=sys.stderr)

    render_fallback()
    print(f"wrote {PDF} (fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
