import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from logger import get_logger

log = get_logger("report_builder")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
PDF_PATH = os.path.join(OUTPUT_DIR, "report.pdf")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=24, textColor=colors.HexColor("#1a1a2e"),
                                spaceAfter=6, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontSize=11, textColor=colors.HexColor("#555555"),
                                   spaceAfter=4, alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             fontSize=14, textColor=colors.HexColor("#16213e"),
                             spaceBefore=14, spaceAfter=6,
                             borderPad=2),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontSize=12, textColor=colors.HexColor("#0f3460"),
                             spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=10, leading=16,
                               textColor=colors.HexColor("#2d2d2d"),
                               alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontSize=10, leading=15,
                                 leftIndent=16, spaceAfter=3,
                                 textColor=colors.HexColor("#2d2d2d")),
        "source": ParagraphStyle("source", parent=base["Normal"],
                                 fontSize=8.5, leading=13,
                                 textColor=colors.HexColor("#444444"),
                                 leftIndent=12, spaceAfter=2),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
                                 fontSize=8, textColor=colors.grey,
                                 alignment=TA_CENTER),
    }


def _parse_summary_to_flowables(summary: str, styles: dict) -> list:
    """Convert plain text summary into styled paragraphs, detecting section headers."""
    flowables = []
    for line in summary.split("\n"):
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 6))
            continue
        # Detect markdown-style headers
        if line.startswith("## "):
            flowables.append(Paragraph(line[3:], styles["h1"]))
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc"), spaceAfter=4))
        elif line.startswith("# "):
            flowables.append(Paragraph(line[2:], styles["h1"]))
        elif line.startswith("**") and line.endswith("**"):
            flowables.append(Paragraph(f"<b>{line[2:-2]}</b>", styles["body"]))
        elif line.startswith("- ") or line.startswith("* "):
            flowables.append(Paragraph(f"• {line[2:]}", styles["bullet"]))
        elif line.startswith(tuple("123456789")) and ". " in line[:4]:
            flowables.append(Paragraph(f"• {line}", styles["bullet"]))
        else:
            flowables.append(Paragraph(line, styles["body"]))
    return flowables


def build_report(result: dict) -> list:
    """Returns the result dict unchanged — PDF is built in save_report."""
    return result


def save_report(result: dict, filename: str = None):
    if filename is None:
        filename = PDF_PATH

    topic = result["topic"]
    summary = result["summary"]
    sources = result.get("sources", [])
    chunks = result.get("chunks_retrieved", 0)
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )

    s = _styles()
    story = []

    # ── Cover block ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("AI Research Assistant", s["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Research Report", s["title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="80%", thickness=2,
                            color=colors.HexColor("#0f3460"), hAlign="CENTER"))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(topic, s["h2"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated: {date}  |  Sources indexed: {chunks} chunks",
                           s["subtitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 0.8 * cm))

    # ── Summary content ───────────────────────────────────────────────────────
    story.append(Paragraph("Research Summary", s["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#cccccc"), spaceAfter=8))
    story += _parse_summary_to_flowables(summary, s)
    story.append(Spacer(1, 0.8 * cm))

    # ── Force at least page 2 ─────────────────────────────────────────────────
    story.append(PageBreak())

    # ── Sources section ───────────────────────────────────────────────────────
    story.append(Paragraph("References & Sources", s["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#cccccc"), spaceAfter=8))

    if sources:
        for i, src in enumerate(sources, 1):
            title = src.get("title") or src["url"]
            url = src["url"]
            story.append(Paragraph(
                f'<b>[{i}]</b> {title}<br/>'
                f'<font color="#0f3460"><u>{url}</u></font>',
                s["source"]
            ))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No sources recorded.", s["body"]))

    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Generated by AI Research Assistant  •  {date}",
        s["footer"]
    ))

    doc.build(story)
    log.info(f"PDF report saved to: {filename}")
    print(f"[Report] Saved to {filename}")
