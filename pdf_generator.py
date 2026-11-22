from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def create_roadmap_pdf(roadmap: dict, output_path: str) -> str:
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="NeriTitle", parent=styles["Title"],
        alignment=1, textColor=colors.HexColor("#17324D"),
        fontSize=18, leading=22, spaceAfter=12
    ))

    content = [
        Paragraph("NERI — PERSONALIZED LEARNING ROADMAP", styles["NeriTitle"]),
        Paragraph("Target role: " + escape(str(roadmap["role"])), styles["Heading2"]),
        Paragraph(f"Duration: {roadmap['duration_weeks']} weeks", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Priority skill gaps", styles["Heading2"]),
    ]

    for gap in roadmap["main_gaps"]:
        content.append(Paragraph("• " + escape(str(gap)), styles["BodyText"]))

    for week in roadmap["weeks"]:
        content.extend([
            Spacer(1, 10),
            Paragraph(
                f"Week {week['week']}: {escape(str(week['objective']))}",
                styles["Heading2"]
            ),
            Paragraph(
                "<b>Focus:</b> " + escape(str(week["focus_area"])),
                styles["BodyText"]
            ),
            Paragraph("Learning tasks", styles["Heading3"]),
        ])

        for task in week["tasks"]:
            content.append(Paragraph("• " + escape(str(task)), styles["BodyText"]))

        content.append(Paragraph(
            "<b>Practice exercise:</b> " +
            escape(str(week["practice_exercise"])), styles["BodyText"]
        ))
        content.append(Paragraph(
            "<b>Milestone:</b> " + escape(str(week["milestone"])),
            styles["BodyText"]
        ))

    content.extend([
        Spacer(1, 10),
        Paragraph("Final goal", styles["Heading2"]),
        Paragraph(escape(str(roadmap["final_goal"])), styles["BodyText"])
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.build(content)
    return output_path
