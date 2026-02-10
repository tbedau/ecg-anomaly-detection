"""Generate manuscript PDF from markdown source.

Usage:
    pip install fpdf2
    python generate_pdf.py

Produces manuscript.pdf in the same directory.
"""

from pathlib import Path

from fpdf import FPDF


class ManuscriptPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, "Researcher et al. - ECG Anomaly Detection", align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def clean_markdown(text):
    """Strip markdown formatting characters for plain text rendering."""
    return text.replace("**", "").replace("*", "")


def write_line(pdf, text, font_style="", font_size=10, align="L", spacing=5):
    """Write a line of text, resetting x position first."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", font_style, font_size)
    pdf.multi_cell(0, spacing, text, align=align)


def generate():
    pdf = ManuscriptPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    md_path = Path(__file__).parent / "manuscript.md"
    content = md_path.read_text()

    for line in content.split("\n"):
        stripped = line.strip()

        # Skip table separator rows (|---|---|)
        if stripped.startswith("|") and not stripped.replace("|", "").replace("-", "").strip():
            continue

        # Title
        if stripped.startswith("# ") and not stripped.startswith("## "):
            write_line(pdf, clean_markdown(stripped[2:]), "B", 14, "C", 7)
            pdf.ln(3)

        # Section heading
        elif stripped.startswith("## "):
            pdf.ln(4)
            write_line(pdf, clean_markdown(stripped[3:]), "B", 12, spacing=7)
            pdf.ln(2)

        # Subsection heading
        elif stripped.startswith("### "):
            pdf.ln(2)
            write_line(pdf, clean_markdown(stripped[4:]), "B", 10, spacing=6)
            pdf.ln(1)

        # Table data rows
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            row_text = "  ".join(f"{c:<14}" for c in cells)
            write_line(pdf, row_text, font_size=9)

        # Numbered list items
        elif stripped and len(stripped) > 2 and stripped[0].isdigit() and "." in stripped[:3]:
            write_line(pdf, clean_markdown(stripped))

        # Regular paragraph text
        elif stripped:
            write_line(pdf, clean_markdown(stripped))

        # Blank line
        else:
            pdf.ln(2)

    output_path = Path(__file__).parent / "manuscript.pdf"
    pdf.output(str(output_path))
    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate()
