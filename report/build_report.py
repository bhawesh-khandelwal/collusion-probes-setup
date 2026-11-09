"""Fill the official Apart template (docx export) with report/REPORT.md and write report/out/report.docx.

Needs python-docx. Convert to PDF with LibreOffice:
  soffice --headless --convert-to pdf --outdir report/out report/out/report.docx
Usage: python -I build_report.py TEMPLATE_DOCX
"""

import copy
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
AUTHOR, AFFIL = "Bhawesh", "Independent researcher"


def md_sections():
    text = (ROOT / "REPORT.md").read_text()
    title = re.search(r"^# (.+)$", text, re.M)[1].strip()
    abstract = text.split("## Abstract", 1)[1].split("\n## ", 1)[0].strip()
    body = "## " + text.split("\n## 1. ", 1)[1]
    body = body.replace("## ", "## 1. ", 1)
    return title, abstract, body


def tables_md():
    """Return {name: rows} from tables.md, rows as lists of cells."""
    t = (ROOT / "tables.md").read_text()
    out = {}
    for name, head in (("main", "## Table 1"), ("vs_peak", "## Table 2"), ("indomain", "## Table 3")):
        block = t.split(head, 1)[1].split("\n## ", 1)[0]
        out[name] = [parse_row(line) for line in block.splitlines()
                     if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line)]
    names = {"transfer": "Instructed", "implicit": "Implicit vs control", "implicit_coord": "Same setup (36)",
             "implicit2_coord": "Same setup (166)", "game": "Pricing game", "27B": "Qwen 27B", "27b": "Qwen 27B",
             "gemma": "Gemma", "27B disc.": "Qwen 27B disc.", "27B final": "Qwen 27B final"}
    for rows in out.values():
        for r in rows[1:]:
            r[0] = names.get(r[0], r[0])
            r[1] = names.get(r[1], r[1])
    for r in out["indomain"][1:]:
        if r[1].startswith("family"):
            r[-1] = "n/a (family unseen)"
    return out


def parse_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def add_runs(p, text, size=None, italic_all=False):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
        if italic_all:
            r.italic = True
        if size:
            r.font.size = Pt(size)


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom", "left", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "999999")
        b.append(e)
    tcPr.append(b)


def add_table(d, rows):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            set_cell_border(c)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            r = p.add_run(val.replace("*", "").strip() + (" *" if val.endswith("*") else ""))
            r.font.size = Pt(7.5)
            r.bold = i == 0
    d.add_paragraph()


def fill_header(d, title, abstract):
    outer = d.tables[0]
    tp = next(p for p in outer.rows[0].cells[0].paragraphs if p.style.name == "Title")
    tp.runs[0].text = title
    inner = outer.rows[1].cells[0].tables[0]
    cell = inner.rows[0].cells[0]
    grid = cell.tables[0]
    style = grid.rows[0].cells[0].paragraphs[0]
    para = copy.deepcopy(style._p)
    grid._tbl.addprevious(para)
    grid._tbl.getparent().remove(grid._tbl)
    p = docx.text.paragraph.Paragraph(para, cell)
    runs = p.runs
    for r in runs:
        r.text = r.text.replace("Author name 1", AUTHOR).replace("Affiliation", AFFIL)
    ab = outer.rows[1].cells[0].tables[1].rows[0].cells[0].paragraphs
    ab[0].runs[0].text = abstract
    for r in ab[0].runs[1:]:
        r.text = ""


def main():
    title, abstract, body = md_sections()
    d = docx.Document(sys.argv[1])
    fill_header(d, title, abstract)

    els = list(d.element.body.iterchildren())
    for el in els[3:]:
        if not el.tag.endswith("sectPr"):
            d.element.body.remove(el)
    sect = d.element.body[-1]

    tabs = tables_md()
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1
            continue
        if ln.startswith("### "):
            d.add_paragraph(ln[4:], style="Heading 3")
        elif ln.startswith("## "):
            d.add_paragraph(ln[3:], style="Heading 2")
        elif ln.startswith("TABLE:"):
            add_table(d, tabs[ln[6:]])
        elif ln.startswith("!["):
            path = re.search(r"\((.+)\)", ln)[1]
            width = {"fig1": 5.0, "fig2": 3.9, "fig3": 4.8}[Path(path).name[:4]]
            d.add_picture(str(ROOT / path), width=Inches(width))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-| ]+\|$", lines[i].strip()):
                    rows.append(parse_row(lines[i]))
                i += 1
            add_table(d, rows)
            continue
        elif ln.startswith("*") and ln.endswith("*") and not ln.startswith("**"):
            p = d.add_paragraph()
            add_runs(p, ln[1:-1], size=9, italic_all=True)
            p.paragraph_format.space_after = Pt(8)
        elif ln.startswith("- ") or re.match(r"^\d+\. ", ln):
            p = d.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            p.paragraph_format.space_after = Pt(3)
            add_runs(p, ("• " + ln[2:]) if ln.startswith("- ") else ln)
        else:
            p = d.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            add_runs(p, ln)
        i += 1

    d.element.body.remove(sect)
    d.element.body.append(sect)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    d.save(out / "report.docx")
    print("wrote", out / "report.docx")


if __name__ == "__main__":
    main()
