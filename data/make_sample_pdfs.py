"""Generate sample policy/finance PDFs (incl. a table) for demo + evaluation.

Run from repo root:
    python -m data.make_sample_pdfs

Produces:
    data/sample_pdfs/annual_report_2025.pdf   (prose + a financial table)
    data/sample_pdfs/risk_policy.pdf          (prose, risk factors)
"""
from __future__ import annotations

import os

import fitz

OUT_DIR = "data/sample_pdfs"

def _draw_table(page, x: float, y: float, rows: list[list[str]],
                col_w: float = 150, row_h: float = 24) -> float:
    """Draw a REAL bordered table (cell rectangles + text) so pdfplumber /
    PyMuPDF find_tables() can detect and extract it. Returns the new y."""
    n_cols = max(len(r) for r in rows)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            cell = row[c] if c < len(row) else ""
            rect = fitz.Rect(x + c * col_w, y + r * row_h,
                             x + (c + 1) * col_w, y + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_text((rect.x0 + 4, rect.y0 + 16), cell, fontsize=10)
    return y + len(rows) * row_h + 10

def _write_pdf(path: str, title: str, paragraphs: list[str], table_rows: list[list[str]] | None = None) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    page.insert_text((72, y), title, fontsize=16)
    y += 30
    for para in paragraphs:
        for line in _wrap(para, 90):
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        y += 8
        if y > 700:
            page = doc.new_page()
            y = 72
    if table_rows:
        y += 10
        page.insert_text((72, y), "Financial Summary Table", fontsize=13)
        y += 22
        _draw_table(page, 72, y, table_rows)
    doc.save(path)
    doc.close()

def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    _write_pdf(
        os.path.join(OUT_DIR, "annual_report_2025.pdf"),
        "Acme Corp Annual Report 2025",
        [
            "Revenue Overview. The total revenue in Q3 2025 was $4.2 million, an "
            "increase of 18% compared to the same quarter in the prior year. Growth "
            "was driven primarily by the enterprise software segment.",
            "Dividend Policy. The Board approved a quarterly dividend of $0.15 per "
            "share, payable to shareholders of record as of November 1, 2025. The "
            "company intends to maintain a stable dividend going forward.",
            "Outlook. Management expects continued growth in the fourth quarter, "
            "supported by new product launches and expansion into international markets.",
        ],
        table_rows=[
            ["Metric", "Q3 2025", "Q3 2024"],
            ["Total Revenue", "$4.2M", "$3.6M"],
            ["Operating Margin", "22%", "19%"],
            ["Net Income", "$0.9M", "$0.7M"],
        ],
    )

    _write_pdf(
        os.path.join(OUT_DIR, "risk_policy.pdf"),
        "Acme Corp Risk & Compliance Policy",
        [
            "Risk Factors. The primary risk factor identified by management is "
            "customer concentration: the top three customers account for 41% of "
            "total revenue. Loss of a major customer could materially affect results.",
            "Compliance. All employees must complete annual compliance training. "
            "Violations of the code of conduct are subject to disciplinary action up "
            "to and including termination.",
            "Data Protection. The company does not sell personal data. Access to "
            "sensitive records is restricted on a need-to-know basis.",
        ],
    )

    print(f"Wrote sample PDFs to {OUT_DIR}/")

if __name__ == "__main__":
    main()
