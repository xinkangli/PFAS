"""
解析 SI:把每篇 SI 的纯文本 + 全部表格抽出。
PDF: PyMuPDF + pdfplumber tables
DOCX: python-docx (paragraphs + all tables)
HTML: BeautifulSoup
输出:
  pfas_si_extracted/{sid}_si_text.txt
  pfas_si_extracted/{sid}_si_tables.xlsx (全部表格,每个 sheet 一张)
"""
import os, glob, re
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
from docx import Document

SI_DIR = "/7t/lxkzero/claude/qd/dier/pfas_si"
OUT = "/7t/lxkzero/claude/qd/dier/pfas_si_extracted"
os.makedirs(OUT, exist_ok=True)

def parse_pdf(path: str):
    text = "".join(p.get_text("text") for p in fitz.open(path))
    tables = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    tabs = page.extract_tables() or []
                except Exception:
                    tabs = []
                for j, tab in enumerate(tabs):
                    if tab and len(tab) >= 2:
                        w = max(len(r) for r in tab)
                        tab2 = [list(r) + [None]*(w-len(r)) for r in tab]
                        tables.append((i, j, pd.DataFrame(tab2)))
    except Exception as e:
        print("    pdfplumber err:", e)
    return text, tables

def parse_docx(path: str):
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    tables = []
    for j, t in enumerate(doc.tables):
        rows = []
        for r in t.rows:
            rows.append([c.text.strip() for c in r.cells])
        if rows and any(any(cell for cell in row) for row in rows):
            w = max(len(r) for r in rows)
            rows2 = [r + [""]*(w-len(r)) for r in rows]
            tables.append((1, j, pd.DataFrame(rows2)))
    return text, tables

def parse_html(path: str):
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    # 去掉 script style
    for tag in soup(["script","style","header","footer","nav"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    tables = []
    for j, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            rows.append([td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])])
        if rows and any(any(cell for cell in r) for r in rows):
            w = max(len(r) for r in rows)
            rows2 = [r + [""]*(w-len(r)) for r in rows]
            tables.append((1, j, pd.DataFrame(rows2)))
    return text, tables

for path in sorted(glob.glob(f"{SI_DIR}/*")):
    name = os.path.basename(path)
    sid = name.split("_")[0]
    ext = name.lower().rsplit(".", 1)[-1]
    print(f"=== {name} ({ext}) ===")
    if ext == "pdf":
        text, tables = parse_pdf(path)
    elif ext == "docx":
        text, tables = parse_docx(path)
    elif ext == "html":
        text, tables = parse_html(path)
    else:
        print("  skip"); continue

    text_path = f"{OUT}/{sid}_si_text.txt"
    with open(text_path, "w") as f: f.write(text)
    print(f"   text: {len(text)//1024} KB → {text_path}")
    if tables:
        out_xlsx = f"{OUT}/{sid}_si_tables.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            for (page, idx, df) in tables:
                sheet = f"p{page:02d}_t{idx}"[:31]
                # truncate to 1M cells max
                if df.shape[0] > 5000: df = df.iloc[:5000]
                if df.shape[1] > 50:   df = df.iloc[:, :50]
                df.to_excel(w, sheet_name=sheet, index=False, header=False)
        print(f"   tables: {len(tables)} → {out_xlsx}")
    else:
        print(f"   tables: 0")

print("\nDone. Outputs:", OUT)
