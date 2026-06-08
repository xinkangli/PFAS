"""
解析已下载的 5 篇 PDF,提取:
  - 文中表格(Table 1, Table 2 等)→ 浓度矩阵 / TMF
  - 文中数值型 TMF / BCF / BAF 描述
输出:
  - pfas_pdfs_extracted/{study_id}_text.txt        正文纯文本
  - pfas_pdfs_extracted/{study_id}_tables.xlsx     所有抽到的表格
  - pfas_pdfs_extracted/{study_id}_tmf_lines.txt   含 TMF/BCF/BAF/biomagnif 的句子
"""
import os, re, glob
import pandas as pd
import pdfplumber

PDF_DIR = "/7t/lxkzero/claude/qd/dier/pfas_pdfs"
OUT_DIR = "/7t/lxkzero/claude/qd/dier/pfas_pdfs_extracted"
os.makedirs(OUT_DIR, exist_ok=True)

KEYWORDS = re.compile(r"\b(TMF|BMF|BAF|BCF|biomagnif|trophic magnif|trophic transfer|δ\s?15N|delta\s?15N|slope)\b", re.I)
NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")

def extract_pdf(path: str, sid: str):
    text_parts = []
    table_parts = []  # list of (page_num, table_idx, df)
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            text_parts.append(f"\n===== PAGE {i} =====\n{t}")
            try:
                tabs = page.extract_tables() or []
            except Exception:
                tabs = []
            for j, tab in enumerate(tabs):
                if not tab or len(tab) < 2:
                    continue
                # normalise jagged rows
                width = max(len(r) for r in tab)
                tab2 = [list(r) + [None]*(width-len(r)) for r in tab]
                df = pd.DataFrame(tab2)
                table_parts.append((i, j, df))
    full_text = "\n".join(text_parts)

    # write full text
    with open(os.path.join(OUT_DIR, f"{sid}_text.txt"), "w") as f:
        f.write(full_text)

    # write tables
    if table_parts:
        out_xlsx = os.path.join(OUT_DIR, f"{sid}_tables.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            for (page, idx, df) in table_parts:
                sheet = f"p{page:02d}_t{idx}"[:31]
                df.to_excel(w, sheet_name=sheet, index=False, header=False)

    # write TMF-relevant sentences
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    hits = [s.strip() for s in sentences if KEYWORDS.search(s)]
    with open(os.path.join(OUT_DIR, f"{sid}_tmf_lines.txt"), "w") as f:
        f.write("\n---\n".join(hits))

    return len(text_parts), len(table_parts), len(hits)

print(f"PDFs in {PDF_DIR}:")
for p in sorted(glob.glob(f"{PDF_DIR}/*.pdf")):
    name = os.path.basename(p)
    sid = name.split("_")[0]
    sz = os.path.getsize(p)
    print(f"  {sid}  {sz//1024:5d} KB  {name}")
    pages, tables, tmf_lines = extract_pdf(p, sid)
    print(f"     → {pages} pages, {tables} tables, {tmf_lines} TMF-relevant sentences")

print(f"\nOutputs in: {OUT_DIR}")
