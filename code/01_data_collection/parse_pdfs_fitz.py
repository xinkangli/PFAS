"""
用 PyMuPDF (fitz) 重抽文本——对 δ/µ 等希腊字符的保留好于 pdfplumber。
输出到 pfas_pdfs_extracted/{sid}_fitz.txt
"""
import os, glob, fitz

PDF_DIR = "/7t/lxkzero/claude/qd/dier/pfas_pdfs"
OUT_DIR = "/7t/lxkzero/claude/qd/dier/pfas_pdfs_extracted"

for p in sorted(glob.glob(f"{PDF_DIR}/*.pdf")):
    sid = os.path.basename(p).split("_")[0]
    doc = fitz.open(p)
    out = []
    for i, page in enumerate(doc, start=1):
        out.append(f"\n===== PAGE {i} =====\n")
        out.append(page.get_text("text"))
    full = "".join(out)
    out_path = os.path.join(OUT_DIR, f"{sid}_fitz.txt")
    with open(out_path, "w") as f:
        f.write(full)
    print(f"  {sid}: {len(full)//1024} KB,  δ count = {full.count('δ')}, µ count = {full.count('µ')}")
