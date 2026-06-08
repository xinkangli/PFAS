"""
对 15 篇文献:
  1) 用 Crossref / Unpaywall / 直接 landing page 抓 supplement-info URL
  2) 下载到 pfas_si/
报告每篇有哪些 SI 文件,哪些拿到,哪些失败。
"""
import os, re, json, time, requests, urllib.parse
import pandas as pd

OUT = "/7t/lxkzero/claude/qd/dier/pfas_si"
os.makedirs(OUT, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (research; mailto:lxk@example.org)"}

# Map study_id -> {doi, landing_page, hint}
df = pd.read_excel("pfas_ranked/keep_for_extraction_20260516_201933.xlsx") \
       .drop_duplicates("openalex_id").sort_values("rank").reset_index(drop=True)
df["study_id"] = ["ST{:03d}".format(i+1) for i in range(len(df))]
studies = df[["study_id","doi","landing_page","oa_url","title"]].fillna("").to_dict("records")

def clean_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d).strip()

def get(url, **kw):
    try:
        return requests.get(url, headers=H, timeout=25, allow_redirects=True, **kw)
    except Exception as e:
        return None

def save_pdf(content: bytes, sid: str, label: str, ext: str = "pdf"):
    p = os.path.join(OUT, f"{sid}_{label}.{ext}")
    with open(p, "wb") as f:
        f.write(content)
    return p

def looks_pdf(b): return b and b[:4] == b"%PDF"
def looks_zip(b): return b and b[:2] == b"PK"
def looks_xml(b): return b and (b.startswith(b"<?xml") or b[:1] == b"<")
def looks_office(b):
    # docx/xlsx 也是 zip; pure PDF/zip 上面已查过
    return b and len(b) > 200

results = []

# === 通用策略 ===
# 1) ACS:  https://pubs.acs.org/doi/suppl/{doi}/suppl_file/{filename}.pdf
# 2) Elsevier (ScienceDirect): "mmc1.docx", "mmc1.pdf" 通过 article landing page 拿
# 3) Wiley: doi://10.xxxx/...; supp 链接在 landing page <a> 中
# 4) RSC: https://www.rsc.org/suppdata/{journal-code}/.../{paper}/ 含 PDF
# 5) Springer (Frontiers): supplementary 链接在 landing page
# 6) BMC (Environmental Sciences Europe): https://enveurope.springeropen.com/articles/10.1186/{doi}#Sec-supp
# 7) ARCTIC: appendix PDF 在 view 页 / article tools

# === 直接尝试已知 SI URLs ===
SI_CANDIDATES = {
    # study_id : [list of (label, url, ext)]
    "ST001": [   # ACS est.3c06944
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/acs.est.3c06944/suppl_file/es3c06944_si_001.pdf", "pdf"),
    ],
    "ST003": [   # ACS es9003894
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/es9003894/suppl_file/es9003894_si_001.pdf", "pdf"),
    ],
    "ST005": [   # ACS est.9b05007
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/acs.est.9b05007/suppl_file/es9b05007_si_001.pdf", "pdf"),
    ],
    "ST006": [   # ACS es5048649
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/es5048649/suppl_file/es5048649_si_001.pdf", "pdf"),
    ],
    "ST007": [   # RSC d0em00510j
        ("SI", "https://www.rsc.org/suppdata/d0/em/d0em00510j/d0em00510j1.pdf", "pdf"),
        ("SI2","https://www.rsc.org/suppdata/d0/em/d0em00510j/d0em00510j2.xlsx","xlsx"),
    ],
    "ST008": [   # Elsevier S0269749122009514 (envpol.2022.119739)
        ("mmc1","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122009514-mmc1.docx","docx"),
        ("mmc1pdf","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122009514-mmc1.pdf","pdf"),
        ("mmc2","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122009514-mmc2.xlsx","xlsx"),
    ],
    "ST009": [   # ACS acsestwater.5c00048
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/acsestwater.5c00048/suppl_file/ew5c00048_si_001.pdf", "pdf"),
    ],
    "ST010": [   # ACS est.4c02134
        ("SI", "https://pubs.acs.org/doi/suppl/10.1021/acs.est.4c02134/suppl_file/es4c02134_si_001.pdf", "pdf"),
    ],
    "ST011": [   # Frontiers fmars.2024.1467022
        ("data_sheet1","https://www.frontiersin.org/articles/10.3389/fmars.2024.1467022/full#supplementary-material","html"),
    ],
    "ST012": [   # Elsevier S0269749122011216 (envpol.2022.119907)
        ("mmc1","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122011216-mmc1.docx","docx"),
        ("mmc1pdf","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122011216-mmc1.pdf","pdf"),
        ("mmc2","https://ars.els-cdn.com/content/image/1-s2.0-S0269749122011216-mmc2.xlsx","xlsx"),
    ],
    "ST013": [   # BMC s12302-020-00404-8
        # supplementary file pattern: https://static-content.springer.com/esm/art%3A10.1186%2Fs12302-020-00404-8/MediaObjects/12302_2020_404_MOESM1_ESM.docx etc.
        ("MOESM1","https://static-content.springer.com/esm/art%3A10.1186%2Fs12302-020-00404-8/MediaObjects/12302_2020_404_MOESM1_ESM.docx","docx"),
        ("MOESM1xlsx","https://static-content.springer.com/esm/art%3A10.1186%2Fs12302-020-00404-8/MediaObjects/12302_2020_404_MOESM1_ESM.xlsx","xlsx"),
        ("MOESM2","https://static-content.springer.com/esm/art%3A10.1186%2Fs12302-020-00404-8/MediaObjects/12302_2020_404_MOESM2_ESM.xlsx","xlsx"),
        ("MOESM2docx","https://static-content.springer.com/esm/art%3A10.1186%2Fs12302-020-00404-8/MediaObjects/12302_2020_404_MOESM2_ESM.docx","docx"),
    ],
    "ST014": [   # ARCTIC journal — Appendix 1
        # 试 article download view tools / supplementary files
        ("Appendix1","https://journalhosting.ucalgary.ca/index.php/arctic/article/downloadSuppFile/67260/4181","pdf"),
        ("Appendix1b","https://journalhosting.ucalgary.ca/index.php/arctic/article/downloadSuppFile/67260/4180","pdf"),
        ("Appendix1c","https://journalhosting.ucalgary.ca/index.php/arctic/article/downloadSuppFile/67260/4179","pdf"),
    ],
    "ST015": [   # Elsevier S0048969724025744 (scitotenv.2024.172357)
        ("mmc1","https://ars.els-cdn.com/content/image/1-s2.0-S0048969724025744-mmc1.docx","docx"),
        ("mmc1pdf","https://ars.els-cdn.com/content/image/1-s2.0-S0048969724025744-mmc1.pdf","pdf"),
        ("mmc2","https://ars.els-cdn.com/content/image/1-s2.0-S0048969724025744-mmc2.xlsx","xlsx"),
    ],
    # ST002 (会议摘要,无 SI), ST004 (DiVA 硕士论文,无 SI 单独文件)
}

for sid in sorted(SI_CANDIDATES.keys()):
    for (label, url, ext) in SI_CANDIDATES[sid]:
        out_path = os.path.join(OUT, f"{sid}_{label}.{ext}")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 5_000:
            print(f"[skip] {sid}_{label} exists")
            results.append((sid, label, url, "exists", out_path)); continue
        print(f"[try]  {sid}  {label}  {url}")
        r = get(url)
        ok = False
        if r and r.ok:
            c = r.content
            # 对于 PDF: 必须 %PDF;对于 office/xlsx: zip 头;对于 html landing: 跳过
            if ext == "pdf" and looks_pdf(c):
                ok = True
            elif ext in ("xlsx","docx") and looks_zip(c):
                ok = True
            elif ext == "html":
                # save html for later parsing
                ok = True
            elif looks_pdf(c) or looks_zip(c):
                ok = True
            if ok:
                p = save_pdf(c, sid, label, ext)
                print(f"       ✓ saved {len(c)//1024} KB → {p}")
                results.append((sid, label, url, "ok", p))
                continue
            else:
                print(f"       ✗ wrong content-type (status={r.status_code}, len={len(c)})")
        else:
            print(f"       ✗ fetch failed (status={r.status_code if r else 'no-resp'})")
        results.append((sid, label, url, "fail", ""))
        time.sleep(0.4)

print("\n=== SI Download Summary ===")
by_sid = {}
for sid, lbl, _, st, _ in results:
    by_sid.setdefault(sid, []).append((lbl, st))
for sid in sorted(by_sid):
    got = [l for l,s in by_sid[sid] if s in ("ok","exists")]
    fail= [l for l,s in by_sid[sid] if s == "fail"]
    print(f"  {sid}: {'OK ' + ','.join(got) if got else 'ALL FAIL'}  (failed: {','.join(fail) or '-'})")
