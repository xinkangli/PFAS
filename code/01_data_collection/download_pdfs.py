"""
下载 15 篇 OA 文献的 PDF。
优先级:有直接 pdf_url > 通过 OpenAlex/landing_page 解析 > unpaywall API。
"""
import os, re, time, requests, urllib.parse
import pandas as pd

OUT_DIR = "/7t/lxkzero/claude/qd/dier/pfas_pdfs"
os.makedirs(OUT_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (research-bot; mailto:lxk@example.org)"}

# 来自 keep_for_extraction 的 study_id → 候选 URL(按摘要里给的 oa_url/pdf_url 已知最稳的几个)
TARGETS = [
    # (study_id, label, list-of-urls-to-try, expected hint)
    ("ST001", "Sun2023_TreeSwallow_ESnT",
     ["https://pubs.acs.org/doi/pdf/10.1021/acs.est.3c06944"]),
    ("ST002", "Badry2022_ApexPredators_GNESt_OA",
     ["https://cms.gnest.org/sites/default/files/cest2019_00973_oral_paper.pdf"]),
    ("ST003", "Tomy2009_ArcticMarine_ES&T",
     ["https://pubs.acs.org/doi/pdf/10.1021/es9003894"]),
    ("ST004", "Bergman2017_LakeVattern_KTH",
     ["http://urn.kb.se/resolve?urn=urn:nbn:se:oru:diva-64616"]),
    ("ST005", "Munoz2020_StLawrence_ES&T_PMC",
     ["https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8190818/pdf/nihms-1707421.pdf",
      "https://www.ncbi.nlm.nih.gov/pmc/articles/8190818"]),
    ("ST006", "Lescord2015_HighArctic_ES&T",
     ["https://pubs.acs.org/doi/pdf/10.1021/es5048649"]),
    ("ST007", "Boulanger2021_NorwegianArctic_ESPI",
     ["https://pubs.rsc.org/en/content/articlepdf/2021/em/d0em00510j"]),
    ("ST008", "Munoz2022_StLawrence_EnvPollut",
     ["https://www.sciencedirect.com/science/article/pii/S0269749122009514/pdfft",
      "https://doi.org/10.1016/j.envpol.2022.119739"]),
    ("ST009", "Diaz2025_PFOSAlgae_ESTWater",
     ["https://pubs.acs.org/doi/pdf/10.1021/acsestwater.5c00048"]),
    ("ST010", "Arnot2024_FoodWebModel_ES&T",
     ["https://pubs.acs.org/doi/pdf/10.1021/acs.est.4c02134"]),
    ("ST011", "Wang2024_SaundersGull_FrontMarSci",
     ["https://public-pages-files-2025.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1467022/pdf",
      "https://www.frontiersin.org/articles/10.3389/fmars.2024.1467022/pdf"]),
    ("ST012", "Groffen2022_BelgianNorthSea_EnvPollut",
     ["https://www.sciencedirect.com/science/article/pii/S0269749122011216/pdfft",
      "https://doi.org/10.1016/j.envpol.2022.119907"]),
    ("ST013", "Schaefer2020_WFD_TMF_review_ESE",
     ["https://enveurope.springeropen.com/track/pdf/10.1186/s12302-020-00404-8"]),
    ("ST014", "Veillette2012_Ellesmere_ARCTIC",
     ["https://journalhosting.ucalgary.ca/index.php/arctic/article/download/67260/51170",
      "https://journalhosting.ucalgary.ca/index.php/arctic/article/view/67260/51170"]),
    ("ST015", "Gewurtz2024_IllinoisFish_STOTEN",
     ["https://www.sciencedirect.com/science/article/pii/S0048969724025744/pdfft",
      "https://doi.org/10.1016/j.scitotenv.2024.172357"]),
]

def looks_like_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"

def try_unpaywall(doi: str) -> str | None:
    if not doi:
        return None
    doi_clean = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi_clean}",
                         params={"email": "research@example.org"},
                         headers=HEADERS, timeout=20)
        if r.ok:
            j = r.json()
            best = j.get("best_oa_location") or {}
            url = best.get("url_for_pdf") or best.get("url")
            return url
    except Exception:
        return None
    return None

def fetch(url: str, timeout=30) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.ok and looks_like_pdf(r.content):
            return r.content
        # 有些 landing page 返回 HTML,看看有无 citation_pdf_url meta
        if r.ok and b"<html" in r.content[:1000].lower():
            m = re.search(rb'name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)', r.content)
            if m:
                pdf_url = m.group(1).decode()
                r2 = requests.get(pdf_url, headers=HEADERS, timeout=timeout, allow_redirects=True)
                if r2.ok and looks_like_pdf(r2.content):
                    return r2.content
        return None
    except Exception as e:
        print(f"     err: {e}")
        return None

# 读 metadata 拿 doi 备用
meta = pd.read_excel("pfas_ranked/keep_for_extraction_20260516_201933.xlsx").drop_duplicates("openalex_id").sort_values("rank").reset_index(drop=True)
meta["study_id"] = ["ST{:03d}".format(i+1) for i in range(len(meta))]
doi_by_id = dict(zip(meta["study_id"], meta["doi"].fillna("")))

results = []
for sid, label, urls in TARGETS:
    out_path = os.path.join(OUT_DIR, f"{sid}_{label}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 50_000:
        print(f"[skip] {sid}: already have {out_path}")
        results.append((sid, label, "exists", out_path)); continue

    print(f"[try]  {sid} {label}")
    pdf_bytes = None
    for u in urls:
        print(f"   - {u}")
        pdf_bytes = fetch(u)
        if pdf_bytes: break
    if not pdf_bytes:
        # fallback: unpaywall
        u2 = try_unpaywall(doi_by_id.get(sid, ""))
        if u2:
            print(f"   unpaywall → {u2}")
            pdf_bytes = fetch(u2)
    if pdf_bytes:
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"   ✓ saved {len(pdf_bytes)//1024} KB → {out_path}")
        results.append((sid, label, "ok", out_path))
    else:
        print(f"   ✗ FAILED")
        results.append((sid, label, "fail", ""))
    time.sleep(0.6)

print("\n=== Summary ===")
for r in results:
    print(f"  {r[0]} {r[2]:>6}  {r[1]}")
print(f"  {sum(1 for r in results if r[2] in ('ok','exists'))} / {len(results)} got PDF")
