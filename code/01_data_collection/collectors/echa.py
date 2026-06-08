"""
ECHA Dissemination Database collector
=====================================
ECHA REACH 是你最看重的 BCF 来源 — OECD 305 测试数据基本都在这。
没有官方 API，但 dossier 网页是 HTML 可抓的 (注意速率！否则会被 ratelimit)。

URL 结构 (2025 验证):
  搜索:        https://echa.europa.eu/search-for-chemicals
  物质页:      https://echa.europa.eu/substance-information/-/substanceinfo/100.XXX.XXX
  Dossier:    https://echa.europa.eu/registration-dossier/-/registered-dossier/{DOSSIER_ID}
  Bioaccum:   .../registered-dossier/{ID}/5/4/2   (aquatic)
             .../registered-dossier/{ID}/5/4/3   (sediment)
             .../registered-dossier/{ID}/5/4/4   (terrestrial)

Bioaccumulation section 列出多条 "endpoint study record"，每条链接到详细页，
详细页里有: study type / guideline / species / BCF value / exposure / temperature / ...

这个 collector 做两件事:
  1. 用 CAS 找到 dossier ID (search → substance page → dossier link)
  2. 抓 bioaccumulation 列表页，逐条抓 endpoint study record，解析为结构化记录

注意:
  - ECHA 对快速访问会返回 429。每次请求间隔 ≥2 秒。
  - 部分 dossier 内容在 IUCLID viewer 里 (JS 渲染)。
    对那些页面，文本提取可能不全 — 我们仍会拿到大部分关键值；
    彻底覆盖需要 Playwright，见模块底部的 fetch_with_playwright() 钩子。
  - 不要并发！ECHA 会拉黑 IP。
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pfas_targets import PFAS_TARGETS, cas_to_meta
from schema import insert_record


ECHA_BASE = "https://echa.europa.eu"
ECHA_SEARCH = f"{ECHA_BASE}/search-for-chemicals"
SLEEP_SECONDS = 2.5  # be polite

HEADERS = {
    "User-Agent": "pfas-bcf-research-collector/1.0 (academic; contact: your_email@example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(session: requests.Session, url: str) -> requests.Response | None:
    """GET with retry + politeness."""
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[ECHA] 429 ratelimit, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(SLEEP_SECONDS)
            return r
        except requests.RequestException as e:
            print(f"[ECHA] retry {attempt}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Step 1: CAS → dossier ID
# ---------------------------------------------------------------------------

def find_dossier_ids(session: requests.Session, cas: str) -> list[str]:
    """
    Search by CAS, return list of dossier IDs (one substance can have multiple dossiers).
    """
    params = {
        "p_p_id": "disssimplesearch_WAR_disssearchportlet",
        "p_p_lifecycle": "0",
        "p_p_state": "normal",
        "_disssimplesearch_WAR_disssearchportlet_searchText": cas,
    }
    url = f"{ECHA_SEARCH}?{urlencode(params)}"
    r = _get(session, url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    # Look for substanceinfo links
    sub_links = {a["href"] for a in soup.find_all("a", href=True)
                 if "substance-information/-/substanceinfo/" in a["href"]}
    dossier_ids: list[str] = []
    for sub_link in sub_links:
        full = urljoin(ECHA_BASE, sub_link)
        r2 = _get(session, full)
        if r2 is None:
            continue
        # On the substance page, look for the "Registered substances" / dossier link
        m = re.findall(r"/registration-dossier/-/registered-dossier/(\d+)", r2.text)
        dossier_ids.extend(set(m))
    return list(dict.fromkeys(dossier_ids))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Step 2: dossier → bioaccumulation endpoint study records
# ---------------------------------------------------------------------------

# bioaccumulation section codes in the dossier toc (verified on REACH dossiers 2025)
BIOACC_SECTIONS = {
    "aquatic":      "5/4/2",
    "sediment":     "5/4/3",
    "terrestrial":  "5/4/4",
}


def get_endpoint_record_links(session: requests.Session, dossier_id: str, section: str) -> list[str]:
    url = f"{ECHA_BASE}/registration-dossier/-/registered-dossier/{dossier_id}/{section}"
    r = _get(session, url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    # endpoint study records live at URLs of the same /5/4/X/N pattern
    pat = re.compile(rf"/registered-dossier/{dossier_id}/{section}/\d+")
    return list({urljoin(ECHA_BASE, a["href"])
                 for a in soup.find_all("a", href=pat)})


# Regexes for the structured endpoint-study fields
RX = {
    "guideline":   re.compile(r"(OECD\s*\d+[A-Z\-IVi]*|EU\s*Method\s*[A-Z\.]+|EPA\s*OPPTS\s*\d+)", re.I),
    "bcf_value":   re.compile(r"BCF[^0-9\-\n]{0,30}([\d,\.]+)\s*(L\s*/\s*kg[^\s\.,;)]*)?", re.I),
    "log_bcf":     re.compile(r"log\s*BCF[^0-9\-\n]{0,15}(-?[\d\.]+)", re.I),
    "species":     re.compile(r"(Cyprinus\s+carpio|Oncorhynchus\s+mykiss|Danio\s+rerio|Pimephales\s+promelas|Lepomis\s+macrochirus|Salmo\s+salar|Ictalurus\s+punctatus|Carassius\s+auratus)", re.I),
    "exposure_d":  re.compile(r"(?:exposure\s+duration|test\s+duration)[^0-9]{0,30}([\d\.]+)\s*(d|day|days|h|hour|hours|wk|weeks?)", re.I),
    "temperature": re.compile(r"(?:temperature)[^0-9\-]{0,20}(-?[\d\.]+)\s*(?:°|deg|degree)?\s*C", re.I),
    "tissue":      re.compile(r"(whole\s+body|liver|muscle|kidney|blood|plasma|fillet|carcass|gill|gonad)", re.I),
    "klimisch":    re.compile(r"reliability[^0-9]{0,30}(\d)", re.I),
    "basis":       re.compile(r"(wet\s*weight|dry\s*weight|lipid\s*weight|w\.w\.|d\.w\.|l\.w\.)", re.I),
    "water_type":  re.compile(r"(freshwater|marine|brackish|seawater)", re.I),
}


def parse_endpoint_record(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    # Strip down to text; ECHA renders mostly in dl/dt/dd or table cells.
    text = soup.get_text(" ", strip=True)
    out: dict = {"raw_text_snippet": text[:2000]}

    if (m := RX["guideline"].search(text)):     out["guideline"] = m.group(1).strip()
    if (m := RX["log_bcf"].search(text)):       out["log_bcf"] = float(m.group(1))
    if (m := RX["bcf_value"].search(text)):
        raw_v = m.group(1).replace(",", "")
        try:
            out["bcf_value"] = float(raw_v)
        except ValueError:
            pass
        if m.group(2):
            out["bcf_unit"] = m.group(2).strip()
    if (m := RX["species"].search(text)):       out["species"] = m.group(1)
    if (m := RX["exposure_d"].search(text)):
        v = float(m.group(1)); u = m.group(2).lower()
        if u.startswith("h"): v /= 24
        elif u.startswith("wk") or "week" in u: v *= 7
        out["exposure_days"] = v
    if (m := RX["temperature"].search(text)):   out["temperature_c"] = float(m.group(1))
    if (m := RX["tissue"].search(text)):        out["tissue"] = m.group(1).lower()
    if (m := RX["klimisch"].search(text)):      out["klimisch"] = m.group(1)
    if (m := RX["basis"].search(text)):         out["basis"] = m.group(1).lower()
    if (m := RX["water_type"].search(text)):    out["water_type"] = m.group(1).lower()
    return out


# ---------------------------------------------------------------------------
# Glue: per-PFAS pipeline
# ---------------------------------------------------------------------------

def collect_for_cas(session: requests.Session, cas: str) -> list[dict]:
    meta = cas_to_meta().get(cas)
    if not meta:
        return []
    print(f"[ECHA] {meta['short']} (CAS {cas}) — searching")
    dossiers = find_dossier_ids(session, cas)
    print(f"[ECHA]   found dossiers: {dossiers}")
    out = []
    for did in dossiers:
        for water_type, sec in BIOACC_SECTIONS.items():
            urls = get_endpoint_record_links(session, did, sec)
            for u in urls:
                r = _get(session, u)
                if r is None:
                    continue
                parsed = parse_endpoint_record(r.text)
                value = parsed.get("bcf_value")
                log_value = parsed.get("log_bcf")
                if value is None and log_value is not None:
                    value = 10 ** log_value
                rec = {
                    "pfas_short": meta["short"], "pfas_name": meta["name"],
                    "cas": meta["cas"], "family": meta["family"], "chain_len": meta["chain_len"],
                    "endpoint_type": "BCF",  # bioaccumulation section default; refine via raw_text
                    "value": value, "log_value": log_value,
                    "value_units": parsed.get("bcf_unit") or "L/kg",
                    "raw_value": str(parsed.get("bcf_value") or parsed.get("log_bcf") or ""),
                    "raw_units": parsed.get("bcf_unit"),
                    "basis": parsed.get("basis"),
                    "species_sci": parsed.get("species"),
                    "tissue": parsed.get("tissue"),
                    "exposure_days": parsed.get("exposure_days"),
                    "exposure_route": "aqueous" if water_type != "sediment" else "sediment",
                    "temperature_c": parsed.get("temperature_c"),
                    "water_type": parsed.get("water_type") or (
                        "freshwater" if water_type == "aquatic" else None),
                    "guideline": parsed.get("guideline"),
                    "study_type": "laboratory",
                    "source": "ECHA",
                    "source_url": u,
                    "citation": f"ECHA dossier {did}",
                    "reliability": f"Klimisch {parsed['klimisch']}" if parsed.get("klimisch") else None,
                    "notes": parsed.get("raw_text_snippet"),
                }
                out.append(rec)
    return out


def run(conn, data_dir: Path | None = None, cas_subset: list[str] | None = None):
    session = requests.Session()
    cas_list = cas_subset or [r["cas"] for r in PFAS_TARGETS]
    n_new = 0
    for cas in cas_list:
        try:
            for rec in collect_for_cas(session, cas):
                if insert_record(conn, rec):
                    n_new += 1
        except Exception as e:
            print(f"[ECHA] error on {cas}: {e}")
    print(f"[ECHA] inserted {n_new} new rows")
    return n_new


# ---------------------------------------------------------------------------
# Optional Playwright fallback for JS-rendered dossier pages
# ---------------------------------------------------------------------------
def fetch_with_playwright(url: str) -> str | None:
    """
    If `requests` returns a near-empty body (a few KB) for a dossier endpoint page,
    that page is rendered by JS and you need a real browser.
    Uncomment and install playwright: `pip install playwright && playwright install chromium`.
    """
    # from playwright.sync_api import sync_playwright
    # with sync_playwright() as p:
    #     b = p.chromium.launch(headless=True)
    #     page = b.new_page()
    #     page.goto(url, wait_until="networkidle", timeout=60_000)
    #     html = page.content()
    #     b.close()
    #     return html
    return None


if __name__ == "__main__":
    from schema import init_db, stats
    conn = init_db("data/pfas_master.sqlite")
    # Try one PFAS first to verify the pipeline before unleashing on all
    run(conn, cas_subset=["1763-23-1"])  # PFOS
    print(stats(conn))
