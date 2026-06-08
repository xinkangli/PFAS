"""
NITE-CHRIP (Japan METI / NITE) collector
========================================
日本 PFAS BCF 数据非常丰富 — 很多 OECD 305 在这里有正式 measured BCF。

URL 结构 (2025 验证):
  Search:  https://www.chem-info.nite.go.jp/chem/chrip/chrip_search/srhInput?slGmnCmpyCd=02_e
  by CAS:  POST 表单 (textCasNo / textChmTrmCmpyCd)
  Result:  https://www.nite.go.jp/chem/chrip/chrip_search/cmpInfDsp?cid=Cxxx-xxx-xxxA&...

每个 substance 页里有 "Bioconcentration / Bioaccumulation" 段，含:
  - BCF value (fish)
  - test method (OECD 305 / MITI test)
  - exposure concentration
  - duration

NITE 页面较稳定但偶尔改版。这个 collector 用宽松正则，
任何字段抓不到不会 crash，只是该字段为 None。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pfas_targets import PFAS_TARGETS, cas_to_meta
from schema import insert_record

CHRIP_BASE = "https://www.chem-info.nite.go.jp"
CHRIP_SEARCH_URL = f"{CHRIP_BASE}/chem/chrip/chrip_search/srhInput"
CHRIP_SEARCH_POST = f"{CHRIP_BASE}/chem/chrip/chrip_search/srhSimResLst"

HEADERS = {
    "User-Agent": "pfas-bcf-research-collector/1.0 (academic)",
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.7",
}
SLEEP = 2.0


def _get(session, url):
    time.sleep(SLEEP)
    r = session.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def _post(session, url, data):
    time.sleep(SLEEP)
    r = session.post(url, headers=HEADERS, data=data, timeout=60)
    r.raise_for_status()
    return r


def find_substance_page(session: requests.Session, cas: str) -> str | None:
    """Submit the CHRIP CAS search form and return the substance detail page URL."""
    # First load the search form to get the session cookie + any CSRF
    _get(session, CHRIP_SEARCH_URL)
    form = {
        "textCasNo": cas,
        "slLng": "en",
        "slGmnCmpyCd": "02_e",
    }
    r = _post(session, CHRIP_SEARCH_POST, form)
    soup = BeautifulSoup(r.text, "html.parser")
    a = soup.find("a", href=re.compile(r"cmpInfDsp\?cid="))
    if a and a.get("href"):
        href = a["href"]
        if href.startswith("http"):
            return href
        return CHRIP_BASE + ("/" + href if not href.startswith("/") else href)
    return None


# Patterns inside CHRIP "Hazard Information / Bioconcentration" panel
RX_BCF      = re.compile(r"BCF[^0-9\-\n]{0,40}([\d,\.]+)\s*(?:L?\s*/?\s*kg)?", re.I)
RX_LOG_BCF  = re.compile(r"log\s*BCF[^0-9\-]{0,15}(-?[\d\.]+)", re.I)
RX_SPECIES  = re.compile(r"(Cyprinus\s+carpio|Oncorhynchus\s+mykiss|Danio\s+rerio|carp|trout|medaka|Oryzias\s+latipes)", re.I)
RX_EXP      = re.compile(r"(?:exposure|test)\s*period[^0-9]{0,20}([\d\.]+)\s*(d|day|week)", re.I)
RX_TEMP     = re.compile(r"(?:temperature|water\s+temp\.?)[^0-9\-]{0,15}(-?[\d\.]+)\s*°?C", re.I)
RX_CONC     = re.compile(r"(?:test\s+conc|exposure\s+conc)[^0-9]{0,15}([\d\.]+)\s*(µg/L|ug/L|mg/L|ng/L)", re.I)


def parse_substance_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    # Find any section about Bioconcentration / Bioaccumulation
    section_text = ""
    for h in soup.find_all(["h2", "h3", "h4", "th", "strong"]):
        title = h.get_text(" ", strip=True).lower()
        if "bioconcent" in title or "bioaccum" in title:
            # Take following text until next h-tag
            nxt = h
            buf = []
            for _ in range(40):
                nxt = nxt.find_next(string=True)
                if nxt is None: break
                buf.append(str(nxt))
            section_text += " ".join(buf) + " "
    if not section_text:
        section_text = soup.get_text(" ", strip=True)  # fallback: whole page

    records = []
    bcf_matches = list(RX_BCF.finditer(section_text))
    log_matches = list(RX_LOG_BCF.finditer(section_text))
    spec_m  = RX_SPECIES.search(section_text)
    exp_m   = RX_EXP.search(section_text)
    temp_m  = RX_TEMP.search(section_text)
    conc_m  = RX_CONC.search(section_text)

    species = spec_m.group(1) if spec_m else None
    exp_days = None
    if exp_m:
        v = float(exp_m.group(1)); u = exp_m.group(2).lower()
        if u.startswith("week"): v *= 7
        exp_days = v
    temp = float(temp_m.group(1)) if temp_m else None
    test_conc = f"{conc_m.group(1)} {conc_m.group(2)}" if conc_m else None

    # If we found a log BCF only, derive value
    if not bcf_matches and log_matches:
        for m in log_matches:
            try:
                lv = float(m.group(1))
                records.append({"value": 10**lv, "log_value": lv})
            except ValueError:
                pass
    else:
        for m in bcf_matches:
            try:
                v = float(m.group(1).replace(",", ""))
                records.append({"value": v, "log_value": None})
            except ValueError:
                pass

    for r in records:
        r["species"] = species
        r["exposure_days"] = exp_days
        r["temperature_c"] = temp
        r["test_concentration"] = test_conc
    return records


def collect_for_cas(session: requests.Session, cas: str) -> list[dict]:
    meta = cas_to_meta().get(cas)
    if not meta:
        return []
    print(f"[NITE] {meta['short']} (CAS {cas})")
    page_url = find_substance_page(session, cas)
    if not page_url:
        print(f"[NITE]   no substance page")
        return []
    print(f"[NITE]   page {page_url}")
    r = _get(session, page_url)
    items = parse_substance_page(r.text)
    rows = []
    for item in items:
        rec = {
            "pfas_short": meta["short"], "pfas_name": meta["name"],
            "cas": meta["cas"], "family": meta["family"], "chain_len": meta["chain_len"],
            "endpoint_type": "BCF",
            "value": item.get("value"),
            "log_value": item.get("log_value"),
            "value_units": "L/kg",
            "raw_value": str(item.get("value") or ""),
            "species_sci": item.get("species"),
            "tissue": "whole body",  # NITE/MITI standard is whole-body fish BCF
            "exposure_days": item.get("exposure_days"),
            "test_concentration": item.get("test_concentration"),
            "temperature_c": item.get("temperature_c"),
            "exposure_route": "aqueous",
            "water_type": "freshwater",
            "guideline": "OECD 305 / METI",
            "study_type": "laboratory",
            "source": "NITE-CHRIP",
            "source_url": page_url,
            "citation": "NITE-CHRIP (Japan METI)",
            "reliability": "Japanese regulatory dossier",
        }
        rows.append(rec)
    return rows


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
            print(f"[NITE] error on {cas}: {e}")
    print(f"[NITE] inserted {n_new} new rows")
    return n_new


if __name__ == "__main__":
    from schema import init_db, stats
    conn = init_db("data/pfas_master.sqlite")
    run(conn, cas_subset=["1763-23-1", "335-67-1"])  # PFOS, PFOA
    print(stats(conn))
