"""
PubMed literature helper for BAF & TMF
======================================
TMF (trophic magnification factor) 几乎不存在于数据库 — 必须从文献抓。
BAF 类似 (real-world field studies)。

这个 collector 做不到的事:
  - 它不会从 PDF 表格里直接抓数字。那需要文本挖掘/手工。
  
它能做的事:
  - 用 NCBI E-utilities (esearch + efetch) 拿到候选论文清单
  - 把 title / abstract / DOI / 年份 / 期刊 / PMID 存进一个 `literature_candidates` 表
  - 你拿到这张表后可以:
      1. 按引用次数排序 (用 Crossref / OpenAlex)
      2. 看 abstract 决定哪些值得手工提取 TMF / BAF
      3. 手工把数字补进 master 表 (source='literature')

NCBI E-utilities 是免费、有 API、对学术用户友好的。每秒 3 个 request，
带 api_key 可以到 10 (https://www.ncbi.nlm.nih.gov/account/ 申请).
"""
from __future__ import annotations

import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.environ.get("NCBI_API_KEY")  # optional but recommended

# Queries to run — tune to your needs. The product PMID lists are union'd.
QUERIES = {
    "TMF_PFAS": '("trophic magnification" OR TMF) AND (PFAS OR PFOS OR PFOA OR perfluor*)',
    "BAF_PFAS_field": '(bioaccumulation OR BAF OR "biota-sediment") AND (PFAS OR PFOS OR PFOA OR perfluor*) AND ("food web" OR field OR wild)',
    "BCF_PFAS_fish": '(BCF OR bioconcentration) AND (PFAS OR PFOS OR PFOA OR perfluor*) AND (fish OR carp OR trout OR zebrafish)',
    "PFAS_Arctic_foodweb": '(PFAS OR PFOS OR perfluor*) AND Arctic AND (food web OR trophic)',
    "PFAS_Baltic": '(PFAS OR PFOS OR perfluor*) AND Baltic AND (food web OR trophic OR biomagnification)',
    "PFAS_marine_mammal": '(PFAS OR PFOS OR perfluor*) AND (marine mammal OR cetacean OR seal OR dolphin OR whale)',
}


def esearch(query: str, retmax: int = 500) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    out = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        if API_KEY:
            params["api_key"] = API_KEY
        r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle") or ""
            journal = art.findtext(".//Journal/Title") or ""
            year = (art.findtext(".//PubDate/Year")
                    or art.findtext(".//PubDate/MedlineDate")
                    or "")[:4]
            abstract = " ".join((t.text or "") for t in art.findall(".//Abstract/AbstractText"))
            doi = ""
            for aid in art.findall(".//ArticleId"):
                if (aid.get("IdType") or "").lower() == "doi":
                    doi = aid.text or ""
            out.append({
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        time.sleep(0.4 if API_KEY else 0.4)
    return out


def init_lit_table(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS literature_candidates (
        pmid     TEXT PRIMARY KEY,
        title    TEXT,
        journal  TEXT,
        year     TEXT,
        abstract TEXT,
        doi      TEXT,
        url      TEXT,
        queries  TEXT,  -- which query labels brought this one in (comma-sep)
        reviewed INTEGER DEFAULT 0,
        relevance TEXT,           -- '5'..'1' set by you after reading abstract
        endpoint_seen TEXT,       -- 'TMF', 'BAF', etc. — set by you
        notes    TEXT
    );
    """)
    conn.commit()


def run(conn, data_dir: Path | None = None, queries: dict | None = None):
    queries = queries or QUERIES
    init_lit_table(conn)
    all_results: dict[str, dict] = {}
    pmid_to_queries: dict[str, set[str]] = {}

    for label, q in queries.items():
        print(f"[PubMed] query={label}: {q}")
        ids = esearch(q, retmax=500)
        print(f"[PubMed]   found {len(ids)} PMIDs")
        for pid in ids:
            pmid_to_queries.setdefault(pid, set()).add(label)
        records = efetch(ids)
        for r in records:
            all_results[r["pmid"]] = r

    n_new = 0
    for pmid, rec in all_results.items():
        cur = conn.execute(
            "INSERT OR IGNORE INTO literature_candidates (pmid, title, journal, year, abstract, doi, url, queries) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (pmid, rec["title"], rec["journal"], rec["year"], rec["abstract"],
             rec["doi"], rec["url"], ",".join(sorted(pmid_to_queries.get(pmid, [])))),
        )
        if cur.rowcount > 0:
            n_new += 1
    conn.commit()
    print(f"[PubMed] inserted {n_new} new literature candidates "
          f"({len(all_results) - n_new} already present)")
    return n_new


if __name__ == "__main__":
    from schema import init_db
    conn = init_db("data/pfas_master.sqlite")
    run(conn)
