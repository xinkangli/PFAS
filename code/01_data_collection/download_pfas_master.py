#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFAS Master Database Auto-Downloader  v2
==========================================
针对 v1 的失败做了三处根本性修复:

1) EPA CompTox:
   - 旧 dashboard-api 接口 (ccdapp1/ccdapp2) 已在 2024 年下线 -> 全部 404.
   - 新 API: https://api-ccte.epa.gov  需要免费 API key.
   - 申请: https://api-ccte.epa.gov/docs/ -> 点 "Request API Key"
   - 拿到后:  export EPA_CCTE_API_KEY=xxxxx  再跑本脚本.

2) PubChem:
   - 旧版用错了 classification 关键词, 只拿到 8 个 CID.
   - 改用 substructure SMARTS 搜索 'C(F)(F)F' (CF3 基团, OECD 2021 PFAS 核心定义).
   - 这个搜法一次能拿几千到上万个 PFAS CID.

3) OECD / NORMAN:
   - OECD 反爬严重, 官方 URL 也不稳; NORMAN 把下载迁到了 Zenodo.
   - 改成: 优先 Zenodo 镜像 -> 失败则给 fallback README.
   - 同时支持读取本地手动文件:
       把手动下载的文件放到 ./pfas_data/manual_input/  脚本会自动捡走.

依赖:
    pip install requests pandas openpyxl

使用:
    # (可选) 设置 EPA API key
    export EPA_CCTE_API_KEY=你的key

    python3 download_pfas_master_v2.py
"""

from __future__ import annotations   # Python 3.9 兼容: 让所有 `X | None` 类型注解延迟解析

import os
import re
import sys
import json
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set

import requests
import pandas as pd


# ============================================================
# 配置
# ============================================================
BASE_DIR     = Path("./pfas_data")
MANUAL_DIR   = BASE_DIR / "manual_input"   # 用户手动放置的文件目录
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR      = BASE_DIR / TIMESTAMP

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_DIR.mkdir(parents=True, exist_ok=True)

EPA_API_KEY = os.environ.get("EPA_CCTE_API_KEY", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "download.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pfas")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,zh;q=0.8",
}


def safe_get(url, *, params=None, headers=None, timeout=120, retries=3, stream=False):
    """GET with retries."""
    h = dict(HEADERS)
    if headers:
        h.update(headers)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=h,
                             timeout=timeout, stream=stream,
                             allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning(f"  [{attempt}/{retries}] GET 失败 {url}\n    -> {e}")
            time.sleep(2 * attempt)
    return None


def safe_post(url, *, data=None, json_body=None, headers=None, timeout=120, retries=3):
    h = dict(HEADERS)
    if headers:
        h.update(headers)
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, data=data, json=json_body,
                              headers=h, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning(f"  [{attempt}/{retries}] POST 失败 {url}\n    -> {e}")
            time.sleep(2 * attempt)
    return None


def save_df(df: pd.DataFrame, stem: Path):
    csv_path = stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(stem.with_suffix(".xlsx"), index=False)
    except Exception as e:
        log.warning(f"  xlsx 写入失败 (csv 已保存): {e}")
    log.info(f"  保存表: {csv_path}  ({len(df)} rows)")


def pick_manual_file(*patterns) -> Path | None:
    """在 manual_input 目录里按文件名子串匹配, 找到第一个就返回."""
    if not MANUAL_DIR.exists():
        return None
    for f in MANUAL_DIR.iterdir():
        if not f.is_file():
            continue
        name = f.name.lower()
        for p in patterns:
            if p.lower() in name:
                return f
    return None


# ============================================================
# 1) EPA CompTox - 新版 CCTE API
# ============================================================
def download_epa_comptox() -> bool:
    log.info("=" * 70)
    log.info("[1/4] EPA CompTox - 新版 CCTE API")
    log.info("=" * 70)

    # Step 0: 手动文件优先
    manual = pick_manual_file("pfasmaster", "comptox", "dsstox", "ccte")
    if manual:
        dst = OUT_DIR / f"01_EPA_PFASMASTER_{manual.name}"
        shutil.copy(manual, dst)
        log.info(f"  使用本地手动文件: {manual.name} -> {dst.name}")
        try:
            if manual.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(manual)
            else:
                df = pd.read_csv(manual, low_memory=False)
            save_df(df, OUT_DIR / "01_EPA_PFASMASTER_chemicals")
            return True
        except Exception as e:
            log.warning(f"  手动文件解析失败但已复制: {e}")
            return True

    # Step 1: API key
    if not EPA_API_KEY:
        log.warning("  未检测到 EPA_CCTE_API_KEY 环境变量")
        (OUT_DIR / "01_EPA_FALLBACK_README.txt").write_text(
            "EPA CompTox 自动下载需要 API key (旧端点已下线).\n\n"
            "拿 key (免费, 30 秒):\n"
            "  https://api-ccte.epa.gov/docs/  -> 点 'Request API Key'\n"
            "  邮箱填一下马上发到邮箱.\n\n"
            "拿到后:\n"
            "  export EPA_CCTE_API_KEY='你的key'\n"
            "  python3 download_pfas_master_v2.py\n\n"
            "或者直接手动下载:\n"
            "  1) https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER\n"
            "  2) 页面右上角下载按钮 -> CSV\n"
            "  3) 把文件放到 ./pfas_data/manual_input/  (文件名包含 'pfasmaster' 即可)\n"
            "  4) 重新跑脚本, 会自动识别.\n",
            encoding="utf-8",
        )
        return False

    log.info("  检测到 API key, 调用新版 CCTE API")
    api_headers = {"x-api-key": EPA_API_KEY, "Accept": "application/json"}

    # PFASMASTER 列表的化学品明细
    url = "https://api-ccte.epa.gov/chemical/list/chemicals/PFASMASTER"
    r = safe_get(url, headers=api_headers, timeout=180)
    if r is None:
        log.error("  CCTE API 调用失败")
        return False

    try:
        data = r.json()
    except Exception as e:
        log.error(f"  CCTE API 返回非 JSON: {e}")
        return False

    # 返回是 DTXSID 列表; 再批量取 detail
    if isinstance(data, list) and data and isinstance(data[0], str):
        dtxsids = data
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        dtxsids = [x.get("dtxsid") for x in data if x.get("dtxsid")]
    else:
        dtxsids = []

    log.info(f"  PFASMASTER 含 {len(dtxsids)} 个 DTXSID")

    # 批量详情 (CCTE API 支持 POST 批量)
    detail_url = "https://api-ccte.epa.gov/chemical/detail/search/by-dtxsid/"
    rows = []
    BATCH = 200
    for i in range(0, len(dtxsids), BATCH):
        batch = dtxsids[i:i + BATCH]
        r = safe_post(detail_url, json_body=batch, headers=api_headers, timeout=180)
        if r is None:
            continue
        try:
            chunk = r.json()
            if isinstance(chunk, list):
                rows.extend(chunk)
            log.info(f"  batch {i // BATCH + 1}: +{len(chunk)} (累计 {len(rows)})")
        except Exception as e:
            log.warning(f"  batch 解析失败: {e}")
        time.sleep(0.3)

    if not rows:
        log.error("  EPA detail 全部失败")
        return False

    df = pd.json_normalize(rows)
    save_df(df, OUT_DIR / "01_EPA_PFASMASTER_chemicals")
    return True


# ============================================================
# 2) OECD PFAS list
# ============================================================
def download_oecd() -> bool:
    log.info("=" * 70)
    log.info("[2/4] OECD PFAS List")
    log.info("=" * 70)

    # Step 0: 手动文件
    manual = pick_manual_file("oecd")
    if manual:
        dst = OUT_DIR / f"02_OECD_{manual.name}"
        shutil.copy(manual, dst)
        log.info(f"  使用本地手动文件: {manual.name} -> {dst.name}")
        return True

    # Step 1: 试 Zenodo / 学术镜像
    # OECD 2018 PFAS Annex 在多个学术镜像有备份, 这里列几个相对稳定的:
    candidates = [
        # ECHEMPORTAL / DOI mirror
        ("02_OECD_2018_PFAS_Annex.xlsx",
         "https://doi.org/10.1021/acs.est.7b06487",   # Wang 2017 ESTL (含OECD分类前身)
         False),  # DOI 解析后可能跳ACS, 不一定能下 xlsx, 留作参考
        # NIST / Wikipedia commons mirror (有时有)
        ("02_OECD_Annex.xlsx",
         "https://zenodo.org/records/6346851/files/OECD_2018_PFAS_Annex.xlsx",
         True),
        # 另一种已知的学术再分发
        ("02_OECD_2021_PFAS_definition.pdf",
         "https://one.oecd.org/document/ENV/CBC/MONO(2021)25/en/pdf",
         True),
    ]

    ok = False
    for fname, url, attempt in candidates:
        if not attempt:
            continue
        log.info(f"  尝试: {url}")
        r = safe_get(url, timeout=180)
        if r is None:
            continue
        content = r.content
        # 检查是否真的是文件 (不是 HTML 错误页)
        if len(content) < 4096:
            log.warning(f"  返回过小 ({len(content)} B), 可能是错误页")
            continue
        if content.lstrip().startswith(b"<!DOCTYPE") or content.lstrip().startswith(b"<html"):
            log.warning("  返回是 HTML 而非二进制文件")
            continue
        (OUT_DIR / fname).write_bytes(content)
        log.info(f"  写入: {fname} ({len(content):,} B)")
        ok = True

    if not ok:
        (OUT_DIR / "02_OECD_FALLBACK_README.txt").write_text(
            "OECD PFAS list 自动下载失败 (OECD 网站反爬严格).\n\n"
            "手动下载 (10 分钟):\n"
            "  1) 2018 PFAS 全球数据库 (含分类 Annex Excel):\n"
            "     https://www.oecd.org/en/publications/2018/05/toward-a-new-comprehensive-global-database-of-per-and-polyfluoroalkyl-substances-pfass_5b134d75.html\n"
            "     页面右侧 'Read online or download' -> 下 Annex.xlsx\n\n"
            "  2) 2021 PFAS 重新定义 (ENV/CBC/MONO(2021)25):\n"
            "     https://one.oecd.org/document/ENV/CBC/MONO(2021)25/en/pdf\n"
            "     如果浏览器能下到 PDF, 命令行经常被 403 拦.\n\n"
            "  3) 学术再分发 (经常更新, 自己 Google '\\\"OECD PFAS list\\\" filetype:xlsx site:zenodo.org'):\n"
            "     https://zenodo.org/search?q=OECD+PFAS\n\n"
            "下载到的文件丢到这里:\n"
            f"     {MANUAL_DIR.resolve()}/\n"
            "文件名含 'oecd' 即可被脚本自动识别.\n",
            encoding="utf-8",
        )
        log.error("  OECD 失败, fallback 已写出")
    return ok


# ============================================================
# 3) PubChem - 用 SMARTS 子结构搜索 (核心改进)
# ============================================================
def _pubchem_smarts_cids(smarts: str, max_records: int = 20000) -> list[str]:
    """
    用 PubChem PUG REST fastsubstructure 接口搜 SMARTS, 返回 CID 列表.
    可能返回 listkey 需要轮询.
    """
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           "fastsubstructure/smarts/cids/JSON")
    log.info(f"  PubChem SMARTS 搜索: {smarts}")
    r = safe_post(
        url,
        data={"smarts": smarts, "MaxRecords": str(max_records)},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=300,
    )
    if r is None:
        return []

    try:
        j = r.json()
    except Exception:
        return []

    # Case A: 直接返回 CID 列表
    if "IdentifierList" in j:
        cids = j["IdentifierList"].get("CID", [])
        return [str(c) for c in cids]

    # Case B: 异步, 给了 ListKey, 需要轮询
    waiting = j.get("Waiting") or j.get("ResponseStatus")
    if isinstance(waiting, dict) and "ListKey" in waiting:
        listkey = waiting["ListKey"]
        log.info(f"  PubChem 进入异步, ListKey={listkey}")
        poll_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"listkey/{listkey}/cids/JSON"
        )
        for attempt in range(20):
            time.sleep(3)
            r2 = safe_get(poll_url, timeout=120, retries=1)
            if r2 is None:
                continue
            try:
                j2 = r2.json()
            except Exception:
                continue
            if "IdentifierList" in j2:
                return [str(c) for c in j2["IdentifierList"].get("CID", [])]
            log.info(f"  轮询中 ({attempt + 1}/20)...")
        log.warning("  PubChem 异步超时")
    return []


def _pubchem_fetch_properties(cids: list[str]) -> list[dict]:
    """批量取属性."""
    props = ("CanonicalSMILES,IsomericSMILES,InChI,InChIKey,"
             "MolecularFormula,MolecularWeight,IUPACName,XLogP")
    rows = []
    BATCH = 100
    n = (len(cids) + BATCH - 1) // BATCH
    for i in range(0, len(cids), BATCH):
        batch = cids[i:i + BATCH]
        # POST 比 GET 安全 (CID 多了 URL 会超长)
        r = safe_post(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"property/{props}/JSON",
            data={"cid": ",".join(batch)},
            timeout=180,
        )
        if r is None:
            continue
        try:
            chunk = r.json().get("PropertyTable", {}).get("Properties", [])
            rows.extend(chunk)
            log.info(f"  属性 batch {i // BATCH + 1}/{n}: "
                     f"+{len(chunk)} (累计 {len(rows)})")
        except Exception as e:
            log.warning(f"  属性 batch 解析失败: {e}")
        time.sleep(0.35)
    return rows


def download_pubchem() -> bool:
    log.info("=" * 70)
    log.info("[3/4] PubChem - SMARTS substructure search")
    log.info("=" * 70)

    # 多个 SMARTS 取并集, 覆盖 OECD 2021 PFAS 核心定义:
    #   含至少一个 CF3 (perfluoromethyl) 或 -CF2-CF2- (perfluoromethylene 链)
    smarts_list = [
        ("CF3 group",        "C(F)(F)F"),
        ("perfluoro CF2-CF2","FC(F)C(F)F"),
    ]

    all_cids: set[str] = set()
    for label, smarts in smarts_list:
        log.info(f"  策略: {label}")
        cids = _pubchem_smarts_cids(smarts, max_records=20000)
        log.info(f"    -> {len(cids)} CIDs")
        all_cids.update(cids)

    # 兜底: 一批已知 PFAS 名字 (扩展版)
    if len(all_cids) < 100:
        log.info("  策略: 已知 PFAS 名字兜底")
        seed_names = [
            "perfluorooctanoic acid", "perfluorooctanesulfonic acid",
            "perfluorobutanoic acid", "perfluorobutanesulfonic acid",
            "perfluorohexanoic acid", "perfluorohexanesulfonic acid",
            "perfluorononanoic acid", "perfluorodecanoic acid",
            "perfluoropentanoic acid", "perfluoroheptanoic acid",
            "GenX", "HFPO-DA", "ADONA", "F-53B",
            "8:2 fluorotelomer alcohol", "6:2 fluorotelomer alcohol",
            "perfluorooctane sulfonamide", "N-ethyl perfluorooctane sulfonamide",
            "hexafluoropropylene oxide dimer acid",
        ]
        for nm in seed_names:
            r = safe_get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"name/{requests.utils.quote(nm)}/cids/JSON"
            )
            if r is None:
                continue
            try:
                for c in r.json().get("IdentifierList", {}).get("CID", []):
                    all_cids.add(str(c))
            except Exception:
                pass
            time.sleep(0.3)

    cids = sorted(all_cids, key=lambda x: int(x))
    log.info(f"  PubChem 合计 PFAS-like CIDs: {len(cids)}")

    if not cids:
        log.error("  PubChem 失败")
        return False

    (OUT_DIR / "03_PubChem_cids.txt").write_text(
        "\n".join(cids), encoding="utf-8"
    )

    rows = _pubchem_fetch_properties(cids)
    if not rows:
        log.error("  PubChem 属性接口全部失败")
        return False

    df = pd.DataFrame(rows)
    save_df(df, OUT_DIR / "03_PubChem_PFAS_properties")
    return True


# ============================================================
# 4) NORMAN SusDat - 走 Zenodo
# ============================================================
def download_norman_susdat() -> bool:
    log.info("=" * 70)
    log.info("[4/4] NORMAN SusDat")
    log.info("=" * 70)

    # 手动文件
    manual = pick_manual_file("norman", "susdat", "s25")
    if manual:
        dst = OUT_DIR / f"04_NORMAN_{manual.name}"
        shutil.copy(manual, dst)
        log.info(f"  使用本地手动文件: {manual.name} -> {dst.name}")
        return True

    # NORMAN 现在主要走 Zenodo 发布. 用 Zenodo API 找最新的 SusDat / S25.
    log.info("  通过 Zenodo API 搜索 NORMAN 最新 PFAS 清单")
    zenodo_search = "https://zenodo.org/api/records"
    queries = [
        'NORMAN SusDat',
        'NORMAN PFAS S25',
        'NORMAN suspect list PFAS',
    ]

    found = []
    for q in queries:
        r = safe_get(zenodo_search, params={
            "q": q, "size": 10, "sort": "mostrecent"
        }, timeout=60)
        if r is None:
            continue
        try:
            hits = r.json().get("hits", {}).get("hits", [])
        except Exception:
            continue
        for h in hits:
            for f in (h.get("files") or []):
                link = (f.get("links") or {}).get("self")
                fname = f.get("key", "")
                if not link or not fname:
                    continue
                if not any(fname.lower().endswith(ext) for ext in
                           (".xlsx", ".csv", ".xls", ".zip", ".tsv")):
                    continue
                found.append((h.get("metadata", {}).get("title", "?"),
                              fname, link))

    if not found:
        log.warning("  Zenodo 未找到匹配的 NORMAN 文件")

    success = False
    seen = set()
    for title, fname, link in found[:8]:
        if fname in seen:
            continue
        seen.add(fname)
        log.info(f"  Zenodo 下载: {fname}  ({title[:60]})")
        r = safe_get(link, timeout=300)
        if r is None or len(r.content) < 4096:
            continue
        out = OUT_DIR / f"04_NORMAN_{fname}"
        out.write_bytes(r.content)
        log.info(f"  写入: {out.name} ({len(r.content):,} B)")
        success = True

    if not success:
        (OUT_DIR / "04_NORMAN_FALLBACK_README.txt").write_text(
            "NORMAN SusDat 自动下载失败.\n\n"
            "手动下载 (推荐 Zenodo, 比官网稳):\n"
            "  1) Zenodo 搜:\n"
            "     https://zenodo.org/search?q=NORMAN+SusDat\n"
            "     https://zenodo.org/search?q=NORMAN+PFAS+S25\n"
            "  2) 官方页面 (有时反爬):\n"
            "     https://www.norman-network.com/nds/susdat/\n"
            "  3) 下到的 xlsx / csv 放到:\n"
            f"     {MANUAL_DIR.resolve()}/\n"
            "  文件名含 'norman' 或 'susdat' 或 's25' 即可被自动识别.\n\n"
            "PFAS 相关清单 ID:\n"
            "  S25 - NORMAN PFAS suspect list\n"
            "  S65 - PFAS in food contact materials\n"
            "  S69 - PFAS environmental screening\n",
            encoding="utf-8",
        )
        log.error("  NORMAN 失败, fallback 已写出")
    return success


# ============================================================
# 主流程
# ============================================================
def main():
    log.info("PFAS Master Auto-Downloader v2 启动")
    log.info(f"输出目录: {OUT_DIR.resolve()}")
    log.info(f"手动输入目录: {MANUAL_DIR.resolve()}")
    if EPA_API_KEY:
        log.info(f"EPA API key: {'*' * 6}{EPA_API_KEY[-4:]} (已检测)")
    else:
        log.info("EPA API key: 未设置 (export EPA_CCTE_API_KEY=xxx)")

    tasks = [
        ("EPA CompTox (PFASMASTER)", download_epa_comptox),
        ("OECD PFAS list",           download_oecd),
        ("PubChem PFAS (SMARTS)",    download_pubchem),
        ("NORMAN SusDat",            download_norman_susdat),
    ]

    results = {}
    for name, fn in tasks:
        try:
            results[name] = fn()
        except Exception as e:
            log.exception(f"{name} 异常: {e}")
            results[name] = False

    log.info("=" * 70)
    log.info("汇总:")
    for k, v in results.items():
        log.info(f"  {'✅ OK ' if v else '❌ FAIL'}  {k}")
    log.info(f"输出: {OUT_DIR.resolve()}")

    (OUT_DIR / "README.txt").write_text(
        "PFAS Master 数据汇总\n"
        f"时间: {TIMESTAMP}\n\n"
        + "\n".join(f"  [{'OK' if v else 'FAIL'}]  {k}" for k, v in results.items())
        + "\n\n所有 FAIL 项的 fallback README 见同目录下 *_FALLBACK_README.txt\n"
          "手动下载好的文件放到:\n"
          f"  {MANUAL_DIR.resolve()}/\n"
          "再重新跑脚本会自动识别 (无需删除已有目录).\n\n"
          "关联主键推荐: InChIKey > CASRN > DTXSID  (不要用 Name)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()