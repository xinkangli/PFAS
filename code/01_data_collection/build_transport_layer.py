#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_transport_layer.py
========================
构建 PFAS 第三层 Transport 数据层。

输入:
  --structure   PFAS_master_structure.xlsx       第一层结构主库
  --tmf         extraction_tables_v4.xlsx        第二层 TMF/浓度数据
  --out         PFAS_transport_layer_v1.xlsx     输出路径

输出 (4 sheets):
  1. target_pfas_list        第二层出现过、需优先补 transport 的 PFAS
  2. transport_template_long 长表：一条 PFAS × endpoint 一行
  3. transport_template_wide 宽表：一条 PFAS 一行，endpoint 做列
  4. missing_priority        缺失矩阵，按第二层出现频率降序

依赖:
    pip install pandas openpyxl
"""

from __future__ import annotations

import re
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("transport")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── 目标 endpoints ────────────────────────────────────────────────────────

ENDPOINTS = [
    "logKow",
    "logKoc",
    "logKd",
    "water_solubility",
    "Henry_constant",
    "vapor_pressure",
    "pKa",
    "logD",
]

# endpoint → 推荐单位
ENDPOINT_UNITS = {
    "logKow":           "dimensionless (log L/L)",
    "logKoc":           "dimensionless (log L/kg OC)",
    "logKd":            "dimensionless (log L/kg)",
    "water_solubility": "mg/L",
    "Henry_constant":   "Pa·m³/mol",
    "vapor_pressure":   "Pa",
    "pKa":              "dimensionless",
    "logD":             "dimensionless (log L/L)",
}

# 长表固定列顺序
LONG_COLS = [
    "pfas_name", "CAS", "InChIKey", "DTXSID", "pfas_class",
    "endpoint", "value", "unit",
    "temperature", "pH",
    "source_database", "source_reference",
    "measured_or_predicted", "reliability", "notes",
]

# ─── 第一层列名规范化 ──────────────────────────────────────────────────────

# 常见别名 → 统一内部名
STRUCT_COL_MAP = {
    "name":      ["name", "preferred name", "preferredname", "chemical name",
                  "compound name", "pfas name", "pfas_name",
                  "iupac name", "iupacname"],
    "CAS":       ["cas", "casrn", "cas rn", "cas number", "cas_rn", "casnumber"],
    "InChIKey":  ["inchikey", "inchi key", "std_inchikey", "stdinchikey"],
    "DTXSID":    ["dtxsid", "dsstox_substance_id", "dsstox substance id",
                  "dtx_substance_id"],
    "SMILES":    ["smiles", "canonical_smiles", "canonical smiles",
                  "isomeric_smiles", "isomericsmiles"],
    "InChI":     ["inchi", "std_inchi", "stdinchi"],
    "MolFormula":["molecularformula", "molecular formula", "molformula"],
    "pfas_class":["pfas class", "pfasclass", "class", "pfas_class",
                  "subclass", "pfas type"],
    "XLogP":     ["xlogp", "xlogp3", "logp", "logkow", "log kow",
                  "log_kow", "xlogp3-aa"],
}

def normalize_struct_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for std, aliases in STRUCT_COL_MAP.items():
        if std in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = std
                break
    if rename:
        log.info(f"  第一层列名映射: {rename}")
        df = df.rename(columns=rename)
    return df


# ─── 读取第一层结构主库 ───────────────────────────────────────────────────

def load_structure_db(path: Path) -> pd.DataFrame:
    log.info(f"[第一层] 读取结构主库: {path.name}")
    if not path.exists():
        log.warning(f"  文件不存在: {path}，将使用空表继续")
        return pd.DataFrame()

    if path.suffix.lower() in (".xlsx", ".xls"):
        xf = pd.ExcelFile(path)
        # 优先找含 'pfas' 或 'master' 或 'structure' 的 sheet
        preferred = [s for s in xf.sheet_names
                     if any(kw in s.lower() for kw in ["pfas","master","structure","chemical"])]
        sheet = preferred[0] if preferred else xf.sheet_names[0]
        log.info(f"  读取 sheet: '{sheet}'")
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法解码: {path}")

    df = normalize_struct_cols(df)
    log.info(f"  第一层: {len(df)} 行, 列: {list(df.columns)}")
    return df


# ─── 读取第二层 TMF 数据 ──────────────────────────────────────────────────


# 汇总标记：不是真实单一 PFAS 化合物，不进入 transport 层
_EXCLUDE_MARKERS = {
    "σpfas", "σpfcs", "σpfsa", "σpfca",
    "sum pfas", "total pfcs", "total pfas",
    "17 pfas", "tbd", "—", "-", "n/a", "na",
    "σpfas (sum)", "pfas sum",
}

def _is_real_pfas(name: str) -> bool:
    """排除汇总标记和空值"""
    n = name.strip().lower()
    if not n or n in _EXCLUDE_MARKERS:
        return False
    # 以 Σ 或 sigma 开头的都是汇总
    if n.startswith("σ") or n.startswith("∑") or n.startswith("sum"):
        return False
    # 纯数字或破折号
    if n in ("—", "–", "-", ""):
        return False
    return True

def load_tmf_pfas(path: Path) -> pd.DataFrame:
    """
    从 extraction_tables_v4.xlsx 的 Table2 + Table3 中
    收集所有出现过的 PFAS 名称及其出现频率。
    返回 DataFrame: pfas_name, freq_table2, freq_table3, total_freq
    """
    log.info(f"[第二层] 读取 TMF 数据: {path.name}")
    if not path.exists():
        log.warning(f"  文件不存在: {path}，将使用示例 PFAS 列表继续")
        return _demo_pfas_list()

    xf = pd.ExcelFile(path)
    sheet_names_lower = {s.lower(): s for s in xf.sheet_names}

    pfas_counter: dict[str, dict] = {}  # pfas_name -> {freq_table2, freq_table3}

    # Table2: concentration
    t2_key = next((v for k, v in sheet_names_lower.items()
                   if "table2" in k or "concentration" in k or "conc" in k), None)
    if t2_key:
        t2 = pd.read_excel(path, sheet_name=t2_key, dtype=str)
        log.info(f"  Table2 ({t2_key}): {len(t2)} 行, 列: {list(t2.columns)}")
        # 找 PFAS 列
        pfas_col = _find_pfas_col(t2)
        if pfas_col:
            for name in t2[pfas_col].dropna():
                name = str(name).strip()
                if name and _is_real_pfas(name):
                    pfas_counter.setdefault(name, {"freq_table2": 0, "freq_table3": 0})
                    pfas_counter[name]["freq_table2"] += 1
        else:
            log.warning(f"  Table2: 未找到 PFAS 列，列名: {list(t2.columns)}")
    else:
        log.warning("  未找到 Table2/concentration sheet")

    # Table3: TMF
    t3_key = next((v for k, v in sheet_names_lower.items()
                   if "table3" in k or "tmf" in k), None)
    if t3_key:
        t3 = pd.read_excel(path, sheet_name=t3_key, dtype=str)
        log.info(f"  Table3 ({t3_key}): {len(t3)} 行, 列: {list(t3.columns)}")
        pfas_col = _find_pfas_col(t3)
        if pfas_col:
            for name in t3[pfas_col].dropna():
                name = str(name).strip()
                if name and _is_real_pfas(name):
                    pfas_counter.setdefault(name, {"freq_table2": 0, "freq_table3": 0})
                    pfas_counter[name]["freq_table3"] += 1
        else:
            log.warning(f"  Table3: 未找到 PFAS 列，列名: {list(t3.columns)}")
    else:
        log.warning("  未找到 Table3/TMF sheet")

    if not pfas_counter:
        log.warning("  第二层未提取到任何 PFAS,使用示例列表")
        return _demo_pfas_list()

    # 大小写 / 别名归一:同 CAS 的不同写法合并 frequency
    log.info("  跑 abbrev_to_CAS 大小写归一...")
    cas_groups: dict[str, dict] = {}   # CAS -> {names:{n:freq2},...freq2,freq3}
    leftover: dict[str, dict] = {}     # 没缩写映射的保留原名
    for raw_name, freqs in pfas_counter.items():
        cas = _abbrev_to_cas(raw_name)
        if cas:
            g = cas_groups.setdefault(cas, {"freq_table2": 0, "freq_table3": 0,
                                            "aliases": set()})
            g["freq_table2"] += freqs["freq_table2"]
            g["freq_table3"] += freqs["freq_table3"]
            g["aliases"].add(raw_name)
        else:
            leftover[raw_name] = freqs

    # CAS 组用最高频的写法做主名
    rows = []
    for cas, g in cas_groups.items():
        # 选频次最高的别名(并列时取最短)
        names_sorted = sorted(g["aliases"],
                              key=lambda n: (-pfas_counter[n]["freq_table2"]
                                             - pfas_counter[n]["freq_table3"],
                                             len(n)))
        primary = names_sorted[0]
        all_aliases = "|".join(sorted(g["aliases"])) if len(g["aliases"]) > 1 else ""
        rows.append({
            "pfas_name":     primary,
            "freq_table2":   g["freq_table2"],
            "freq_table3":   g["freq_table3"],
            "total_freq":    g["freq_table2"] + g["freq_table3"],
            "aliases":       all_aliases,
        })
    for name, freqs in leftover.items():
        rows.append({
            "pfas_name":     name,
            "freq_table2":   freqs["freq_table2"],
            "freq_table3":   freqs["freq_table3"],
            "total_freq":    freqs["freq_table2"] + freqs["freq_table3"],
            "aliases":       "",
        })

    df = pd.DataFrame(rows).sort_values("total_freq", ascending=False).reset_index(drop=True)
    n_merged = sum(1 for r in rows if r["aliases"])
    log.info(f"  第二层提取 PFAS: {len(df)} 种(其中 {n_merged} 条合并自别名)")
    return df


def _find_pfas_col(df: pd.DataFrame) -> Optional[str]:
    """在 DataFrame 中找 PFAS 名称列（优先 'pfas', 'compound', 'chemical'）"""
    candidates = ["pfas", "pfas_name", "compound", "chemical", "substance",
                  "pfas_unified_name", "analyte", "congener"]
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # fallback: 找含 'pfas' 的列名
    for c in df.columns:
        if "pfas" in c.lower():
            return c
    return None


def _demo_pfas_list() -> pd.DataFrame:
    """当文件不存在时的示例 PFAS 列表"""
    demo = [
        ("PFOS",   45, 12),
        ("PFOA",   38, 10),
        ("PFHxS",  22,  8),
        ("PFNA",   18,  7),
        ("PFDA",   15,  6),
        ("PFUnDA", 12,  5),
        ("PFDoDA",  8,  3),
        ("PFBS",    7,  2),
        ("PFHxA",   6,  2),
        ("PFBA",    5,  1),
        ("6:2 FTOH", 9, 4),
        ("8:2 FTOH", 6, 3),
        ("HFPO-DA (GenX)", 4, 2),
        ("ADONA",   3,  1),
        ("N-EtFOSA", 4, 2),
    ]
    rows = [{"pfas_name": n, "freq_table2": f2, "freq_table3": f3,
             "total_freq": f2 + f3} for n, f2, f3 in demo]
    return pd.DataFrame(rows)


# ─── 名称标准化（用于模糊匹配）────────────────────────────────────────────

# 常见 PFAS 缩写 → CAS 号（直接跳过名称匹配）
PFAS_ABBREV_CAS = {
    "pfos":         "1763-23-1",
    "pfoa":         "335-67-1",
    "pfhxs":        "355-46-4",
    "pfna":         "375-95-1",
    "pfda":         "335-76-2",
    "pfunda":       "2058-94-8",
    "pfdoda":       "307-55-1",
    "pftrda":       "72629-94-8",
    "pfteda":       "376-06-7",
    "pfba":         "375-22-4",
    "pfbs":         "375-73-5",
    "pfhxa":        "307-24-4",
    "pfhpa":        "375-85-9",
    "pfopa":        "335-67-1",
    "pfpea":        "2706-90-3",
    "6:2 ftoh":     "647-42-7",
    "62ftoh":       "647-42-7",
    "8:2 ftoh":     "678-39-7",
    "82ftoh":       "678-39-7",
    "n-etfosa":     "4151-50-2",
    "netfosa":      "4151-50-2",
    "n-mefosa":     "31506-32-8",
    "nmefosa":      "31506-32-8",
    "fosa":         "754-91-6",
    "pfosa":        "754-91-6",
    "hfpo-da":      "13252-13-6",
    "genx":         "13252-13-6",
    "adona":        "958445-44-8",
    "f-53b":        "73606-19-6",
    "pfechs":       "757124-72-4",
    "pfmoaa":       "2043-47-2",
    "pf3ons":       "2706-91-4",
    # ── 补充第二批（来自 extraction_tables_v4 未匹配名单）
    "pfhps":        "375-92-8",     # perfluoroheptanesulfonic acid
    "pfhpsa":       "375-92-8",
    "fbsa":         "30334-69-1",   # perfluorobutanesulfonamide
    "pfds":         "335-77-3",     # perfluorodecane sulfonic acid
    "fhxsa":        "41997-13-1",   # perfluorohexanesulfonamide
    "etfosaa":      "2991-50-6",    # N-EtFOSAA
    "n-etfosaa":    "2991-50-6",
    "pfuna":        "2058-94-8",    # = PFUnDA
    "pfunda":       "2058-94-8",
    "6:2 ftsa":     "27619-97-2",   # 6:2 fluorotelomer sulfonic acid
    "62ftsa":       "27619-97-2",
    "6:2 fts":      "27619-97-2",   # same compound, alternate name
    "62fts":        "27619-97-2",
    "8:2 ftsa":     "39108-34-4",   # 8:2 fluorotelomer sulfonic acid
    "82ftsa":       "39108-34-4",
    "6:2 dipap":    "57678-51-4",   # 6:2 disubstituted polyfluoroalkyl phosphate
    "62dipap":      "57678-51-4",
    "pfdoa":        "307-55-1",     # = PFDoDA
    "pfdoda":       "307-55-1",
    "pfta":         "376-06-7",     # perfluorotetradecanoic acid = PFTeDA
    "pfteda":       "376-06-7",
    "pfpes":        "2706-91-4",    # perfluoropentanesulfonic acid
    "pfpents":      "2706-91-4",
}

def _norm_name(s: str) -> str:
    """小写 + 去除空白/连字符/括号/前缀"""
    s = str(s or "").lower().strip()
    s = re.sub(r"[\s\-_/\(\)]", "", s)
    s = s.replace("perfluoro", "pf").replace("acid", "")
    s = s.replace("sulfonic", "s").replace("carboxylic", "c")
    return s

def _abbrev_to_cas(name: str) -> str:
    """尝试把缩写转为 CAS"""
    key = name.lower().strip()
    if key in PFAS_ABBREV_CAS:
        return PFAS_ABBREV_CAS[key]
    # 去掉括号和空格再试
    key2 = re.sub(r"[\s\(\)\-]", "", key)
    return PFAS_ABBREV_CAS.get(key2, "")


# ─── 第一层 × 第二层匹配 ─────────────────────────────────────────────────

def match_pfas(struct_df: pd.DataFrame, pfas_list: pd.DataFrame) -> pd.DataFrame:
    """
    将第二层 PFAS 列表与第一层结构库匹配。
    匹配策略（按优先级）：
      1. InChIKey 精确匹配
      2. CAS 精确匹配
      3. DTXSID 精确匹配
      4. Name 精确匹配（大小写不敏感）
      5. Name 模糊匹配（标准化后）
    返回 pfas_list 增加第一层字段的 DataFrame。
    """
    if struct_df.empty:
        log.warning("  第一层结构库为空，跳过匹配")
        for col in ["CAS", "InChIKey", "DTXSID", "SMILES", "MolFormula",
                    "pfas_class", "XLogP", "match_method"]:
            pfas_list[col] = ""
        return pfas_list

    log.info(f"[匹配] 第二层 {len(pfas_list)} 种 PFAS × 第一层 {len(struct_df)} 条")

    # 构建查找索引
    def build_index(col):
        if col not in struct_df.columns:
            return {}
        return {str(v).strip().lower(): i
                for i, v in struct_df[col].items()
                if pd.notna(v) and str(v).strip()}

    idx_cas      = build_index("CAS")
    idx_inchikey = build_index("InChIKey")
    idx_dtxsid   = build_index("DTXSID")
    idx_name     = build_index("name") if "name" in struct_df.columns else {}
    idx_name_norm = {_norm_name(k): v for k, v in idx_name.items()}

    result_rows = []
    matched = 0

    for _, row in pfas_list.iterrows():
        pfas_name = str(row["pfas_name"]).strip()
        struct_idx = None
        method = "unmatched"

        name_lower = pfas_name.lower()

        # 策略 0: 缩写 → CAS 硬编码表
        abbrev_cas = _abbrev_to_cas(pfas_name)
        if abbrev_cas and abbrev_cas.lower() in idx_cas:
            struct_idx = idx_cas[abbrev_cas.lower()]; method = "abbrev_to_CAS"
        # 策略 1: InChIKey 精确
        elif name_lower in idx_inchikey:
            struct_idx = idx_inchikey[name_lower]; method = "InChIKey_exact"
        # 策略 2: CAS 精确
        elif name_lower in idx_cas:
            struct_idx = idx_cas[name_lower]; method = "CAS_exact"
        # 策略 3: DTXSID 精确
        elif name_lower in idx_dtxsid:
            struct_idx = idx_dtxsid[name_lower]; method = "DTXSID_exact"
        # 策略 4: Name 精确
        elif name_lower in idx_name:
            struct_idx = idx_name[name_lower]; method = "name_exact"
        else:
            # 策略 5: 模糊名称匹配
            norm = _norm_name(pfas_name)
            if norm in idx_name_norm:
                struct_idx = idx_name_norm[norm]; method = "name_fuzzy"

        if struct_idx is not None:
            srow = struct_df.loc[struct_idx]
            matched += 1
            result_rows.append({
                **row.to_dict(),
                "CAS":         srow.get("CAS", ""),
                "InChIKey":    srow.get("InChIKey", ""),
                "DTXSID":      srow.get("DTXSID", ""),
                "SMILES":      srow.get("SMILES", ""),
                "MolFormula":  srow.get("MolFormula", ""),
                "pfas_class":  srow.get("pfas_class", ""),
                "XLogP":       srow.get("XLogP", ""),
                "match_method": method,
            })
        else:
            result_rows.append({
                **row.to_dict(),
                "CAS": "", "InChIKey": "", "DTXSID": "",
                "SMILES": "", "MolFormula": "", "pfas_class": "",
                "XLogP": "", "match_method": "unmatched",
            })

    log.info(f"  匹配成功: {matched}/{len(pfas_list)} "
             f"({100*matched/max(len(pfas_list),1):.1f}%)")
    unmatched = [r["pfas_name"] for r in result_rows if r.get("match_method") == "unmatched"]
    if unmatched:
        log.warning(f"  未匹配 {len(unmatched)} 种: {unmatched}")
        log.warning("  提示: 将未匹配的缩写加入 PFAS_ABBREV_CAS 字典即可")
    return pd.DataFrame(result_rows)


# ─── 构建长表 ─────────────────────────────────────────────────────────────

def build_long_table(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个 PFAS × 每个 endpoint 生成一行模板。
    如果已有 XLogP，填入 logKow 行。
    """
    rows = []
    for _, pfas_row in target_df.iterrows():
        pfas_name = str(pfas_row.get("pfas_name", ""))
        cas       = str(pfas_row.get("CAS",      "") or "")
        inchikey  = str(pfas_row.get("InChIKey",  "") or "")
        dtxsid    = str(pfas_row.get("DTXSID",   "") or "")
        pfas_cls  = str(pfas_row.get("pfas_class","") or "")
        xlogp_raw = str(pfas_row.get("XLogP",    "") or "").strip()

        for ep in ENDPOINTS:
            # logKow: 如果第一层有 XLogP，直接填入
            if ep == "logKow" and xlogp_raw and xlogp_raw not in ("", "nan", "None"):
                value             = xlogp_raw
                measured_or_pred  = "predicted"     # PubChem XLogP3 是计算值
                source_db         = "PubChem XLogP3"
                reliability       = "2"             # Klimisch 2 (计算/预测)
                note              = "From PubChem XLogP3 (calculated); verify with experimental data"
            else:
                value             = ""
                measured_or_pred  = ""
                source_db         = ""
                reliability       = ""
                note              = ""

            rows.append({
                "pfas_name":            pfas_name,
                "CAS":                  cas,
                "InChIKey":             inchikey,
                "DTXSID":               dtxsid,
                "pfas_class":           pfas_cls,
                "endpoint":             ep,
                "value":                value,
                "unit":                 ENDPOINT_UNITS.get(ep, ""),
                "temperature":          "",   # e.g. 25 (°C)
                "pH":                   "",   # relevant for pKa, logD, Koc
                "source_database":      source_db,
                "source_reference":     "",   # DOI or database version
                "measured_or_predicted": measured_or_pred,
                "reliability":          reliability,
                "notes":                note,
            })

    df = pd.DataFrame(rows, columns=LONG_COLS)
    log.info(f"[长表] {len(df)} 行 "
             f"({len(target_df)} PFAS × {len(ENDPOINTS)} endpoints)")
    return df


# ─── 构建宽表 ─────────────────────────────────────────────────────────────

def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    从长表 pivot 成宽表：
    每行一个 PFAS，endpoint 值 / 来源 / 类型 各占一列。
    """
    id_cols = ["pfas_name", "CAS", "InChIKey", "DTXSID", "pfas_class"]
    pivoted = long_df.pivot_table(
        index=id_cols,
        columns="endpoint",
        values=["value", "unit", "source_database", "measured_or_predicted", "reliability"],
        aggfunc="first",
    )
    # 展平多级列名: (field, endpoint) -> endpoint_field
    pivoted.columns = [f"{ep}_{field}" for field, ep in pivoted.columns]
    pivoted = pivoted.reset_index()

    # 整理列顺序：id cols first，然后按 endpoint 分组
    ordered = list(id_cols)
    for ep in ENDPOINTS:
        for field in ["value", "unit", "source_database", "measured_or_predicted", "reliability"]:
            col = f"{ep}_{field}"
            if col in pivoted.columns:
                ordered.append(col)
    # 任何未列出的列追加到末尾
    extra = [c for c in pivoted.columns if c not in ordered]
    ordered += extra
    pivoted = pivoted[[c for c in ordered if c in pivoted.columns]]

    log.info(f"[宽表] {len(pivoted)} 行, {len(pivoted.columns)} 列")
    return pivoted


# ─── 构建缺失优先级矩阵 ──────────────────────────────────────────────────

def build_missing_priority(long_df: pd.DataFrame,
                            target_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 PFAS 统计哪些 endpoint 缺失（value 为空）。
    按第二层出现频率（total_freq）降序排列。
    """
    freq_map = dict(zip(target_df["pfas_name"], target_df["total_freq"]))

    rows = []
    for pfas_name, grp in long_df.groupby("pfas_name"):
        missing = []
        filled  = []
        for _, r in grp.iterrows():
            ep = r["endpoint"]
            if str(r["value"]).strip() in ("", "nan", "None"):
                missing.append(ep)
            else:
                filled.append(ep)
        n_missing = len(missing)
        n_filled  = len(filled)
        rows.append({
            "pfas_name":        pfas_name,
            "total_freq":       freq_map.get(pfas_name, 0),
            "n_endpoints_total": len(ENDPOINTS),
            "n_filled":         n_filled,
            "n_missing":        n_missing,
            "completeness_pct": round(100 * n_filled / max(len(ENDPOINTS), 1), 1),
            "missing_endpoints": "|".join(missing),
            "filled_endpoints":  "|".join(filled),
            "priority_note":    _priority_note(n_missing, freq_map.get(pfas_name, 0)),
        })

    df = (pd.DataFrame(rows)
          .sort_values(["total_freq", "n_missing"], ascending=[False, False])
          .reset_index(drop=True))
    df.insert(0, "priority_rank", df.index + 1)
    log.info(f"[缺失矩阵] {len(df)} 行")
    return df


def _priority_note(n_missing: int, freq: int) -> str:
    if freq >= 20 and n_missing >= 6:
        return "HIGH PRIORITY — high TMF freq, most endpoints empty"
    elif freq >= 10 and n_missing >= 4:
        return "MEDIUM PRIORITY — moderate freq, several endpoints empty"
    elif n_missing == 0:
        return "COMPLETE — all endpoints filled"
    elif n_missing <= 2:
        return "NEARLY COMPLETE — 1-2 endpoints missing"
    else:
        return "LOW PRIORITY — low TMF freq"


# ─── 写出 Excel ───────────────────────────────────────────────────────────

def write_excel(out_path: Path,
                target_df: pd.DataFrame,
                long_df:   pd.DataFrame,
                wide_df:   pd.DataFrame,
                miss_df:   pd.DataFrame) -> None:
    log.info(f"[输出] 写入: {out_path}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # Sheet 1: target_pfas_list
        target_out = target_df.copy()
        target_out.insert(0, "priority_rank", range(1, len(target_out) + 1))
        col_order = ["priority_rank", "pfas_name", "aliases", "total_freq", "freq_table2",
                     "freq_table3", "CAS", "InChIKey", "DTXSID", "SMILES",
                     "MolFormula", "pfas_class", "XLogP", "match_method"]
        target_out = target_out[[c for c in col_order if c in target_out.columns]]
        target_out.to_excel(writer, sheet_name="target_pfas_list", index=False)

        # Sheet 2: transport_template_long
        long_df.to_excel(writer, sheet_name="transport_template_long", index=False)

        # Sheet 3: transport_template_wide
        wide_df.to_excel(writer, sheet_name="transport_template_wide", index=False)

        # Sheet 4: missing_priority
        miss_df.to_excel(writer, sheet_name="missing_priority", index=False)

        # 格式美化
        _format_workbook(writer.book, target_out, long_df, wide_df, miss_df)

    log.info(f"  ✅ 写入完成: {out_path}")


def _format_workbook(wb, target_df, long_df, wide_df, miss_df):
    """设置列宽 + 冻结首行 + 标题行颜色"""
    try:
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")
        HEADER_FONT   = Font(color="FFFFFF", bold=True, size=10)
        WARN_FILL     = PatternFill("solid", fgColor="FFF2CC")  # 浅黄 = 待填
        GOOD_FILL     = PatternFill("solid", fgColor="E2EFDA")  # 浅绿 = 已填

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            # 标题行样式
            for cell in ws[1]:
                cell.fill  = HEADER_FILL
                cell.font  = HEADER_FONT
                cell.alignment = Alignment(horizontal="center", wrap_text=True)

            # 自动列宽（最多 50）
            for col_cells in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    try:
                        cell_len = len(str(cell.value or ""))
                        if cell_len > max_len:
                            max_len = cell_len
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 3, 50)

        # transport_template_long: 高亮空 value 行（待填）
        ws_long = wb["transport_template_long"]
        # 找 value 列索引
        header = [cell.value for cell in ws_long[1]]
        val_col_idx = None
        mp_col_idx  = None
        for i, h in enumerate(header, 1):
            if str(h or "").strip() == "value":
                val_col_idx = i
            if str(h or "").strip() == "measured_or_predicted":
                mp_col_idx = i

        if val_col_idx:
            for row in ws_long.iter_rows(min_row=2):
                val_cell = row[val_col_idx - 1]
                if str(val_cell.value or "").strip() in ("", "None", "nan"):
                    for cell in row:
                        cell.fill = WARN_FILL
                else:
                    for cell in row:
                        cell.fill = GOOD_FILL

    except Exception as e:
        log.warning(f"  格式化失败 (不影响内容): {e}")


# ─── 主流程 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PFAS Transport Layer Builder")
    ap.add_argument("--structure", "-s", type=Path,
                    default=Path("PFAS_master_structure.csv"),
                    help="第一层结构主库 (xlsx 或 csv，推荐 csv)")
    ap.add_argument("--tmf",       "-t", type=Path,
                    default=Path("pfas_ranked/extraction_tables_v4.xlsx"),
                    help="第二层 TMF 数据 (extraction_tables_v4.xlsx)")
    ap.add_argument("--out",       "-o", type=Path,
                    default=Path("PFAS_transport_layer_v1.xlsx"),
                    help="输出路径")
    args = ap.parse_args()

    log.info("=" * 65)
    log.info("  PFAS Transport Layer Builder")
    log.info("=" * 65)
    log.info(f"  第一层结构库: {args.structure}")
    log.info(f"  第二层 TMF:   {args.tmf}")
    log.info(f"  输出:         {args.out}")

    # 确保输出目录存在
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. 读取第一层
    struct_df = load_structure_db(args.structure)

    # 2. 读取第二层，提取 PFAS 列表
    pfas_list_df = load_tmf_pfas(args.tmf)
    log.info(f"[第二层] 提取到 {len(pfas_list_df)} 种独立 PFAS")

    # 3. 匹配
    target_df = match_pfas(struct_df, pfas_list_df)

    # 4. 建长表（模板 + 已有 XLogP → logKow）
    long_df = build_long_table(target_df)

    # 5. 建宽表
    wide_df = build_wide_table(long_df)

    # 6. 缺失优先级矩阵
    miss_df = build_missing_priority(long_df, target_df)

    # 7. 写出
    write_excel(args.out, target_df, long_df, wide_df, miss_df)

    # ── 汇总日志
    log.info("")
    log.info("=" * 65)
    log.info("  汇总")
    log.info("=" * 65)
    log.info(f"  第一层结构库条目:        {len(struct_df)}")
    log.info(f"  第二层 PFAS 种数:        {len(pfas_list_df)}")
    matched_n = (target_df["match_method"] != "unmatched").sum()
    log.info(f"  第一层匹配成功:          {matched_n}/{len(target_df)}")
    log.info(f"  长表总行数:              {len(long_df)}")
    filled = (long_df["value"].str.strip().replace("", np.nan).notna()).sum()
    log.info(f"  已有值 (logKow from XLogP): {filled}")
    log.info(f"  待填 endpoint 行:        {len(long_df) - filled}")
    high_pri = (miss_df["priority_note"].str.startswith("HIGH")).sum()
    log.info(f"  HIGH PRIORITY PFAS:      {high_pri}")
    log.info("")
    log.info(f"  输出文件: {args.out.resolve()}")
    log.info("")
    log.info("  Sheet 说明:")
    log.info("    target_pfas_list        第二层出现的 PFAS + 第一层结构字段")
    log.info("    transport_template_long 长表（一行 = 一个 PFAS × endpoint）")
    log.info("    transport_template_wide 宽表（一行 = 一个 PFAS）")
    log.info("    missing_priority        缺失矩阵，按出现频率排序")
    log.info("")
    log.info("  下一步:")
    log.info("    1) 打开 missing_priority，从 HIGH PRIORITY 开始填")
    log.info("    2) 数据来源优先: EPA DSSTox > OECD eChemPortal > ACD/Labs")
    log.info("    3) 实测值标 measured + reliability=1，预测值标 predicted + reliability=2/3")
    log.info("    4) 填完后可直接进入 Layer 3 × Layer 2 联合分析")


if __name__ == "__main__":
    main()