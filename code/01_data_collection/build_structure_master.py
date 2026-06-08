#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_structure_master.py
==========================
将 4 个原始 PFAS 结构库合并成 PFAS_master_structure.xlsx。

输入 (4 个源):
  EPA    Chemical List pfasmaster-2026-05-16.csv
  OECD   OECDPFAS_list_22012019.csv
  NORMAN susdat_2025-06-03-092022.csv
  PubChem 03_PubChem_PFAS_properties.csv

主键优先级: InChIKey > CASRN > DTXSID
XLogP 只从 PubChem 取 (其余源无此字段)。
NORMAN logKow 映射为 logKow_measured_NORMAN。

用法:
    python3 build_structure_master.py \
        --epa     /7t/lxkzero/claude/qd/diyi/pfas_data/EPA/"Chemical List pfasmaster-2026-05-16.csv" \
        --oecd    /7t/lxkzero/claude/qd/diyi/pfas_data/OECD/OECDPFAS_list_22012019.csv \
        --norman  /7t/lxkzero/claude/qd/diyi/pfas_data/NORMAN/susdat_2025-06-03-092022.csv \
        --pubchem /7t/lxkzero/claude/qd/diyi/pfas_data/20260516_130259/03_PubChem_PFAS_properties.csv \
        --out     /7t/lxkzero/claude/qd/disan/PFAS_master_structure.xlsx
"""

from __future__ import annotations
import sys, re, logging, argparse
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("struct")

# ─── 列名别名映射（每个源独立处理）──────────────────────────────────────────

EPA_MAP = {
    "DTXSID":      ["dtxsid", "dsstox_substance_id"],
    "name":        ["preferred name", "preferredname", "name"],
    "CAS":         ["casrn", "cas rn", "cas"],
    "InChIKey":    ["inchikey", "std_inchikey"],
    "SMILES":      ["smiles", "qsar_ready_smiles", "canonical_smiles"],
    "InChI":       ["inchi", "std_inchi"],
    "MolFormula":  ["molecularformula", "molecular formula"],
    "pfas_class":  ["pfas_category", "category", "pfas class"],
    "MW":          ["averagemass", "average mass", "molweight"],
}

OECD_MAP = {
    "DTXSID":      ["dtxsid"],
    "name":        ["preferred name", "preferredname", "name", "substance name"],
    "CAS":         ["casrn", "cas rn", "cas", "cas number"],
    "InChIKey":    ["inchikey", "std_inchikey"],
    "SMILES":      ["smiles", "canonical_smiles"],
    "InChI":       ["inchi", "std_inchi"],
    "MolFormula":  ["molecularformula", "molecular formula"],
    "pfas_class":  ["pfas_category", "category", "pfas class", "subgroup"],
}

NORMAN_MAP = {
    "CAS":         ["cas_rn", "cas", "casrn", "cas number"],
    "name":        ["name", "preferredname", "substance name", "chemical name"],
    "InChIKey":    ["stdinchikey", "inchikey", "std_inchikey"],
    "DTXSID":      ["dtxsid"],
    "SMILES":      ["smiles", "canonical_smiles"],
    "InChI":       ["stdinchi", "inchi", "std_inchi"],
    "MolFormula":  ["molecular_formula", "molecularformula", "molecular formula", "formula"],
    # Exp_logKow_EPISuite 优先(实测),其次 logKow_EPISuite(预测)
    "logKow_Norman":          ["exp_logkow_episuite"],
    "logKow_Norman_predicted": ["logkow_episuite"],
    "MW":          ["average_mass", "molweight", "molecularweight", "mw"],
    "PubChemCID":  ["pubchem_cid", "pubchem cid", "pubchemcid", "cid"],
}

PUBCHEM_MAP = {
    "PubChemCID":  ["cid"],
    "name":        ["iupacname", "iupac name", "preferredname"],
    "InChIKey":    ["inchikey"],
    "InChI":       ["inchi"],
    "SMILES":      ["isomericsmiles", "canonicalsmiles", "smiles"],
    "MolFormula":  ["molecularformula"],
    "MW":          ["molecularweight"],
    "XLogP":       ["xlogp", "xlogp3", "xlogp3-aa"],
}


def _remap(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """按 col_map 把原始列名统一成标准名（大小写不敏感，多余列保留）"""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for std, aliases in col_map.items():
        if std in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = std
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _read(path: Path, label: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        log.warning(f"  [{label}] 文件不存在: {path}，跳过")
        return None
    log.info(f"  [{label}] 读取: {path.name}")
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path, dtype=str)
            else:
                df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            log.info(f"    {len(df)} 行, {len(df.columns)} 列")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.error(f"    读取失败: {e}")
            return None
    log.error(f"    [{label}] 无法解码")
    return None


def _clean_inchikey(s) -> str:
    s = str(s or "").strip().upper()
    # InChIKey 格式: 14-10-1 大写字母
    if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", s):
        return s
    return ""


def _clean_cas(s) -> str:
    s = str(s or "").strip()
    # NORMAN 的 CAS_RN 列值带 "CAS_RN: " 前缀
    s = re.sub(r"^cas[_ ]?rn\s*[:：]\s*", "", s, flags=re.I)
    if re.match(r"^\d{2,7}-\d{2}-\d$", s):
        return s
    return ""


def _classify_pfas(smiles: str, name: str = "") -> str:
    """
    极简 PFAS 子类标注(基于 SMILES + 名称回退)。
    返回:PFCA / PFSA / FASA / FTOH / FTS / FTCA / PAP / FOSAA / Ether(GenX 类) / 其它全氟
    """
    s = str(smiles or "").upper()
    n = str(name or "").lower()
    # 至少要有 3 个 F 才算全氟
    has_perfluoro = s.count("F") >= 3 and "C(F)(F)" in s

    # 1) 名称硬规则(优先,因为最可靠)
    if "fosaa" in n or "fosa-acetic" in n or "etfosaa" in n or "mefosaa" in n:
        return "FOSAA"
    if "diPAP".lower() in n or n.endswith("pap") or " pap" in n or "polyfluoroalkyl phosphate" in n:
        return "PAP"
    if "ftoh" in n or "fluorotelomer alcohol" in n:
        return "FTOH"
    if "ftsa" in n or " fts" in n or n.endswith("fts") or "fluorotelomer sulfon" in n:
        return "FTS"
    if "ftca" in n or "fluorotelomer carboxylic" in n:
        return "FTCA"
    if "hfpo" in n or "genx" in n or "adona" in n or "f-53b" in n or "pfechs" in n:
        return "Ether"
    # 磺酰胺(FOSA / FBSA / FHxSA)
    if (n.endswith("osa") or n.endswith("hxsa") or n.endswith("bsa")
            or "sulfonamide" in n):
        return "FASA"

    # 2) SMILES 兜底 — 注意 SMILES 可能从任一端写起
    has_carboxyl   = ("C(=O)O" in s) or ("OC(=O)" in s)
    has_sulfonate  = ("S(=O)(=O)O" in s) or ("OS(=O)(=O)" in s)
    has_sulfonamide_n = ("S(=O)(=O)N" in s) or ("NS(=O)(=O)" in s)
    has_phosphate  = ("OP(=O)" in s) or ("P(=O)(O)" in s)
    has_ether_o    = ("OC(F)" in s) or ("C(F)(F)O" in s)

    if not has_perfluoro:
        return ""

    if has_phosphate:
        return "PAP"
    if has_sulfonamide_n:
        return "FASA"
    # 氟调聚:全氟链 + -CC- 桥 + -OH/-S/-COOH
    if "C(F)(F)CCO" in s or "OCCC(F)(F)" in s:
        return "FTOH"
    if "C(F)(F)CCS(=O)(=O)O" in s or "OS(=O)(=O)CCC(F)(F)" in s:
        return "FTS"
    if has_ether_o and (has_carboxyl or has_sulfonate):
        # 主链含 C-O-C 醚的全氟酸(GenX/HFPO-DA、ADONA、F-53B)
        # 用 ether O 出现次数判断:链上至少有一个 O 在 C(F)... 之间
        # 简化:若 SMILES 中 "F)O" 或 "OC(F)" 出现且不是端基 OH,认为有醚
        return "Ether"
    if has_sulfonate:
        return "PFSA"
    if has_carboxyl:
        return "PFCA"
    return "其它全氟"


def _clean_dtxsid(s) -> str:
    s = str(s or "").strip().upper()
    if s.startswith("DTXSID"):
        return s
    return ""


# ─── 主合并逻辑 ───────────────────────────────────────────────────────────

def build_master(epa_path, oecd_path, norman_path, pubchem_path, out_path):

    # 1. 读取 + 列名规范化
    epa_raw     = _read(epa_path,     "EPA")
    oecd_raw    = _read(oecd_path,    "OECD")
    norman_raw  = _read(norman_path,  "NORMAN")
    pubchem_raw = _read(pubchem_path, "PubChem")

    epa_df     = _remap(epa_raw,     EPA_MAP)     if epa_raw     is not None else pd.DataFrame()
    oecd_df    = _remap(oecd_raw,    OECD_MAP)    if oecd_raw    is not None else pd.DataFrame()
    norman_df  = _remap(norman_raw,  NORMAN_MAP)  if norman_raw  is not None else pd.DataFrame()
    pubchem_df = _remap(pubchem_raw, PUBCHEM_MAP) if pubchem_raw is not None else pd.DataFrame()

    # 打印识别到的列
    for label, df in [("EPA",epa_df),("OECD",oecd_df),("NORMAN",norman_df),("PubChem",pubchem_df)]:
        key_cols = ["name","CAS","InChIKey","DTXSID","SMILES","XLogP","logKow_Norman","pfas_class"]
        found = [c for c in key_cols if c in df.columns]
        log.info(f"  [{label}] 识别关键列: {found}")

    # 2. 给每个源加前缀标记并确保关键列存在
    def ensure_cols(df, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df

    KEY_COLS = ["name","CAS","InChIKey","DTXSID","SMILES","InChI","MolFormula","pfas_class","MW"]
    epa_df     = ensure_cols(epa_df,     KEY_COLS)
    oecd_df    = ensure_cols(oecd_df,    KEY_COLS)
    norman_df  = ensure_cols(norman_df,  KEY_COLS + ["logKow_Norman","logKow_Norman_predicted","PubChemCID"])
    pubchem_df = ensure_cols(pubchem_df, KEY_COLS + ["XLogP","PubChemCID"])

    # 3. 清洗主键
    for df in [epa_df, oecd_df, norman_df, pubchem_df]:
        df["InChIKey"] = df["InChIKey"].apply(_clean_inchikey)
        df["CAS"]      = df["CAS"].apply(_clean_cas)
        df["DTXSID"]   = df["DTXSID"].apply(_clean_dtxsid)

    # 4. 合并策略：以 InChIKey 为主键构建全集
    #    优先级: EPA > OECD > NORMAN > PubChem（结构字段）
    #    XLogP 只从 PubChem 补
    #    logKow_Norman 只从 NORMAN 补

    log.info("[合并] 构建全集...")

    # Step A: 先取 EPA + OECD + NORMAN 的 union（按 InChIKey 去重）
    UNION_COLS = KEY_COLS + ["logKow_Norman", "logKow_Norman_predicted", "PubChemCID"]
    structural_sources = []
    for label, df in [("EPA", epa_df), ("OECD", oecd_df), ("NORMAN", norman_df)]:
        sub = df.reindex(columns=UNION_COLS, fill_value="").copy()
        sub["_src"] = label
        structural_sources.append(sub)

    all_struct = pd.concat(structural_sources, ignore_index=True)
    log.info(f"  合并前总行: {len(all_struct)}")

    # InChIKey 去重（优先 EPA > OECD > NORMAN）
    src_priority = {"EPA": 0, "OECD": 1, "NORMAN": 2}
    all_struct["_pri"] = all_struct["_src"].map(src_priority).fillna(9)

    # 有 InChIKey 的先按 InChIKey 去重
    has_ik  = all_struct[all_struct["InChIKey"] != ""].sort_values("_pri")
    no_ik   = all_struct[all_struct["InChIKey"] == ""]
    has_ik  = has_ik.drop_duplicates("InChIKey", keep="first")

    # 无 InChIKey 的按 CAS 去重
    has_cas = no_ik[no_ik["CAS"] != ""].sort_values("_pri")
    no_cas  = no_ik[no_ik["CAS"] == ""]
    has_cas = has_cas.drop_duplicates("CAS", keep="first")

    master = pd.concat([has_ik, has_cas, no_cas], ignore_index=True)
    master = master.drop(columns=["_pri"])
    log.info(f"  去重后: {len(master)} 条唯一化合物")

    # Step B: 从 PubChem 补充 XLogP
    log.info("[合并] 从 PubChem 补 XLogP...")
    pc_sub = pubchem_df[["InChIKey","CAS","SMILES","MolFormula","MW","XLogP","PubChemCID"]].copy()
    pc_sub = pc_sub[pc_sub["XLogP"].notna() & (pc_sub["XLogP"] != "")]

    # 按 InChIKey merge
    pc_by_ik  = pc_sub[pc_sub["InChIKey"] != ""].drop_duplicates("InChIKey")
    master = master.merge(
        pc_by_ik[["InChIKey","XLogP","PubChemCID"]].rename(
            columns={"XLogP":"XLogP_pc","PubChemCID":"PubChemCID_pc"}),
        on="InChIKey", how="left"
    )

    # 合并 XLogP（如已有则保留，否则用 PubChem）
    if "XLogP" not in master.columns:
        master["XLogP"] = ""
    master["XLogP"] = master["XLogP"].where(
        master["XLogP"].fillna("").str.strip() != "",
        master["XLogP_pc"]
    )
    if "PubChemCID" not in master.columns:
        master["PubChemCID"] = ""
    master["PubChemCID"] = master["PubChemCID"].where(
        master["PubChemCID"].fillna("").str.strip() != "",
        master["PubChemCID_pc"]
    )
    master = master.drop(columns=["XLogP_pc","PubChemCID_pc"], errors="ignore")

    pc_filled = master["XLogP"].notna() & (master["XLogP"].fillna("").str.strip() != "")
    log.info(f"  XLogP 已填: {pc_filled.sum()} / {len(master)}")

    # Step C-1: logKow_Norman 用 EPISuite 预测值兜底
    if "logKow_Norman" not in master.columns:
        master["logKow_Norman"] = ""
    if "logKow_Norman_predicted" in master.columns:
        empty_mask = master["logKow_Norman"].fillna("").str.strip() == ""
        master.loc[empty_mask, "logKow_Norman"] = master.loc[empty_mask, "logKow_Norman_predicted"].fillna("")
        master = master.drop(columns=["logKow_Norman_predicted"], errors="ignore")
    n_logkow = (master["logKow_Norman"].fillna("").str.strip() != "").sum()
    log.info(f"  logKow_Norman (实测+预测) 已填: {n_logkow} / {len(master)}")

    # Step C-2: pfas_class 自动分类(基于 SMILES)
    log.info("[分类] 跑 SMILES → pfas_class...")
    if "pfas_class" not in master.columns:
        master["pfas_class"] = ""
    master["pfas_class"] = master.apply(
        lambda r: r["pfas_class"] if str(r.get("pfas_class","")).strip() else
                  _classify_pfas(r.get("SMILES",""), r.get("name","")),
        axis=1,
    )
    cls_counts = master["pfas_class"].value_counts(dropna=False).head(10)
    log.info(f"  pfas_class 分布 top10:")
    for k, v in cls_counts.items():
        log.info(f"    {k or '<空>':12s} {v:>7d}")

    # Step C: 整理最终列顺序
    final_cols = [
        "name", "CAS", "InChIKey", "DTXSID", "PubChemCID",
        "SMILES", "InChI", "MolFormula", "MW",
        "pfas_class",
        "XLogP",               # PubChem XLogP3 (predicted)
        "logKow_Norman",       # NORMAN logKow (may be measured)
        "_src",
    ]
    for c in final_cols:
        if c not in master.columns:
            master[c] = ""
    master = master[[c for c in final_cols if c in master.columns]]
    master = master.rename(columns={"_src": "primary_source"})

    # 清空 nan 字符串
    master = master.replace({"nan": "", "None": "", "NaN": ""})
    master = master.fillna("")

    # 清除 openpyxl 不允许的非法字符（控制字符 + 私用区 Unicode）
    import re as _re
    _ILLEGAL = _re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"   # C0/C1 控制字符
        r"\ufffd\ufffe\uffff"                         # 替换字符 / BOM
        r"\U000f0000-\U000fffff"                      # PUA
        r"]"
    )
    def _clean_cell(v):
        if isinstance(v, str):
            return _ILLEGAL.sub("", v)
        return v

    log.info("  清除非法字符...")
    master = master.applymap(_clean_cell)

    log.info(f"[最终] 主库: {len(master)} 条, {len(master.columns)} 列")

    # 5. 写出
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 5a. 完整主库写 CSV（13万行 xlsx 太慢，且规避非法字符）
    csv_path = out_path.with_suffix(".csv")
    master.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"  完整主库 CSV: {csv_path}  ({len(master):,} 行)")

    src_stats = pd.DataFrame([
        {"source": "EPA",               "count": len(epa_df)},
        {"source": "OECD",              "count": len(oecd_df)},
        {"source": "NORMAN",            "count": len(norman_df)},
        {"source": "PubChem",           "count": len(pubchem_df)},
        {"source": "MERGED (unique)",   "count": len(master)},
        {"source": "with InChIKey",     "count": int((master["InChIKey"]!="").sum())},
        {"source": "with CAS",          "count": int((master["CAS"]!="").sum())},
        {"source": "with DTXSID",       "count": int((master["DTXSID"]!="").sum())},
        {"source": "with XLogP",        "count": int(pc_filled.sum())},
        {"source": "with logKow_Norman","count": int((master["logKow_Norman"]!="").sum())
                                                  if "logKow_Norman" in master.columns else 0},
    ])

    # 5b. Excel 只写统计 + 前5000行预览
    log.info("  写 Excel 预览 (stats + 前5000行)...")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        master.head(5000).to_excel(writer, sheet_name="preview_5000rows", index=False)
        src_stats.to_excel(writer, sheet_name="merge_stats", index=False)
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.utils import get_column_letter
            FILL = PatternFill("solid", fgColor="1F4E79")
            FONT = Font(color="FFFFFF", bold=True)
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for cell in ws[1]:
                    cell.fill = FILL; cell.font = FONT
                    cell.alignment = Alignment(horizontal="center")
                for col_cells in ws.columns:
                    ml = max((len(str(c.value or "")) for c in col_cells), default=8)
                    ws.column_dimensions[
                        get_column_letter(col_cells[0].column)].width = min(ml + 3, 45)
        except Exception as e:
            log.warning(f"  格式化失败: {e}")

    log.info(f"✅ 写出: {out_path}  (预览 xlsx) + {csv_path.name}  (完整 csv)")

    # 汇总
    log.info("")
    log.info("=" * 60)
    log.info("  合并统计")
    log.info("=" * 60)
    for _, r in src_stats.iterrows():
        log.info(f"  {r['source']:<30} {r['count']:>8,}")
    log.info("")
    log.info("下一步:")
    log.info(f"  python3 build_transport_layer.py \\")
    log.info(f"      --structure {out_path} \\")
    log.info(f"      --tmf  <extraction_tables_v4.xlsx> \\")
    log.info(f"      --out  PFAS_transport_layer_v1.xlsx")


def main():
    ap = argparse.ArgumentParser(description="PFAS Structure Master Builder")
    ap.add_argument("--epa",     type=Path, default=None)
    ap.add_argument("--oecd",    type=Path, default=None)
    ap.add_argument("--norman",  type=Path, default=None)
    ap.add_argument("--pubchem", type=Path, default=None)
    ap.add_argument("--out",     type=Path,
                    default=Path("PFAS_master_structure.xlsx"))
    args = ap.parse_args()

    log.info("=" * 60)
    log.info("  PFAS Structure Master Builder")
    log.info("=" * 60)
    build_master(args.epa, args.oecd, args.norman, args.pubchem, args.out)


if __name__ == "__main__":
    main()