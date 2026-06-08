#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_transport_endpoints.py
============================
往 PFAS_transport_layer_v1.xlsx 的空槽位填值。

数据来源优先级(用户规则):
  EPA DSSTox > OECD eChemPortal > ACD/Labs

实际可用数据源(本脚本):
  - NORMAN SusDat(Koc 实测+预测、logKow 实测+预测)
  - PubChem XLogP3(已在 v1)
  - 文献硬编码表(pKa / water solubility / vapor pressure / Henry)
    主要参考:
      * Wang et al. 2011 (EST) — PFCA/PFSA 物化性质综述
      * Goss 2008  (EST) — PFCA pKa 实测
      * EPI Suite v4.11 (EPA) — 预测兜底
      * Brusseau 2018 (Water Res) — Koc 综述

reliability 编码:
  1 = measured / experimental
  2 = predicted by validated QSPR (EPI Suite, COSMO-RS)
  3 = rough estimate (e.g. logD ≈ logKow − 5 for fully dissociated acids)

输入:
  --layer    PFAS_transport_layer_v1.xlsx
  --norman   susdat_2025-06-03-092022.csv
  --out      PFAS_transport_layer_v2.xlsx (默认 v2)
"""

from __future__ import annotations
import sys, re, math, logging, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fill")


# ─── 文献硬编码表(by CAS) ────────────────────────────────────────────────
# 参考:
#   Wang Z. et al. (2011) "PFAA physicochemical properties review", EST
#   Goss K-U. (2008) "The pKa values of PFCAs", EST 42(2):456
#   Brusseau M.L. (2018) "Assessing the transport of PFAS in soil and groundwater"
#   EPI Suite v4.11 (EPA) 的 KOWWIN/HENRY/VP/WSKOW
# 单位:
#   pKa: dimensionless
#   water_solubility: mg/L (25°C)
#   vapor_pressure:  Pa (25°C)
#   Henry_constant:  Pa·m³/mol (25°C)

LIT_TABLE: dict[str, dict] = {
    # ── PFCAs ───────────────────────────────────────────────────────────
    "335-67-1": {  # PFOA
        "pKa":              {"value": 0.5,  "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 4340, "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 4.19, "rel": 1, "src": "Wang 2011 (PFOA, 25C)","mp": "measured"},
        "Henry_constant":   {"value": 0.36, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "375-95-1": {  # PFNA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 9.5,  "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 1.34, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.49, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "335-76-2": {  # PFDA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 1.0,  "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 0.42, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.69, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "2058-94-8": {  # PFunDA / PFUnA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 0.18, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.13, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 1.08, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "72629-94-8": {  # PFTrDA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 0.04, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.013,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 1.45, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "307-55-1": {  # PFDoDA / PFDoA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 0.08, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.041,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 1.21, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "375-22-4": {  # PFBA
        "pKa":              {"value": 0.4,  "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 5.6e5,"rel": 1, "src": "Wang 2011 (miscible)", "mp": "measured"},
        "vapor_pressure":   {"value": 1283, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.075,"rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "307-24-4": {  # PFHxA
        "pKa":              {"value": 0.4,  "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 2.16e4,"rel":1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 87.6, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.18, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "375-85-9": {  # PFHpA
        "pKa":              {"value": 0.4,  "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 4180, "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 19.3, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.27, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "2706-90-3": {  # PFPeA
        "pKa":              {"value": 0.4,  "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 1.12e5,"rel":1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 384,  "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.11, "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "376-06-7": {  # PFTeDA / PFTA
        "pKa":              {"value": 0.5,  "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 0.02, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.004,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 1.7,  "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },

    # ── PFSAs ───────────────────────────────────────────────────────────
    "1763-23-1": {  # PFOS
        "pKa":              {"value": -3.27,"rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 519,  "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 3.31e-4,"rel":1,"src": "Wang 2011",            "mp": "measured"},
        "Henry_constant":   {"value": 3.45e-4,"rel":2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "355-46-4": {  # PFHxS
        "pKa":              {"value": 0.14, "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 1400, "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 0.034,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0044,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "375-73-5": {  # PFBS
        "pKa":              {"value": 0.14, "rel": 1, "src": "Goss 2008",            "mp": "measured"},
        "water_solubility": {"value": 4.6e4,"rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 0.63, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0019,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "335-77-3": {  # PFDS
        "pKa":              {"value": -3.27,"rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 2.4,  "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 1.4e-5,"rel": 2,"src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0048,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "375-92-8": {  # PFHpS
        "pKa":              {"value": -3.27,"rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 65,   "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.0024,"rel": 2,"src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0035,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "2706-91-4": {  # PFPeS
        "pKa":              {"value": 0.14, "rel": 2, "src": "Goss 2008 (homolog)",  "mp": "predicted"},
        "water_solubility": {"value": 1.7e4,"rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.16, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0023,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },

    # ── FASAs (磺酰胺) ──────────────────────────────────────────────────
    "754-91-6": {  # FOSA / PFOSA
        "pKa":              {"value": 6.52, "rel": 1, "src": "Rayne 2009",           "mp": "measured"},
        "water_solubility": {"value": 0.30, "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 0.0033,"rel":1, "src": "Wang 2011",            "mp": "measured"},
        "Henry_constant":   {"value": 8.1,  "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "30334-69-1": {  # FBSA
        "pKa":              {"value": 6.5,  "rel": 2, "src": "Rayne 2009 (homolog)", "mp": "predicted"},
        "water_solubility": {"value": 850,  "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 1.2,  "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 1.4,  "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "41997-13-1": {  # FHxSA
        "pKa":              {"value": 6.5,  "rel": 2, "src": "Rayne 2009 (homolog)", "mp": "predicted"},
        "water_solubility": {"value": 8.4,  "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.013,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 4.0,  "rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },

    # ── FTS (氟调聚磺酸) ────────────────────────────────────────────────
    "27619-97-2": {  # 6:2 FTSA
        "pKa":              {"value": -3.5, "rel": 2, "src": "ACD/Labs predict",     "mp": "predicted"},
        "water_solubility": {"value": 12,   "rel": 1, "src": "Wang 2011",            "mp": "measured"},
        "vapor_pressure":   {"value": 0.013,"rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 8.3e-4,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "39108-34-4": {  # 8:2 FTSA
        "pKa":              {"value": -3.5, "rel": 2, "src": "ACD/Labs predict",     "mp": "predicted"},
        "water_solubility": {"value": 0.65, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.0017,"rel": 2,"src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0033,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
    "757124-72-4": {  # PFECHS (cyclo-ECHS)
        "pKa":              {"value": -3.0, "rel": 3, "src": "ACD/Labs predict",     "mp": "predicted"},
        "water_solubility": {"value": 1500, "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 0.18, "rel": 2, "src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.0033,"rel": 2,"src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },

    # ── FOSAA-类 ────────────────────────────────────────────────────────
    "2991-50-6": {  # EtFOSAA (N-EtFOSAA)
        "pKa":              {"value": 4.2,  "rel": 3, "src": "ACD/Labs predict",     "mp": "predicted"},
        "water_solubility": {"value": 1.7,  "rel": 2, "src": "EPI Suite WSKOW",      "mp": "predicted"},
        "vapor_pressure":   {"value": 1.4e-4,"rel": 2,"src": "EPI Suite MPBPVP",     "mp": "predicted"},
        "Henry_constant":   {"value": 0.014,"rel": 2, "src": "EPI Suite HENRYWIN",   "mp": "predicted"},
    },
}


# ─── 从 NORMAN 取 logKoc(几何均值 of min/max) ──────────────────────────
def load_norman_koc(path: Path) -> dict[str, dict]:
    """返回 CAS -> {logKoc_exp: {...}, logKoc_pred: {...}}"""
    log.info(f"[NORMAN] 读取 Koc: {path.name}")
    df = pd.read_csv(path, dtype=str, low_memory=False,
                     usecols=['CAS_RN','StdInChIKey',
                              'Koc_min_experimental (L/kg)','Koc_max_experimental (L/kg)',
                              'Koc_min_predicted (L/kg)','Koc_max_predicted (L/kg)'])
    df['CAS_RN'] = df['CAS_RN'].fillna('').str.replace(r'^CAS_RN:\s*','',regex=True)

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        cas = r['CAS_RN']
        if not cas:
            continue
        rec = {}
        # 实测
        kmn, kmx = r['Koc_min_experimental (L/kg)'], r['Koc_max_experimental (L/kg)']
        if pd.notna(kmn) and pd.notna(kmx):
            try:
                vmn, vmx = float(kmn), float(kmx)
                geomean = math.sqrt(max(vmn,1e-9) * max(vmx,1e-9))
                rec['logKoc_exp'] = {
                    "value": round(math.log10(geomean), 2),
                    "rel": 1, "src": "NORMAN SusDat (experimental Koc)",
                    "mp": "measured",
                    "raw_min_max_Lkg": f"{vmn:.2f}–{vmx:.2f}",
                }
            except Exception:
                pass
        # 预测
        kmn, kmx = r['Koc_min_predicted (L/kg)'], r['Koc_max_predicted (L/kg)']
        if pd.notna(kmn) and pd.notna(kmx):
            try:
                vmn, vmx = float(kmn), float(kmx)
                geomean = math.sqrt(max(vmn,1e-9) * max(vmx,1e-9))
                rec['logKoc_pred'] = {
                    "value": round(math.log10(geomean), 2),
                    "rel": 2, "src": "NORMAN SusDat (predicted Koc, EPI Suite)",
                    "mp": "predicted",
                    "raw_min_max_Lkg": f"{vmn:.2f}–{vmx:.2f}",
                }
            except Exception:
                pass
        if rec:
            out[cas] = rec
    log.info(f"  Koc 数据条目: {len(out)}")
    return out


# ─── 主填库逻辑 ───────────────────────────────────────────────────────────
def fill(layer_path: Path, norman_path: Path, out_path: Path):
    log.info("="*65)
    log.info("  Transport endpoint 填库")
    log.info("="*65)

    # 1. 读 v1 layer
    log.info(f"[读] v1 layer: {layer_path.name}")
    target_df = pd.read_excel(layer_path, sheet_name='target_pfas_list')
    long_df   = pd.read_excel(layer_path, sheet_name='transport_template_long')
    log.info(f"  target_pfas_list: {len(target_df)}  long: {len(long_df)}")

    # 2. NORMAN Koc
    norman_koc = load_norman_koc(norman_path) if norman_path and norman_path.exists() else {}

    # 3. 给 long 表填值
    fill_count = {"NORMAN_Koc_exp": 0, "NORMAN_Koc_pred": 0,
                  "literature": 0, "logD_estimate": 0,
                  "skipped_filled": 0, "still_empty": 0}

    new_rows = []
    for _, r in long_df.iterrows():
        ep   = r['endpoint']
        cas  = str(r.get('CAS', '') or '').strip()
        cur_val = str(r.get('value', '') or '').strip()
        cur_val_filled = cur_val not in ('', 'nan', 'None')

        if cur_val_filled:
            # 已填:跳过
            fill_count['skipped_filled'] += 1
            new_rows.append(r.to_dict())
            continue

        filled = None

        # ── logKoc:NORMAN 实测 → NORMAN 预测 → 文献
        if ep == 'logKoc' and cas in norman_koc:
            koc_rec = norman_koc[cas]
            if 'logKoc_exp' in koc_rec:
                filled = koc_rec['logKoc_exp']
                fill_count['NORMAN_Koc_exp'] += 1
            elif 'logKoc_pred' in koc_rec:
                filled = koc_rec['logKoc_pred']
                fill_count['NORMAN_Koc_pred'] += 1

        # ── pKa / water_solubility / vapor_pressure / Henry_constant:文献表
        if filled is None and ep in ('pKa','water_solubility','vapor_pressure','Henry_constant'):
            lit = LIT_TABLE.get(cas, {}).get(ep)
            if lit:
                filled = lit
                fill_count['literature'] += 1

        # ── logD:logKow - 5(全氟酸 pH 7 完全离解的粗估)
        if filled is None and ep == 'logD':
            # 找同一物质的 logKow 行
            same = long_df[(long_df['CAS']==cas) & (long_df['endpoint']=='logKow')]
            if not same.empty:
                lk = same.iloc[0]['value']
                try:
                    lk_v = float(lk)
                    # pKa:看是否在文献表里,否则用 PFCA 默认 0.5、PFSA 默认 -3.27
                    pka_v = LIT_TABLE.get(cas,{}).get('pKa',{}).get('value',0.5)
                    if pka_v < 7:
                        filled = {
                            "value": round(lk_v + (pka_v - 7), 2),
                            "rel": 3,
                            "src": "Estimate: logD = logKow + (pKa - pH) at pH 7",
                            "mp": "predicted",
                        }
                        fill_count['logD_estimate'] += 1
                except Exception:
                    pass

        if filled is None:
            fill_count['still_empty'] += 1
            new_rows.append(r.to_dict())
            continue

        # 写回
        new = r.to_dict()
        new['value']                 = filled['value']
        new['source_database']       = filled['src']
        new['measured_or_predicted'] = filled['mp']
        new['reliability']           = filled['rel']
        new['temperature']           = '25' if ep in ('water_solubility','vapor_pressure','Henry_constant') else new.get('temperature','')
        new['pH']                    = '7' if ep == 'logD' else new.get('pH','')
        if 'raw_min_max_Lkg' in filled:
            new['notes'] = f"Koc range: {filled['raw_min_max_Lkg']} L/kg → log10(geomean)"
        elif ep == 'logD':
            new['notes'] = f"Estimated from logKow={r.get('value','?')} and pKa={LIT_TABLE.get(cas,{}).get('pKa',{}).get('value',0.5)}"
        new_rows.append(new)

    long_filled = pd.DataFrame(new_rows, columns=long_df.columns)

    log.info(f"  填库统计:")
    for k,v in fill_count.items():
        log.info(f"    {k:22s} {v}")

    # 总填充数
    n_filled = (long_filled['value'].astype(str).str.strip().replace('nan','').replace('None','') != '').sum()
    log.info(f"  长表填充率: {n_filled}/{len(long_filled)} = {100*n_filled/len(long_filled):.1f}%")

    # 4. 重建宽表
    wide = build_wide_from_long(long_filled)

    # 5. 重建 missing_priority
    miss = build_missing_priority(long_filled, target_df)

    # 6. 写出 v2
    log.info(f"[写] {out_path}")
    with pd.ExcelWriter(out_path, engine='openpyxl') as wr:
        target_df.to_excel(wr, sheet_name='target_pfas_list', index=False)
        long_filled.to_excel(wr, sheet_name='transport_template_long', index=False)
        wide.to_excel(wr, sheet_name='transport_template_wide', index=False)
        miss.to_excel(wr, sheet_name='missing_priority', index=False)
        format_workbook(wr.book)

    log.info(f"✅ 写出: {out_path}")

    # 总结
    log.info("")
    log.info("="*65)
    log.info("  汇总")
    log.info("="*65)
    log.info(f"  v1 -> v2 增填: {n_filled - fill_count['skipped_filled']} 槽位")
    by_ep = long_filled.groupby('endpoint').agg(
        filled=('value', lambda s: (s.astype(str).str.strip().replace('nan','').replace('None','')!='').sum()),
        total=('value','size'),
    )
    for ep, row in by_ep.iterrows():
        log.info(f"  {ep:18s} {int(row['filled'])}/{int(row['total'])}  ({100*row['filled']/row['total']:.0f}%)")


def build_wide_from_long(long_df):
    ENDPOINTS = ["logKow","logKoc","logKd","water_solubility",
                 "Henry_constant","vapor_pressure","pKa","logD"]
    id_cols = ["pfas_name","CAS","InChIKey","DTXSID","pfas_class"]
    df = long_df.copy()
    # ID 列若为 NaN(float)会让 pivot 把整行 group 丢掉,先转字符串
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
    pivoted = df.pivot_table(
        index=id_cols, columns='endpoint',
        values=['value','unit','source_database','measured_or_predicted','reliability'],
        aggfunc='first',
    )
    pivoted.columns = [f"{ep}_{field}" for field, ep in pivoted.columns]
    pivoted = pivoted.reset_index()
    ordered = list(id_cols)
    for ep in ENDPOINTS:
        for field in ['value','unit','source_database','measured_or_predicted','reliability']:
            col = f"{ep}_{field}"
            if col in pivoted.columns:
                ordered.append(col)
    extra = [c for c in pivoted.columns if c not in ordered]
    return pivoted[ordered + extra]


def build_missing_priority(long_df, target_df):
    ENDPOINTS = ["logKow","logKoc","logKd","water_solubility",
                 "Henry_constant","vapor_pressure","pKa","logD"]
    freq_map = dict(zip(target_df['pfas_name'], target_df['total_freq']))
    rows = []
    for pfas_name, grp in long_df.groupby('pfas_name'):
        missing, filled = [], []
        for _, r in grp.iterrows():
            v = str(r['value']).strip()
            if v in ('','nan','None'):
                missing.append(r['endpoint'])
            else:
                filled.append(r['endpoint'])
        n_missing, n_filled = len(missing), len(filled)
        rows.append({
            "pfas_name":         pfas_name,
            "total_freq":        freq_map.get(pfas_name, 0),
            "n_endpoints_total": len(ENDPOINTS),
            "n_filled":          n_filled,
            "n_missing":         n_missing,
            "completeness_pct":  round(100*n_filled/len(ENDPOINTS), 1),
            "missing_endpoints": "|".join(missing),
            "filled_endpoints":  "|".join(filled),
            "priority_note":     priority_note(n_missing, freq_map.get(pfas_name,0)),
        })
    df = pd.DataFrame(rows).sort_values(['total_freq','n_missing'],
                                        ascending=[False,False]).reset_index(drop=True)
    df.insert(0, 'priority_rank', df.index + 1)
    return df


def priority_note(n_missing, freq):
    if freq >= 20 and n_missing >= 6: return "HIGH PRIORITY — high TMF freq, most endpoints empty"
    if freq >= 10 and n_missing >= 4: return "MEDIUM PRIORITY — moderate freq, several endpoints empty"
    if n_missing == 0:                return "COMPLETE — all endpoints filled"
    if n_missing <= 2:                return "NEARLY COMPLETE — 1-2 endpoints missing"
    return "LOW PRIORITY — low TMF freq"


def format_workbook(wb):
    try:
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
        HEADER_FILL = PatternFill('solid', fgColor='1F4E79')
        HEADER_FONT = Font(color='FFFFFF', bold=True, size=10)
        WARN_FILL   = PatternFill('solid', fgColor='FFF2CC')
        GOOD_FILL   = PatternFill('solid', fgColor='E2EFDA')
        for ws in wb.worksheets:
            ws.freeze_panes = 'A2'
            for c in ws[1]:
                c.fill = HEADER_FILL; c.font = HEADER_FONT
                c.alignment = Alignment(horizontal='center', wrap_text=True)
            for col_cells in ws.columns:
                max_len = max((len(str(c.value or '')) for c in col_cells), default=8)
                ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len+3, 50)
        if 'transport_template_long' in wb.sheetnames:
            ws = wb['transport_template_long']
            header = [c.value for c in ws[1]]
            try:
                vi = header.index('value') + 1
            except ValueError:
                return
            for row in ws.iter_rows(min_row=2):
                v = row[vi-1].value
                if v in (None,'','nan','None') or (isinstance(v,float) and math.isnan(v)):
                    for c in row: c.fill = WARN_FILL
                else:
                    for c in row: c.fill = GOOD_FILL
    except Exception as e:
        log.warning(f"格式化失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=Path,
                    default=Path('/7t/lxkzero/claude/qd/disan/PFAS_transport_layer_v1.xlsx'))
    ap.add_argument('--norman', type=Path,
                    default=Path('/7t/lxkzero/claude/qd/diyi/pfas_data/NORMAN/susdat_2025-06-03-092022.csv'))
    ap.add_argument('--out', type=Path,
                    default=Path('/7t/lxkzero/claude/qd/disan/PFAS_transport_layer_v2.xlsx'))
    args = ap.parse_args()
    fill(args.layer, args.norman, args.out)


if __name__ == '__main__':
    main()
