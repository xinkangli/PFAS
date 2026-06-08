"""
extraction_tables_v4.xlsx — 把 SI 抽到的逐物种 δ15N / TL / lipid / 浓度 / TMF 全部合并

关键逻辑:
  ST003: SI text (无表) → 营养级范围、TMF 表(已知 ww/PW basis)→ 软抽
  ST007: SI 27 张表 → 浓度,无 SIA
  ST008: SI t13/t14/t15 浓度矩阵, t16 BSAF, t17-t19 BMF, t20 TMF (17 PFAS)
  ST011: HTML 全文 → fractional TL via Ecopath, 但具体数值在 Suppl Table S4
  ST012: SI t13 δ15N, t14 δ13C, t15 TL, t6 muscle conc, t7 liver conc, t8 crustacean conc
  ST013: review,各德国水体鱼 PFOS 浓度 + 文献 PFOS TMF 汇总
  ST015: 鱼类 PFAS 浓度,无食物网

输出 4 个 sheet:
  Table1 study_info (与 v3 同)
  Table2 concentration (大幅扩充)
  Table3 TMF (大幅扩充, 含 BMF/BAF/BSAF 字段)
  Table4 trophic_isotope (大幅扩充, 含 δ15N/δ13C/TL/lipid)
"""
import re
import pandas as pd

V3 = "pfas_ranked/extraction_tables_v3.xlsx"
OUT = "pfas_ranked/extraction_tables_v4.xlsx"

# ===== 读 v3 沿用 Table1 =====
table1 = pd.read_excel(V3, sheet_name="Table1_study_info")

# ============================================================
# Table 4: trophic_isotope rows
# ============================================================
isotope_rows = []

# ----- ST012 isotope (from SI t13/t14/t15) -----
xl12 = "/7t/lxkzero/claude/qd/dier/pfas_si_extracted/ST012_si_tables.xlsx"
df_d15 = pd.read_excel(xl12, sheet_name='p01_t13', header=0)
df_d13 = pd.read_excel(xl12, sheet_name='p01_t14', header=0)
df_tl  = pd.read_excel(xl12, sheet_name='p01_t15', header=0)
# 第一列是 station,header 是物种
station_col = df_d15.columns[0]
species_cols = [c for c in df_d15.columns[1:] if not str(c).startswith("Unnamed")]
for i in range(len(df_d15)):
    station = df_d15.iloc[i][station_col]
    if pd.isna(station): continue
    try: station = int(station)
    except: pass
    for sp in species_cols:
        v15 = df_d15.iloc[i][sp]
        v13 = df_d13.iloc[i][sp] if sp in df_d13.columns else None
        vtl = df_tl.iloc[i][sp] if sp in df_tl.columns else None
        if pd.isna(v15) and pd.isna(vtl): continue
        tissue = "whole body" if sp in ("C. crangon","L. holsatus") else "muscle"
        isotope_rows.append({
            "study_id":"ST012", "species":sp, "tissue":tissue,
            "station":station, "n_samples":"pooled (3-4 ind)",
            "delta15N_permil": float(v15) if pd.notna(v15) else "",
            "delta15N_sd": "",
            "delta13C_permil": float(v13) if pd.notna(v13) else "",
            "delta13C_sd": "",
            "lipid_pct": "",  # 该 SI 未给
            "lipid_pct_sd": "",
            "trophic_level": float(vtl) if pd.notna(vtl) else "",
            "source": "Byns 2022 SI Table S13/S14/S15",
        })

# ----- ST003 isotope (from SI text — 摘要 baseline 物种 + TL 范围) -----
# 文中正文范围: δ15N 6.1-21.1‰, TL 1-5; baseline: Fucus gardneri (TL=1), Mytilis edulis
ST003_iso = [
    ("Fucus gardneri (macroalgae)","whole",      "TL=1.0 (baseline)"),
    ("Mytilis edulis (bivalve)",   "whole",      "baseline (used to compute TLs)"),
    ("Mallotus villosus (capelin)","whole",      "TL ~3"),
    ("Boreogadus saida (Arctic cod)","whole",    "TL ~3.5"),
    ("Somateria mollissima sedentaria (eider)","liver","TL ~3.5-4"),
    ("Melanitta fusca (white-winged scoter)","liver","TL ~3.5-4"),
    ("Delphinapterus leucas (beluga)","liver",   "TL ~5 (top predator)"),
    ("Delphinapterus leucas (beluga)","blood",   "TL ~5"),
    ("Delphinapterus leucas (beluga)","muscle",  "TL ~5"),
    ("Delphinapterus leucas (beluga)","milk",    "TL ~5"),
]
for sp,tis,note in ST003_iso:
    isotope_rows.append({
        "study_id":"ST003","species":sp,"tissue":tis,
        "station":"Eastern Hudson Bay",
        "n_samples":"see SI Table",
        "delta15N_permil":"in 6.1–21.1 (range)","delta15N_sd":"",
        "delta13C_permil":"","delta13C_sd":"",
        "lipid_pct":"",  "lipid_pct_sd":"",
        "trophic_level": note,
        "source":"Tomy 2009 SI text (range only;具体逐物种值需查 Table S2/S3)",
    })

# ----- ST007 / ST011 / ST008 / ST013 / ST015: SI 中无 δ15N/TL 实测 -----
no_iso_studies = {
    "ST007":"无 SIA;Boulanger 2021 用定性 food-web structure",
    "ST008":"未做 SIA;Munoz 2022 用 aquatic plants 作 TL=1 baseline,GLMM 算 TMF",
    "ST011":"未做 SIA;Wang 2024 用 Ecopath 模型 (fractional TL 在 Suppl Table S4 — 我没拿到)",
    "ST013":"review,无原始 δ15N",
    "ST015":"只有鱼,无食物网,无 SIA",
}
for sid, note in no_iso_studies.items():
    isotope_rows.append({
        "study_id":sid,"species":"—","tissue":"—","station":"—",
        "n_samples":"—",
        "delta15N_permil":"not measured","delta15N_sd":"",
        "delta13C_permil":"not measured","delta13C_sd":"",
        "lipid_pct":"not measured","lipid_pct_sd":"",
        "trophic_level":note,"source":"see notes",
    })

# 其余 study (无 PDF)
covered_iso = {r["study_id"] for r in isotope_rows}
for sid in table1["study_id"]:
    if sid not in covered_iso:
        isotope_rows.append({"study_id":sid,"species":"TBD","tissue":"TBD","station":"TBD",
                             "n_samples":"TBD","delta15N_permil":"TBD","delta15N_sd":"",
                             "delta13C_permil":"TBD","delta13C_sd":"",
                             "lipid_pct":"TBD","lipid_pct_sd":"",
                             "trophic_level":"TBD","source":"PDF/SI unavailable"})

table4 = pd.DataFrame(isotope_rows)

# ============================================================
# Table 2: concentration rows (扩充)
# ============================================================
conc_rows = []

# ---------- ST012 浓度 (SI t6 muscle, t7 liver, t8 crustacean) ----------
def parse_st012_table(sn, tissue, source_label, pfas_cols=('PFOS','PFOA','PFNA','PFDA','PFunDA','PFDoDA','PFTrDA','PFUnDA')):
    df = pd.read_excel(xl12, sheet_name=sn, header=0)
    if "Location" in df.columns and "Species" in df.columns:
        loc_col, sp_col = "Location","Species"
    else:
        loc_col, sp_col = df.columns[0], df.columns[1]
    rows = []
    for _, r in df.iterrows():
        loc = r.get(loc_col); sp = r.get(sp_col)
        if pd.isna(loc) or pd.isna(sp) or str(sp).startswith("LOQ"):
            continue
        try: loc = int(loc)
        except: pass
        for pfas in pfas_cols:
            if pfas not in df.columns: continue
            v = r[pfas]
            if pd.isna(v): continue
            if isinstance(v,str) and ("ND" in v or "LOQ" in v):
                conc = v; n = "<LOQ"
            else:
                try: conc = float(v); n = ""
                except: conc = str(v); n = ""
            rows.append({
                "study_id":"ST012","species":str(sp),"tissue":tissue,
                "station":loc,"PFAS":pfas,"n_samples":1,
                "concentration_mean":conc,"concentration_sd":"",
                "concentration_min":"","concentration_max":"",
                "unit":"ng/g","basis":"ww","source":source_label,
            })
    return rows
conc_rows += parse_st012_table('p01_t6', 'muscle',     'Byns 2022 SI Table S7')
conc_rows += parse_st012_table('p01_t7', 'liver',      'Byns 2022 SI Table S8')
conc_rows += parse_st012_table('p01_t8', 'whole body', 'Byns 2022 SI Table S9')

# ---------- ST008 浓度 (SI t13/t14/t15: 23物种 × ~21 PFAS,值是 ng/g ww) ----------
xl08 = "/7t/lxkzero/claude/qd/dier/pfas_si_extracted/ST008_si_tables.xlsx"
def parse_st008_table(sn, source_label):
    df = pd.read_excel(xl08, sheet_name=sn, header=0)
    sp_col = df.columns[0]
    pfas_cols = [c for c in df.columns[1:] if not str(c).startswith("Unnamed")]
    rows = []
    for _, r in df.iterrows():
        sp = r[sp_col]
        if pd.isna(sp): continue
        sp = str(sp).strip()
        if "(mean)" in sp:
            tissue = "category mean"
        elif sp in ("Carex sp.","H. dubia","P. richardsonii"):
            tissue = "whole plant"
        elif sp in ("Crayfish","Gammarids","Insects","Molluscs (snails)"):
            tissue = "whole body"
        else:
            tissue = "whole-body fish (skull/viscera removed)"
        for pfas in pfas_cols:
            v = r[pfas]
            if pd.isna(v) or str(v).strip() in ("-","ND"):
                continue
            try: conc = float(v)
            except: conc = str(v)
            rows.append({
                "study_id":"ST008","species":sp,"tissue":tissue,
                "station":"St. Lawrence River, Canada",
                "PFAS":str(pfas),"n_samples":"",
                "concentration_mean":conc,"concentration_sd":"",
                "concentration_min":"","concentration_max":"",
                "unit":"ng/g","basis":"ww","source":source_label,
            })
    return rows
conc_rows += parse_st008_table('p01_t13','Munoz 2022 SI Table (PFCAs)')
conc_rows += parse_st008_table('p01_t14','Munoz 2022 SI Table (PFSAs)')
conc_rows += parse_st008_table('p01_t15','Munoz 2022 SI Table (FOSA/FBSA/PFECHS/diPAP/etc)')

# ---------- ST007 浓度 (大表,简化:从 v2 已有的浓度 + 不再重复) ----------
# 把 v2 已有的 ST007/ST011/ST014 conc 拷贝(单位已修正)
v2_conc = pd.read_excel("pfas_ranked/extraction_tables_v2.xlsx", sheet_name="Table2_concentration")
for _, r in v2_conc[v2_conc["study_id"].isin(["ST007","ST011","ST014"])].iterrows():
    conc_rows.append({
        "study_id":r["study_id"],"species":r["species"],"tissue":r["tissue"],
        "station":"","PFAS":r["PFAS"],"n_samples":"",
        "concentration_mean":r["concentration"],"concentration_sd":"",
        "concentration_min":"","concentration_max":"",
        "unit":r["unit"],"basis":"","source":r["source_table"],
    })

# ---------- ST013 浓度 (review 给的德国 Moselle/Havel 鱼 PFOS) ----------
ST013_conc = [
    ("chub","Moselle","fillet","PFOS",1.68,"µg/kg","ww"),
    ("roach","Moselle","fillet","PFOS",4.78,"µg/kg","ww"),
    ("perch","Moselle","fillet","PFOS",14.7,"µg/kg","ww"),
    ("roach","Havel","fillet","PFOS",2.4, "µg/kg","ww"),
    ("bream","Havel","fillet","PFOS",4.8, "µg/kg","ww"),
    ("perch","Havel","fillet","PFOS",10.5,"µg/kg","ww"),
]
for sp,loc,tis,pfas,c,u,b in ST013_conc:
    conc_rows.append({
        "study_id":"ST013","species":sp,"tissue":tis,"station":loc,
        "PFAS":pfas,"n_samples":"","concentration_mean":c,"concentration_sd":"",
        "concentration_min":"","concentration_max":"",
        "unit":u,"basis":b,"source":"Schaefer 2020 review SI/正文",
    })

# ---------- ST015 浓度 (Illinois 鱼) ----------
# 从 SI xlsx 大表抽
xl15 = "/7t/lxkzero/claude/qd/dier/pfas_si_extracted/ST015_si_tables.xlsx"
df15 = pd.read_excel(xl15, sheet_name='p01_t6', header=None)
# 该表是 71x23 — 推测是各采样点 × 各鱼种 × 各 PFAS。先把它整体导出查看
# 简单作法:把 v2 的 ST015 占位换成 一个 reference 行
conc_rows.append({
    "study_id":"ST015","species":"9 fish species (Illinois)","tissue":"fillet",
    "station":"Rock River + 3 other waterways (15 sites)","PFAS":"17 PFAS",
    "n_samples":"","concentration_mean":"see SI t5/t6 (53x14, 71x23)",
    "concentration_sd":"","concentration_min":"","concentration_max":"",
    "unit":"ng/g","basis":"ww","source":"Gewurtz 2024 SI Table 1+ (待逐行展开)",
})

# 其他 (PDF unavailable)
covered_c = {r["study_id"] for r in conc_rows}
for sid in table1["study_id"]:
    if sid not in covered_c:
        conc_rows.append({"study_id":sid,"species":"TBD","tissue":"TBD","station":"TBD",
                          "PFAS":"TBD","n_samples":"TBD",
                          "concentration_mean":"TBD","concentration_sd":"",
                          "concentration_min":"","concentration_max":"",
                          "unit":"TBD","basis":"TBD","source":"PDF/SI unavailable"})

table2 = pd.DataFrame(conc_rows)

# ============================================================
# Table 3: TMF / BMF / BSAF / BAF
# ============================================================
tmf_rows = []

# ---------- ST003 TMF (3 张表 ww / pw whole web / pw fish-only) ----------
ST003_tmf = [
    # Table S5 — TMF_ww 整食物网
    ("PFOS","17.4","[11.9–26.0]","whole food web (mammalian + fish)","ww","Tomy SI Table S5"),
    ("PFDA","8.29","[5.68–12.1]","whole food web","ww","Tomy SI Table S5"),
    ("PFUnA","7.98","[6.25–10.2]","whole food web","ww","Tomy SI Table S5"),
    ("PFNA","6.49","[4.43–9.50]","whole food web","ww","Tomy SI Table S5"),
    ("PFDoA","4.59","[3.57–5.91]","whole food web","ww","Tomy SI Table S5"),
    ("PFTA","2.53","[2.05–3.13]","whole food web","ww","Tomy SI Table S5"),
    ("PFOA","3.28","[2.27–4.74]","whole food web","ww","Tomy SI Table S5"),
    ("PFOSA","11.5","[7.36–17.9]","whole food web","ww","Tomy SI Table S5"),
    ("PFHpA","1.43","[1.02–2.00]","whole food web","ww","Tomy SI Table S5"),
    # Table S6 — TMF_pw 整食物网
    ("PFOS","11.0","[6.90–17.4]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFDA","4.81","[3.31–6.99]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFUnA","4.43","[3.50–5.61]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFNA","4.23","[2.90–6.19]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFDoA","2.53","[1.96–3.27]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFTA","1.41","[1.13–1.75]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFOA","1.93","[1.34–2.78]","whole food web","protein-corrected","Tomy SI Table S6"),
    ("PFOSA","6.06","[3.86–9.50]","whole food web","lipid-equivalent","Tomy SI Table S6"),
    # Table S7 — fish-only piscivorous web (TMF<1)
    ("PFOS","0.47","[0.27–0.85]","piscivorous fish-only food web","protein-corrected","Tomy SI Table S7"),
    ("PFDA","0.62","[0.40–0.96]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFUnA","0.55","[0.36–0.84]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFNA","0.45","[0.30–0.69]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFDoA","0.44","[0.30–0.65]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFTA","0.34","[0.23–0.48]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFOA","0.55","[0.37–0.81]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
    ("PFOSA","2.30","[1.49–3.55]","piscivorous fish-only","lipid-equivalent","Tomy SI Table S7"),
    ("PFHpA","0.83","[0.54–1.27]","piscivorous fish-only","protein-corrected","Tomy SI Table S7"),
]
for pfas,tmf,ci,fw,basis,src in ST003_tmf:
    tmf_rows.append({"study_id":"ST003","PFAS":pfas,"TMF":tmf,
                     "p_value":"","R2":"","slope":"","intercept":"",
                     "food_web":fw,"basis":basis,"notes":f"95% CI {ci}; {src}"})

# ---------- ST008 TMF (SI p01_t20, 17 行) ----------
df_t20 = pd.read_excel(xl08, sheet_name='p01_t20', header=0)
# col0=PFAS, col1=TMF, col2=95% CI, col3=AIC
for _, r in df_t20.iterrows():
    p = r.iloc[0]
    if pd.isna(p): continue
    tmf_rows.append({
        "study_id":"ST008","PFAS":str(p),"TMF":r.iloc[1],
        "p_value":"","R2":"","slope":"","intercept":"",
        "food_web":"St. Lawrence River (full web: aquatic plants → invert → fish)",
        "basis":"protein-normalized (GLMM censored)",
        "notes":f"95% CI {r.iloc[2]}; AIC={r.iloc[3]}; Munoz 2022 SI Table S14",
    })

# ---------- ST013 review 汇总 PFOS TMF ----------
ST013_tmf = [
    ("PFOS","5.9", "Lake Ontario (Houde 2004); pelagic; whole fish","ww","review"),
    ("PFOS","3.0", "Lake Mergozzo, Italy (Mazzoni 2020); fillet","ww","review"),
    ("PFOS","4.2±0.87","Lake Ontario (Houde 2008); TL-corrected (fish-fish)","ww","review"),
    ("PFOS","3.8±0.98","Lake Ontario (Houde 2008); TL-corrected (fish-invert)","ww","review"),
    ("PFOS","1.5", "Orge River, France (Simonnet-Laprade 2019); biofilm-macrophyte-invert-fish","ww","review"),
    ("PFOS","2.4–4.1 (geomean 2.9)","French rivers (Rhône/Furan/Lyules)","ww","review"),
    ("PFOS","2.60","Stream geometric mean (chosen for WFD norm.)","ww","Schaefer 2020 Table 3"),
    ("PFOS","3.13","Stream+lake geometric mean","ww","Schaefer 2020 Table 3"),
    ("PFOS","4.98","Lake geometric mean","ww","Schaefer 2020 Table 3"),
]
for pfas,tmf,fw,basis,src in ST013_tmf:
    tmf_rows.append({"study_id":"ST013","PFAS":pfas,"TMF":tmf,
                     "p_value":"","R2":"","slope":"","intercept":"",
                     "food_web":fw,"basis":basis,"notes":src})

# ---------- 其他 study TMF ----------
no_tmf_studies = {
    "ST007":"未计算 TMF (无 SIA);Boulanger 2021 报道 BAF/log BAF",
    "ST011":"未直接报道 TMF;Wang 2024 用 Ecopath 模型",
    "ST012":"SI 未含 TMF 表",
    "ST014":"摘要/正文未计算 TMF (只 2 营养级)",
    "ST015":"无食物网,未算 TMF",
    "ST002":"会议摘要,无数据",
}
for sid, note in no_tmf_studies.items():
    tmf_rows.append({"study_id":sid,"PFAS":"—","TMF":"not reported",
                     "p_value":"","R2":"","slope":"","intercept":"",
                     "food_web":"","basis":"","notes":note})

# 其他 study (无 PDF)
covered_t = {r["study_id"] for r in tmf_rows}
for sid in table1["study_id"]:
    if sid not in covered_t:
        tmf_rows.append({"study_id":sid,"PFAS":"TBD","TMF":"TBD",
                         "p_value":"TBD","R2":"TBD","slope":"","intercept":"",
                         "food_web":"TBD","basis":"TBD","notes":"PDF/SI unavailable"})

table3 = pd.DataFrame(tmf_rows)

# ============================================================
# Table 5 (新): BMF / BAF / BSAF
# ============================================================
extra_rows = []

# ---------- ST008 BSAF (p01_t16) ----------
df_t16 = pd.read_excel(xl08, sheet_name='p01_t16', header=1)
# row 0 = species names, col 0 = PFAS
df_t16_full = pd.read_excel(xl08, sheet_name='p01_t16', header=None)
species_t16 = df_t16_full.iloc[1, 1:].dropna().tolist()
for i in range(2, len(df_t16_full)):
    pfas = df_t16_full.iloc[i, 0]
    if pd.isna(pfas): continue
    for j, sp in enumerate(species_t16, start=1):
        v = df_t16_full.iloc[i, j]
        if pd.isna(v) or str(v).strip() == "-": continue
        try: val = float(v)
        except: val = str(v)
        extra_rows.append({
            "study_id":"ST008","metric":"BSAF","species":sp,"PFAS":str(pfas),
            "value":val,"basis":"biota ng/g ww / sediment ng/g dw",
            "source":"Munoz 2022 SI Table",
        })

# ---------- ST008 BMF (p01_t17/18/19, predator/prey rows) ----------
for sn, src in [('p01_t17','SI Table BMF (PFCAs)'),
                 ('p01_t18','SI Table BMF (PFSAs)'),
                 ('p01_t19','SI Table BMF (FOSA/FBSA/PFECHS)')]:
    df = pd.read_excel(xl08, sheet_name=sn, header=0)
    pp_col = df.columns[0]
    pfas_cols = list(df.columns[1:])
    for _, r in df.iterrows():
        pp = r[pp_col]
        if pd.isna(pp) or pp == "Predator/Prey": continue
        for pfas in pfas_cols:
            v = r[pfas]
            if pd.isna(v) or str(v).strip() == "-": continue
            try: val = float(v)
            except: val = str(v)
            extra_rows.append({
                "study_id":"ST008","metric":"BMF","species":str(pp),"PFAS":str(pfas),
                "value":val,"basis":"predator ng/g ww / prey ng/g ww",
                "source":f"Munoz 2022 {src}",
            })

# ---------- ST007 BAF (12 行) ----------
import json
js = json.load(open("/7t/lxkzero/claude/qd/dier/pfas_si_extracted/ST007_extracted_data.json"))
for r in js.get("ST007_baf", []):
    extra_rows.append({
        "study_id":"ST007","metric":"BAF (or log BAF)",
        "species":r.get("species"),"PFAS":r.get("PFAS"),
        "value": r.get("BAF") or r.get("log_BAF") or "",
        "basis":r.get("basis","ww"),"source":r.get("source",""),
    })

table5 = pd.DataFrame(extra_rows)

# ============================================================
# 写入 v4
# ============================================================
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    table1.to_excel(w, sheet_name="Table1_study_info", index=False)
    table2.to_excel(w, sheet_name="Table2_concentration", index=False)
    table3.to_excel(w, sheet_name="Table3_TMF", index=False)
    table4.to_excel(w, sheet_name="Table4_trophic_isotope", index=False)
    table5.to_excel(w, sheet_name="Table5_BAF_BMF_BSAF", index=False)

print(f"Wrote: {OUT}")
print(f"  Table1: {len(table1)} studies × {len(table1.columns)} cols")
print(f"  Table2: {len(table2)} concentration rows")
print(f"          - ST008 (St Lawrence): {(table2['study_id']=='ST008').sum()}")
print(f"          - ST012 (Belgian): {(table2['study_id']=='ST012').sum()}")
print(f"          - ST007 (from v2): {(table2['study_id']=='ST007').sum()}")
print(f"          - ST011 (from v2): {(table2['study_id']=='ST011').sum()}")
print(f"          - ST013 review: {(table2['study_id']=='ST013').sum()}")
print(f"          - ST014 (from v2): {(table2['study_id']=='ST014').sum()}")
print(f"  Table3: {len(table3)} TMF rows")
print(f"          - ST003 (3 baselines): {(table3['study_id']=='ST003').sum()}")
print(f"          - ST008 (17 PFAS GLMM): {(table3['study_id']=='ST008').sum()}")
print(f"          - ST013 (literature PFOS): {(table3['study_id']=='ST013').sum()}")
print(f"  Table4: {len(table4)} trophic+isotope rows")
print(f"          - ST012 (full δ15N+δ13C+TL matrix): {(table4['study_id']=='ST012').sum()}")
print(f"          - ST003 (text-derived): {(table4['study_id']=='ST003').sum()}")
print(f"  Table5: {len(table5)} BAF/BMF/BSAF rows")
print(f"          - ST008 BMF/BSAF: {(table5['study_id']=='ST008').sum()}")
print(f"          - ST007 BAF: {(table5['study_id']=='ST007').sum()}")
