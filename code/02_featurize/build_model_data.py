"""Build 3 model-ready Excel files in model_data/ by joining the 5-layer data.

Outputs:
  model_data/Model1_Transport.xlsx
  model_data/Model2_Bioaccumulation.xlsx
  model_data/Model3_Persistence.xlsx
"""
import os
import sqlite3
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = "/7t/lxkzero/claude/qd/data"
OUT = os.path.join(ROOT, "model_data")
os.makedirs(OUT, exist_ok=True)


def normalize_cas(s):
    if pd.isna(s):
        return None
    return str(s).strip().upper().replace(" ", "")


def read_csv_ragged(path, expected_ncols):
    """Read a CSV where the SECOND column (name) may contain unquoted commas.
    Merges extra fields back into column index 1."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if len(r) > expected_ncols:
                extra = len(r) - expected_ncols
                # merge fields [1 .. 1+extra] back into name (col 1)
                name = ",".join(r[1:2 + extra])
                r = [r[0]] + [name] + r[2 + extra:]
            rows.append(r)
    return pd.DataFrame(rows, columns=header)


def write_readme(writer, lines, sheet_name="README"):
    """Write a styled README sheet as the first sheet."""
    wb = writer.book
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name, 0)
    ws["A1"] = lines[0]
    ws["A1"].font = Font(bold=True, size=14)
    for i, line in enumerate(lines[1:], 2):
        ws.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 110


def autosize(ws, max_w=50):
    for col in ws.columns:
        # skip merged
        try:
            length = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        except Exception:
            length = 12
        ws.column_dimensions[col[0].column_letter].width = min(max(length + 2, 10), max_w)


def style_header(ws):
    fill = PatternFill("solid", fgColor="4472C4")
    font = Font(bold=True, color="FFFFFF")
    for c in ws[1]:
        c.fill = fill
        c.font = font
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"


# =====================================================================
# Load all source data
# =====================================================================
print("Loading source data ...")

# Layer 1 — F-count features and unique smiles
fcnt = pd.read_csv(f"{ROOT}/01_structure/qd_smiles_F_stats_per_mol.csv")
usmi = pd.read_csv(f"{ROOT}/01_structure/qd_unique_smiles.csv")

# Layer 3 — master structure (X pool) + transport template
master = pd.read_csv(f"{ROOT}/03_transport/PFAS_master_structure.csv", low_memory=False)
master["CAS_norm"] = master["CAS"].map(normalize_cas)
print(f"  master structure: {len(master):,} rows")

tpath = f"{ROOT}/03_transport/PFAS_transport_layer_v2.xlsx"
target_list = pd.read_excel(tpath, sheet_name="target_pfas_list")
trans_long = pd.read_excel(tpath, sheet_name="transport_template_long")
trans_wide = pd.read_excel(tpath, sheet_name="transport_template_wide")
miss_prio = pd.read_excel(tpath, sheet_name="missing_priority")
print(f"  transport Y: 25 PFAS, long={len(trans_long)}, wide={len(trans_wide)}")

# Layer 4 — main Y
con = sqlite3.connect(f"{ROOT}/04_bioaccumulation/pfas_master.sqlite")
bcf = pd.read_sql_query("select * from pfas_bcf_master", con)
con.close()
bcf["CAS_norm"] = bcf["cas"].map(normalize_cas)
print(f"  pfas_bcf_master: {len(bcf):,} rows")

# Layer 2 — TMF + Table5 supplement
ext_path = f"{ROOT}/02_literature_evidence/extracted/extraction_tables_v5.xlsx"
tmf = pd.read_excel(ext_path, sheet_name="Table3_TMF")
t5 = pd.read_excel(ext_path, sheet_name="Table5_BAF_BMF_BSAF")
print(f"  TMF: {len(tmf)} rows, Table5_BAF_BMF_BSAF: {len(t5)} rows")

# Layer 5 — six tables
L5 = f"{ROOT}/05_persistence_transformation"
compounds = read_csv_ragged(f"{L5}/compounds.csv", expected_ncols=8)
persistence = pd.read_csv(f"{L5}/persistence.csv")
precursor = pd.read_csv(f"{L5}/precursor_transformation.csv")
chem_l5 = pd.read_csv(f"{L5}/chem_properties.csv")
env_cond = pd.read_csv(f"{L5}/env_media_conditions.csv")
sources = pd.read_csv(f"{L5}/sources.csv")
compounds["CAS_norm"] = compounds["cas"].map(normalize_cas)
print(f"  Layer 5: compounds={len(compounds)}, persistence={len(persistence)}, "
      f"precursor={len(precursor)}, chem={len(chem_l5)}, env={len(env_cond)}, sources={len(sources)}")


# =====================================================================
# Model 1 — Transport Foundation Model
# =====================================================================
print("\n[Model 1] Building Transport Excel ...")

# X pool: master structure + F-counts joined on SMILES
x_pool = master.merge(fcnt, on="SMILES", how="left")
# silver-label subset: rows with at least one of XLogP / logKow_Norman
y_silver = x_pool[(x_pool["XLogP"].notna()) | (x_pool["logKow_Norman"].notna())].copy()

# matched gold X for the 25 target PFAS — target_list already has SMILES + class + XLogP,
# only need to top up with logKow_Norman from master (joined on CAS) and F-stats (on SMILES).
target_list["CAS_norm"] = target_list["CAS"].map(normalize_cas)
master_slim = (master[["CAS_norm", "logKow_Norman", "MW", "primary_source"]]
                 .dropna(subset=["CAS_norm"])
                 .drop_duplicates(subset=["CAS_norm"]))
target_x = target_list.merge(master_slim, on="CAS_norm", how="left")
target_x = target_x.merge(fcnt, on="SMILES", how="left")
target_x = target_x.drop(columns=["CAS_norm"])
print(f"  target_25_X: {len(target_x)} rows (should stay 25)")

out1 = os.path.join(OUT, "Model1_Transport.xlsx")
with pd.ExcelWriter(out1, engine="openpyxl") as w:
    # README first
    pd.DataFrame({"_": [""]}).to_excel(w, sheet_name="README", index=False)
    # X pool — full structure features
    x_pool.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="X_pool_full", index=False)
    # Silver labels for weak supervision
    y_silver.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="Y_silver_logKow", index=False)
    # Gold target list X
    target_x.to_excel(w, sheet_name="target_25_X", index=False)
    # Gold Y — long form (200 rows: 25 × 8 endpoints)
    trans_long.to_excel(w, sheet_name="Y_gold_long", index=False)
    # Gold Y — wide form (25 rows)
    trans_wide.to_excel(w, sheet_name="Y_gold_wide", index=False)
    # endpoint priority
    miss_prio.to_excel(w, sheet_name="missing_priority", index=False)
    # PFAS-specific F-count feature dictionary
    fcnt.to_excel(w, sheet_name="F_stats_dict", index=False)

# Post-style README + headers
from openpyxl import load_workbook
wb = load_workbook(out1)
readme_lines = [
    "Model 1 — Transport Foundation Model · 数据包",
    "",
    "【任务】 multi-task regression：SMILES → logKow / logKoc / pKa / sol / Henry / vp",
    "",
    "【Sheet 速查】",
    "  • X_pool_full        — 主结构池 127,993 行，13 列 + 5 列氟原子计数 (F-stats)",
    "                         join 自 03_transport/PFAS_master_structure.csv",
    "                         + 01_structure/qd_smiles_F_stats_per_mol.csv (on SMILES)",
    "  • Y_silver_logKow    — 含 XLogP 或 logKow_Norman 的子集，做 logKow head 的弱监督预训练",
    "  • target_25_X        — 25 个目标 PFAS 的 X 特征（已 join 主表 + F-stats）",
    "  • Y_gold_long        — 25 PFAS × 8 endpoints 的长表 (multi-task 用)",
    "  • Y_gold_wide        — 25 行宽表，每个 endpoint 一列",
    "  • missing_priority   — 哪些 endpoint 必须先补，决定 multi-task 的权重",
    "  • F_stats_dict       — n_F / n_CF1-4 字典 (PFAS-specific 特征)",
    "",
    "【训练顺序建议】",
    "  1. 用 Y_silver_logKow 做 logKow 单任务的弱监督预训练 (~38k-128k 样本)",
    "  2. 再用 Y_gold_long fine-tune 8 个 endpoint head (25 PFAS × 8 = 200 行)",
    "",
    "【ID 主键】",
    "  CAS (规范化为大写无空格) 是首选 join key；InChIKey 兜底；",
    "  跨源 join 不要用 DTXSID / PubChemCID — 只在原源内一致。",
    "",
    "【额外要派生的 PFAS-specific 特征】（脚本里没造，请用 RDKit SMARTS 在 SMILES 上派生）",
    "  - head group (羧酸 / 磺酸 / 醚类)",
    "  - ether bridge count (GenX 类)",
    "  - branching index",
    "  - aromatic fluorination",
    "  - precursor flag",
]
ws = wb["README"]
ws.delete_rows(1, ws.max_row)
ws["A1"] = readme_lines[0]
ws["A1"].font = Font(bold=True, size=14)
for i, line in enumerate(readme_lines[1:], 2):
    ws.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 110

for sn in wb.sheetnames:
    if sn == "README":
        continue
    style_header(wb[sn])
wb.save(out1)
print(f"  wrote {out1}")


# =====================================================================
# Model 2 — Bioaccumulation Foundation Model (论文核心)
# =====================================================================
print("\n[Model 2] Building Bioaccumulation Excel ...")

# Y_all: full 2404 rows
y_all = bcf.copy()

# Y per endpoint
y_bcf = bcf[bcf["endpoint_type"] == "BCF"].copy()
y_baf = bcf[bcf["endpoint_type"] == "BAF"].copy()
y_bcfd = bcf[bcf["endpoint_type"] == "BCFD"].copy()

# X joined: Y left join structure features by CAS
struct_cols = ["name", "CAS", "InChIKey", "DTXSID", "PubChemCID", "SMILES",
               "MolFormula", "MW", "pfas_class", "XLogP", "logKow_Norman",
               "primary_source"]
x_struct = (master[["CAS_norm"] + struct_cols]
              .dropna(subset=["CAS_norm"])
              .drop_duplicates(subset=["CAS_norm"]))
fcnt_for_join = fcnt.rename(columns={"SMILES": "SMILES_struct"})

joined = bcf.merge(x_struct, on="CAS_norm", how="left", suffixes=("", "_struct"))
# add F-counts via SMILES (from joined struct table)
joined = joined.merge(fcnt, left_on="SMILES", right_on="SMILES", how="left")
print(f"  X_joined: {len(joined):,} rows, "
      f"{joined['SMILES'].notna().sum():,} 有 SMILES (用于结构特征派生)")

# pfas-level summary
summary_by_pfas = (bcf.groupby(["pfas_short", "endpoint_type"])
                     .agg(n=("value", "size"),
                          log_value_median=("log_value", "median"),
                          log_value_min=("log_value", "min"),
                          log_value_max=("log_value", "max"))
                     .reset_index()
                     .sort_values(["pfas_short", "endpoint_type"]))

out2 = os.path.join(OUT, "Model2_Bioaccumulation.xlsx")
with pd.ExcelWriter(out2, engine="openpyxl") as w:
    pd.DataFrame({"_": [""]}).to_excel(w, sheet_name="README", index=False)
    # Joined training table (★ 直接用这个建模)
    joined.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="X_Y_joined", index=False)
    # Y per endpoint
    y_baf.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="Y_BAF_1381", index=False)
    y_bcf.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="Y_BCF_880", index=False)
    y_bcfd.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="Y_BCFD_143", index=False)
    # Y all (raw)
    y_all.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="Y_all_2404", index=False)
    # Layer 2 supplement
    t5.to_excel(w, sheet_name="evidence_Table5_BAF_BMF_BSAF", index=False)
    # PFAS-level summary
    summary_by_pfas.to_excel(w, sheet_name="summary_by_pfas", index=False)

wb = load_workbook(out2)
readme_lines = [
    "Model 2 — Bioaccumulation Foundation Model · 数据包  (★ 论文核心)",
    "",
    "【任务】 multi-task regression：Structure + Transport + 生物上下文 → logBCF / logBAF / logBCFD",
    "",
    "【★ 直接用 X_Y_joined 这一张做训练 ★】",
    "  X_Y_joined 是把 pfas_bcf_master (2,404 行) 左连了：",
    "    • 03_transport/PFAS_master_structure.csv (按 CAS 拿 SMILES / MW / XLogP / logKow_Norman / pfas_class)",
    "    • 01_structure/qd_smiles_F_stats_per_mol.csv (按 SMILES 拿 n_F, n_CF1-4)",
    "  Y 列：value / log_value / endpoint_type",
    "  生物上下文列：species_sci / organism_group / tissue / exposure_days / temperature_c / water_type ...",
    "  ★ 别忘了 reliability 列 — 直接拿来做 sample weight",
    "",
    "【Sheet 速查】",
    "  • X_Y_joined                  — 训练表：2,404 行 Y 已经 join 好结构特征",
    "  • Y_BAF_1381 / Y_BCF_880 / Y_BCFD_143   — 按 endpoint 拆好的子集",
    "  • Y_all_2404                  — 原始 Y 主表",
    "  • evidence_Table5_BAF_BMF_BSAF — 文献抽取的 BAF/BMF/BSAF 441 条，可作补样本或证据加权",
    "  • summary_by_pfas             — 每个 PFAS × endpoint 的中位数 / 极值",
    "",
    "【Transport features 怎么加？】",
    "  X_Y_joined 现在只有 XLogP / logKow_Norman 两列 transport-相关；",
    "  建议跑完 Model 1 后，把模型 1 的预测值 (logKow / logKoc / pKa / sol / Henry / vp)",
    "  再 join 到这张表上 — 这就是 'structure → transport → bioaccumulation' 的核心链路。",
    "",
    "【数据分布提醒】",
    "  by_source : ECOTOX 856 + ITRC 1,548",
    "  by_endpoint: BAF 1,381 / BCF 880 / BCFD 143",
    "  Top 5 PFAS: PFOA 629 / PFOS 472 / PFDA 170 / PFNA 160 / PFHxS 144",
    "  Y 严重偏向 PFOA/PFOS — 训练时建议按 pfas_short 分层抽样或加 class weight",
    "",
    "【ID 主键】",
    "  CAS 是首选 join key；本表已用规范化 CAS 把结构特征左连进来。",
]
ws = wb["README"]
ws.delete_rows(1, ws.max_row)
ws["A1"] = readme_lines[0]
ws["A1"].font = Font(bold=True, size=14)
for i, line in enumerate(readme_lines[1:], 2):
    ws.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 115

for sn in wb.sheetnames:
    if sn == "README":
        continue
    style_header(wb[sn])
wb.save(out2)
print(f"  wrote {out2}")


# =====================================================================
# Model 3 — Persistence + Transformation 机制发现
# =====================================================================
print("\n[Model 3] Building Persistence Excel ...")

# cross table: persistence × pfas_bcf_master (high-P + high-BCF)
# join key: try abbreviation/pfas_short
pfas_summary = (bcf.groupby("pfas_short")
                  .agg(n_BCF=("endpoint_type", lambda s: (s == "BCF").sum()),
                       n_BAF=("endpoint_type", lambda s: (s == "BAF").sum()),
                       median_log_BCF=("log_value",
                                       lambda s: bcf.loc[s.index]
                                                    .query("endpoint_type=='BCF'")["log_value"].median()),
                       median_log_BAF=("log_value",
                                       lambda s: bcf.loc[s.index]
                                                    .query("endpoint_type=='BAF'")["log_value"].median()),
                       cas=("cas", "first"))
                  .reset_index())
pfas_summary["CAS_norm"] = pfas_summary["cas"].map(normalize_cas)

# persistence side
pers_compound = persistence.merge(compounds[["compound_id", "name", "cas", "family",
                                              "carbon_chain", "functional_group", "role",
                                              "CAS_norm"]],
                                   on="compound_id", how="left",
                                   suffixes=("_pers", "_cmp"))
# join with pfas_summary by CAS_norm
risk_table = pers_compound.merge(pfas_summary, on="CAS_norm", how="left",
                                  suffixes=("", "_bio"))

# transformation network edge table — already long form, just add node attrs
edges = precursor.merge(
    compounds[["compound_id", "abbreviation", "family", "role"]]
        .rename(columns={"compound_id": "precursor_id",
                         "abbreviation": "precursor_abbr",
                         "family": "precursor_family",
                         "role": "precursor_role"}),
    on="precursor_id", how="left",
)
edges = edges.merge(
    compounds[["compound_id", "abbreviation", "family", "role"]]
        .rename(columns={"compound_id": "terminal_id",
                         "abbreviation": "terminal_abbr",
                         "family": "terminal_family",
                         "role": "terminal_role"}),
    on="terminal_id", how="left",
)

out3 = os.path.join(OUT, "Model3_Persistence.xlsx")
with pd.ExcelWriter(out3, engine="openpyxl") as w:
    pd.DataFrame({"_": [""]}).to_excel(w, sheet_name="README", index=False)
    # 核心交叉表 — 高持久 × 高富集
    risk_table.to_excel(w, sheet_name="risk_persistence_x_BCF", index=False)
    # 单层各表
    compounds.drop(columns=["CAS_norm"], errors="ignore").to_excel(
        w, sheet_name="compounds_104", index=False)
    persistence.to_excel(w, sheet_name="persistence_129", index=False)
    edges.to_excel(w, sheet_name="transformation_edges_90", index=False)
    chem_l5.to_excel(w, sheet_name="chem_properties_l5_110", index=False)
    env_cond.to_excel(w, sheet_name="env_conditions_52", index=False)
    sources.to_excel(w, sheet_name="sources_60", index=False)
    # TMF axis (from Layer 2)
    tmf.to_excel(w, sheet_name="TMF_axis_l2_133", index=False)

wb = load_workbook(out3)
readme_lines = [
    "Model 3 — Persistence + Transformation 机制发现 · 数据包",
    "",
    "【任务】 不监督，做 clustering / UMAP / SHAP / 风险评分 / 转化网络",
    "",
    "【★ 核心交叉表 risk_persistence_x_BCF ★】",
    "  把 persistence.csv (129 行) → join compounds.csv 拿到 CAS",
    "  → 再 join Model 2 主表的 PFAS-level summary (median log BCF/BAF)",
    "  得到「高持久 × 高富集」的双轴评分表，对应你列出的 Q1。",
    "",
    "【Sheet 速查】",
    "  • risk_persistence_x_BCF       — ★ Q1 双高风险表（persistence × log_BCF/BAF 中位数）",
    "  • compounds_104                — 化合物主名单 (family / role / chain)",
    "  • persistence_129              — 半衰期 + persistence_label (vP / P / not-P_self)",
    "  • transformation_edges_90      — 前体→终端边表（已 join 节点的 family/role）→ 网络分析",
    "  • chem_properties_l5_110       — Layer 5 理化（与 Layer 3 重叠，优先用 Model 1 预测）",
    "  • env_conditions_52            — 持久性条件校正 (pH/T/redox/...)",
    "  • sources_60                   — 引用清单 + type (peer-reviewed/regulatory/...)",
    "  • TMF_axis_l2_133              — ★ 食物链放大金标签 (从 Layer 2 抽来)",
    "                                   关键列：PFAS / log_TMF / CI_low/high / data_provenance / DOI",
    "",
    "【4 个核心问题的数据组合】",
    "  Q1 高持久 + 高富集：       risk_persistence_x_BCF (已 join 好)",
    "  Q2 前体转化路径：          transformation_edges_90 → networkx 构图",
    "  Q3 long-range transport：  需要 Model 1 预测的 logKoa / 半衰期 + SHAP",
    "  Q4 优先管控名单：          risk_persistence_x_BCF + TMF_axis_l2_133 综合打分",
    "",
    "【提醒】",
    "  • NA = 文献未报告，不是脏数据，原因记在 notes 列",
    "  • 第五层样本量小 (52-129)，不要做监督深度学习",
    "  • TMF 只有 ~30 个 PFAS 有数据，做 zero-shot 推断更合适，不要硬塞监督",
]
ws = wb["README"]
ws.delete_rows(1, ws.max_row)
ws["A1"] = readme_lines[0]
ws["A1"].font = Font(bold=True, size=14)
for i, line in enumerate(readme_lines[1:], 2):
    ws.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 115

for sn in wb.sheetnames:
    if sn == "README":
        continue
    style_header(wb[sn])
wb.save(out3)
print(f"  wrote {out3}")

print("\nAll done.")
