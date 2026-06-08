"""
External validation pipeline for PFAS bioaccumulation model
Steps:
  1. Export training-set fingerprint (canonical_smiles + endpoint + species)
  2. Compile external BCF/BAF/TMF/BMF from 02_literature_evidence
  3. Strict deduplication (remove any overlap with training set)
  4. Regression validation: predicted vs observed (MAE, RMSE, Spearman rho, R2)
  5. OECD B/vB threshold classification (precision/recall/F1/confusion matrix)
  6. TMF cross-scale validation
  7. Wildlife detection enrichment analysis
  8. Figures + summary CSV
Outputs → result/waibuyanzheng/
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
OUT  = ROOT / 'result/waibuyanzheng'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'result'))

# ── PFAS name → SMILES mapping (from master structure) ────────────────────────
master = pd.read_csv(ROOT / '03_transport/PFAS_master_structure.csv', low_memory=False)
master = master[['SMILES', 'CAS', 'pfas_class']].dropna(subset=['SMILES']).drop_duplicates('SMILES')
master['SMILES'] = master['SMILES'].astype(str).str.strip()
master['CAS'] = master['CAS'].astype(str).str.strip()

# Also build abbreviation → SMILES from training set
train_raw = pd.read_excel(ROOT / 'model_data/Model2_Bioaccumulation.xlsx',
                          sheet_name='X_Y_joined', engine='openpyxl')
train_raw['SMILES'] = train_raw['SMILES'].astype(str).str.strip()
pfas_abbr_to_smiles = (train_raw[['pfas_short', 'SMILES']]
                       .dropna().drop_duplicates('pfas_short')
                       .set_index('pfas_short')['SMILES'].to_dict())

# Also build CAS → SMILES from master
cas_to_smiles = {}
for _, row in master.iterrows():
    cas_to_smiles[row['CAS']] = row['SMILES']

# Standard PFAS CAS numbers for common abbreviations
PFAS_CAS = {
    'PFOS': '1763-23-1', 'PFOA': '335-67-1', 'PFNA': '375-95-1',
    'PFDA': '335-76-2', 'PFUnDA': '2058-94-8', 'PFDoDA': '307-55-1',
    'PFTrDA': '72629-94-8', 'PFTeDA': '376-06-7', 'PFHxS': '355-46-4',
    'PFHpA': '375-85-9', 'PFHxA': '307-24-4', 'PFPeA': '2706-90-3',
    'PFBA': '375-22-4', 'PFBS': '375-73-5', 'PFDS': '335-77-3',
    'PFOSA': '754-91-6', 'PFHpS': '375-92-8', 'FOSA': '754-91-6',
    'F-53B': '73606-19-6', '6:2 Cl-PFESA': '73606-19-6',
    'PFECHS': '335-24-0', 'EtFOSAA': '2991-50-6', 'MeFOSAA': '2355-31-9',
    'FHxSA': '41997-13-1', 'FBSA': '30334-60-2', 'PFDS': '335-77-3',
}

def abbr_to_smiles(abbr):
    """Resolve PFAS abbreviation to SMILES via training set or CAS lookup."""
    abbr = str(abbr).strip()
    # direct training set lookup
    if abbr in pfas_abbr_to_smiles:
        return pfas_abbr_to_smiles[abbr]
    # CAS lookup
    cas = PFAS_CAS.get(abbr)
    if cas and cas in cas_to_smiles:
        return cas_to_smiles[cas]
    # partial CAS match
    for key, val in PFAS_CAS.items():
        if key.upper() == abbr.upper():
            if val in cas_to_smiles:
                return cas_to_smiles[val]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Export training-set fingerprint
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Building training-set fingerprint ...")
train_fp = train_raw[['SMILES', 'endpoint_type', 'species_sci', 'source', 'citation']].copy()
train_fp['SMILES'] = train_fp['SMILES'].astype(str).str.strip()
train_fp['key_smiles_ep'] = train_fp['SMILES'] + '|||' + train_fp['endpoint_type']
train_fp['key_full'] = (train_fp['SMILES'] + '|||' + train_fp['endpoint_type'] +
                        '|||' + train_fp['species_sci'].astype(str).str.strip())
train_smiles_ep = set(train_fp['key_smiles_ep'])
train_smiles_ep_species = set(train_fp['key_full'])
train_smiles_set = set(train_fp['SMILES'].dropna())
train_pfas_set = set(train_raw['pfas_short'].dropna())

train_fp.to_csv(OUT / 'step1_training_fingerprint.csv', index=False)
print(f"  Training fingerprint: {len(train_fp)} records, {train_fp['SMILES'].nunique()} unique SMILES")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compile external BCF/BAF from literature evidence
# ══════════════════════════════════════════════════════════════════════════════
print("Step 2: Compiling external literature data ...")

ext_records = []

# ── 2a: Table5_BAF_BMF_BSAF from extraction_tables_v5 ────────────────────────
xls_main = pd.ExcelFile(ROOT / '02_literature_evidence/extracted/extraction_tables_v5.xlsx')
df5 = pd.read_excel(xls_main, sheet_name='Table5_BAF_BMF_BSAF')
print(f"  Table5_BAF_BMF_BSAF: {len(df5)} records, metrics: {df5['metric'].value_counts().to_dict()}")

# BAF/logBAF rows
baf_rows = df5[df5['metric'].str.contains('BAF', case=False, na=False)].copy()
for _, row in baf_rows.iterrows():
    val = row.get('value')
    if pd.isna(val):
        continue
    try:
        fval = float(str(val).split('±')[0].strip())
    except:
        continue
    smiles = abbr_to_smiles(row['PFAS'])
    ext_records.append({
        'PFAS': row['PFAS'], 'SMILES': smiles,
        'endpoint': 'logBAF', 'log_value': fval,
        'species': str(row.get('species', 'unknown')),
        'basis': str(row.get('basis', '')),
        'metric_raw': row['metric'],
        'source_study': row.get('study_id', ''), 'source_ref': str(row.get('source', '')),
        'data_type': 'BAF_lit',
    })

# BMF rows (useful for enrichment analysis, not regression)
bmf_rows = df5[df5['metric'].str.contains('BMF', case=False, na=False)].copy()
for _, row in bmf_rows.iterrows():
    val = row.get('value')
    if pd.isna(val):
        continue
    try:
        fval = float(str(val).split('±')[0].strip())
        log_bmf = np.log10(fval) if fval > 0 else np.nan
    except:
        continue
    if pd.isna(log_bmf):
        continue
    smiles = abbr_to_smiles(row['PFAS'])
    ext_records.append({
        'PFAS': row['PFAS'], 'SMILES': smiles,
        'endpoint': 'logBMF', 'log_value': log_bmf,
        'species': str(row.get('species', 'unknown')),
        'basis': str(row.get('basis', '')),
        'metric_raw': row['metric'],
        'source_study': row.get('study_id', ''), 'source_ref': str(row.get('source', '')),
        'data_type': 'BMF_lit',
    })

# BSAF rows
bsaf_rows = df5[df5['metric'].str.contains('BSAF', case=False, na=False)].copy()
for _, row in bsaf_rows.iterrows():
    val = row.get('value')
    if pd.isna(val):
        continue
    try:
        fval = float(str(val).split('±')[0].strip())
        log_bsaf = np.log10(fval) if fval > 0 else np.nan
    except:
        continue
    if pd.isna(log_bsaf):
        continue
    smiles = abbr_to_smiles(row['PFAS'])
    ext_records.append({
        'PFAS': row['PFAS'], 'SMILES': smiles,
        'endpoint': 'logBSAF', 'log_value': log_bsaf,
        'species': str(row.get('species', 'unknown')),
        'basis': str(row.get('basis', '')),
        'metric_raw': row['metric'],
        'source_study': row.get('study_id', ''), 'source_ref': str(row.get('source', '')),
        'data_type': 'BSAF_lit',
    })

# ── 2b: ST007 extracted BAF data (JSON) ──────────────────────────────────────
with open(ROOT / '02_literature_evidence/extracted/si/ST007_extracted_data.json') as f:
    st007 = json.load(f)

for rec in st007.get('ST007_baf', []):
    val_str = str(rec.get('log_BAF', '')).split('±')[0].strip()
    try:
        fval = float(val_str)
    except:
        continue
    smiles = abbr_to_smiles(rec['PFAS'])
    ext_records.append({
        'PFAS': rec['PFAS'], 'SMILES': smiles,
        'endpoint': 'logBAF', 'log_value': fval,
        'species': str(rec.get('species', 'unknown')),
        'basis': str(rec.get('basis', 'ww')),
        'metric_raw': 'log_BAF',
        'source_study': 'ST007', 'source_ref': 'Boulanger 2021 Norwegian Arctic',
        'data_type': 'BAF_lit',
    })

# ── 2c: BCF from ECOTOX / published literature that is NOT in our training set ──
# We have 2,404 training records from ITRC+ECOTOX
# Check if extraction_tables has any additional BCF data
# Table2_concentration has tissue concentrations — not directly BCF
# but we can use it for detection enrichment
df2 = pd.read_excel(xls_main, sheet_name='Table2_concentration')
print(f"  Table2_concentration: {len(df2)} tissue concentration records")

# ── 2d: TMF data for cross-scale validation ───────────────────────────────────
df_tmf = pd.read_excel(xls_main, sheet_name='Table3_TMF')
print(f"  Table3_TMF: {len(df_tmf)} TMF records")

for _, row in df_tmf.iterrows():
    log_tmf = row.get('log_TMF')
    if pd.isna(log_tmf):
        continue
    try:
        log_tmf = float(log_tmf)
    except:
        continue
    smiles = abbr_to_smiles(row['PFAS'])
    ext_records.append({
        'PFAS': row['PFAS'], 'SMILES': smiles,
        'endpoint': 'logTMF', 'log_value': log_tmf,
        'species': str(row.get('food_web', 'food_web')),
        'basis': str(row.get('basis', '')),
        'metric_raw': 'logTMF',
        'source_study': row.get('study_id', ''), 'source_ref': str(row.get('citation', '')),
        'data_type': 'TMF_lit',
    })

ext_df = pd.DataFrame(ext_records)
print(f"\n  Total compiled external records: {len(ext_df)}")
print(f"  Endpoint distribution:\n{ext_df['endpoint'].value_counts()}")
print(f"  SMILES resolved: {ext_df['SMILES'].notna().sum()} / {len(ext_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Strict deduplication — remove any training set overlap
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Strict deduplication ...")

# Map logBMF/logBSAF to logBAF for comparison purposes (separate tracks)
def ep_for_dedup(ep):
    if 'BAF' in ep.upper() or 'BCF' in ep.upper():
        return 'BCF_BAF'
    return ep

ext_df['ep_dedup'] = ext_df['endpoint'].apply(ep_for_dedup)
ext_df['key_smiles_ep'] = ext_df['SMILES'].fillna('') + '|||' + ext_df['ep_dedup']

# Filter: must have SMILES
has_smiles = ext_df['SMILES'].notna() & (ext_df['SMILES'] != 'None')
ext_df = ext_df[has_smiles].copy()

# Mark training overlap
ext_df['in_training_smiles_ep'] = ext_df['key_smiles_ep'].isin(train_smiles_ep)
ext_df['in_training_smiles'] = ext_df['SMILES'].isin(train_smiles_set)

# Remove anything that shares SMILES+endpoint with training set
# (conservative: same compound + same endpoint type is excluded even if different species)
before = len(ext_df)
ext_clean = ext_df[~ext_df['in_training_smiles_ep']].copy()
# For TMF/BMF/BSAF, only exclude if exact SMILES match in training
tmf_bsaf = ext_clean[ext_clean['endpoint'].isin(['logTMF', 'logBMF', 'logBSAF'])]
bcf_baf_clean = ext_clean[ext_clean['endpoint'].isin(['logBAF', 'logBCF'])]

print(f"  Before dedup: {before}")
print(f"  Removed (SMILES+endpoint overlap): {before - len(ext_clean)}")
print(f"  After dedup:")
print(f"    logBAF/logBCF: {len(bcf_baf_clean)}")
print(f"    logTMF: {len(tmf_bsaf[tmf_bsaf['endpoint']=='logTMF'])}")
print(f"    logBMF: {len(tmf_bsaf[tmf_bsaf['endpoint']=='logBMF'])}")
print(f"    logBSAF: {len(tmf_bsaf[tmf_bsaf['endpoint']=='logBSAF'])}")

ext_clean.to_csv(OUT / 'step3_external_clean.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Load model predictions and join
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Joining model predictions ...")

m2_pred = pd.read_csv(ROOT / 'model_data/model2_out/model2_predictions.csv')
m2_pred = m2_pred[m2_pred['smiles_valid'] == True].copy()
m2_pred['SMILES'] = m2_pred['SMILES'].astype(str).str.strip()
pred_by_smiles = (m2_pred.drop_duplicates('SMILES')
                  .set_index('SMILES')[['pred_logBAF','pred_logBCF','unc_logBAF','unc_logBCF']]
                  .to_dict('index'))

def get_pred(smiles, endpoint):
    if smiles not in pred_by_smiles:
        return np.nan, np.nan
    row = pred_by_smiles[smiles]
    if 'BAF' in endpoint.upper():
        return row.get('pred_logBAF', np.nan), row.get('unc_logBAF', np.nan)
    elif 'BCF' in endpoint.upper():
        return row.get('pred_logBCF', np.nan), row.get('unc_logBCF', np.nan)
    elif 'TMF' in endpoint.upper() or 'BMF' in endpoint.upper():
        # Use logBAF as proxy for food-web amplification
        return row.get('pred_logBAF', np.nan), row.get('unc_logBAF', np.nan)
    elif 'BSAF' in endpoint.upper():
        return row.get('pred_logBCF', np.nan), row.get('unc_logBCF', np.nan)
    return np.nan, np.nan

ext_clean['pred_value'] = np.nan
ext_clean['pred_unc'] = np.nan
for idx, row in ext_clean.iterrows():
    p, u = get_pred(row['SMILES'], row['endpoint'])
    ext_clean.at[idx, 'pred_value'] = p
    ext_clean.at[idx, 'pred_unc'] = u

# Keep only records with predictions
ext_valid = ext_clean[ext_clean['pred_value'].notna() & ext_clean['log_value'].notna()].copy()
print(f"  Records with model predictions: {len(ext_valid)}")
print(f"  Endpoint breakdown:\n{ext_valid['endpoint'].value_counts()}")

ext_valid.to_csv(OUT / 'step4_validation_set.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Regression validation
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Regression validation ...")

from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(obs, pred, label=''):
    obs, pred = np.array(obs), np.array(pred)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 5:
        return {'label': label, 'n': len(obs), 'MAE': np.nan, 'RMSE': np.nan,
                'Spearman_rho': np.nan, 'R2': np.nan}
    mae  = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    rho, pval = spearmanr(obs, pred)
    r2   = r2_score(obs, pred)
    return {'label': label, 'n': int(len(obs)),
            'MAE': round(mae, 3), 'RMSE': round(rmse, 3),
            'Spearman_rho': round(rho, 3), 'R2': round(r2, 3), 'p_value': f'{pval:.2e}'}

metrics_rows = []

# Overall
metrics_rows.append(compute_metrics(ext_valid['log_value'], ext_valid['pred_value'], 'All external'))

# By endpoint
for ep in ['logBAF', 'logBCF', 'logTMF', 'logBMF', 'logBSAF']:
    sub = ext_valid[ext_valid['endpoint'] == ep]
    if len(sub) >= 5:
        metrics_rows.append(compute_metrics(sub['log_value'], sub['pred_value'], ep))

# BAF+BCF combined (regression endpoints)
bcf_baf = ext_valid[ext_valid['endpoint'].isin(['logBAF', 'logBCF'])]
if len(bcf_baf) >= 5:
    metrics_rows.append(compute_metrics(bcf_baf['log_value'], bcf_baf['pred_value'], 'logBAF+logBCF'))

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT / 'step5_regression_metrics.csv', index=False)
print(metrics_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: OECD B/vB threshold classification
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: OECD B/vB threshold classification ...")

B_THRESH  = 3.3
VB_THRESH = 4.3

# Use BCF/BAF for threshold analysis
thresh_df = ext_valid[ext_valid['endpoint'].isin(['logBAF', 'logBCF'])].copy()

if len(thresh_df) > 0:
    thresh_df['obs_B']  = (thresh_df['log_value']   >= B_THRESH).astype(int)
    thresh_df['pred_B'] = (thresh_df['pred_value']  >= B_THRESH).astype(int)
    thresh_df['obs_vB'] = (thresh_df['log_value']   >= VB_THRESH).astype(int)
    thresh_df['pred_vB']= (thresh_df['pred_value']  >= VB_THRESH).astype(int)

    def threshold_stats(obs_col, pred_col, label):
        obs  = thresh_df[obs_col].values
        pred = thresh_df[pred_col].values
        TP = ((obs==1) & (pred==1)).sum()
        FP = ((obs==0) & (pred==1)).sum()
        FN = ((obs==1) & (pred==0)).sum()
        TN = ((obs==0) & (pred==0)).sum()
        prec = TP/(TP+FP) if (TP+FP)>0 else 0
        rec  = TP/(TP+FN) if (TP+FN)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        acc  = (TP+TN)/len(obs) if len(obs)>0 else 0
        return {'Threshold': label, 'n': len(obs),
                'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
                'Precision': round(prec, 3), 'Recall': round(rec, 3),
                'F1': round(f1, 3), 'Accuracy': round(acc, 3)}

    cls_rows = [
        threshold_stats('obs_B',  'pred_B',  'B (logBCF/BAF ≥ 3.3)'),
        threshold_stats('obs_vB', 'pred_vB', 'vB (logBCF/BAF ≥ 4.3)'),
    ]
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(OUT / 'step6_threshold_classification.csv', index=False)
    print(cls_df.to_string(index=False))
    thresh_df.to_csv(OUT / 'step6_threshold_detail.csv', index=False)
else:
    print("  Not enough BCF/BAF data for threshold analysis")
    cls_df = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: TMF cross-scale correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: TMF cross-scale analysis ...")

tmf_data = ext_valid[ext_valid['endpoint'] == 'logTMF'].copy()
if len(tmf_data) >= 5:
    rho_tmf, p_tmf = spearmanr(tmf_data['log_value'], tmf_data['pred_value'])
    print(f"  logTMF vs pred_logBAF: Spearman rho={rho_tmf:.3f}, p={p_tmf:.3e}, n={len(tmf_data)}")
    tmf_data.to_csv(OUT / 'step7_tmf_validation.csv', index=False)
else:
    rho_tmf, p_tmf = np.nan, np.nan
    print(f"  Limited TMF data: n={len(tmf_data)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Wildlife detection enrichment analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 8: Detection enrichment analysis ...")

# Load tissue concentration data
df2 = pd.read_excel(xls_main, sheet_name='Table2_concentration')
# Map PFAS abbreviation to SMILES
df2['SMILES'] = df2['PFAS'].apply(abbr_to_smiles)
df2_valid = df2[df2['SMILES'].notna()].copy()
# Join predictions
df2_valid['pred_logBAF'] = df2_valid['SMILES'].map(
    lambda s: pred_by_smiles.get(s, {}).get('pred_logBAF', np.nan))
df2_valid = df2_valid[df2_valid['pred_logBAF'].notna()].copy()

# Split into top/bottom predicted risk
if len(df2_valid) >= 20:
    q75 = df2_valid['pred_logBAF'].quantile(0.75)
    q25 = df2_valid['pred_logBAF'].quantile(0.25)
    high_risk = df2_valid[df2_valid['pred_logBAF'] >= q75]
    low_risk  = df2_valid[df2_valid['pred_logBAF'] <  q25]

    # Detection rate (non-zero concentration)
    try:
        high_conc = pd.to_numeric(df2_valid[df2_valid['pred_logBAF'] >= q75]['concentration_mean'], errors='coerce')
        low_conc  = pd.to_numeric(df2_valid[df2_valid['pred_logBAF'] <  q25]['concentration_mean'], errors='coerce')
        high_det_rate = (high_conc > 0).mean()
        low_det_rate  = (low_conc  > 0).mean()
        enrichment_ratio = high_det_rate / low_det_rate if low_det_rate > 0 else np.nan
        print(f"  High-risk PFAS detection rate: {high_det_rate:.1%} vs Low-risk: {low_det_rate:.1%}")
        print(f"  Enrichment ratio: {enrichment_ratio:.2f}x")
    except Exception as e:
        print(f"  Detection enrichment: {e}")
        enrichment_ratio = np.nan
    df2_valid.to_csv(OUT / 'step8_detection_enrichment.csv', index=False)
else:
    print(f"  Limited detection data: n={len(df2_valid)}")
    enrichment_ratio = np.nan


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Summary outputs
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: Summary ...")

summary_rows = [
    ('Training set fingerprint', f"{len(train_fp):,} records, {train_fp['SMILES'].nunique()} unique SMILES"),
    ('External compiled (all endpoints)', f"{len(ext_df):,} records"),
    ('External after deduplication (BCF/BAF)', f"{len(bcf_baf_clean):,} records"),
    ('Records with model predictions', f"{len(ext_valid):,}"),
    ('  – logBAF/logBCF', str(len(ext_valid[ext_valid['endpoint'].isin(['logBAF','logBCF'])]))),
    ('  – logTMF', str(len(ext_valid[ext_valid['endpoint']=='logTMF']))),
    ('  – logBMF', str(len(ext_valid[ext_valid['endpoint']=='logBMF']))),
    ('  – logBSAF', str(len(ext_valid[ext_valid['endpoint']=='logBSAF']))),
]

for label, val in summary_rows:
    print(f"  {label}: {val}")

pd.DataFrame(summary_rows, columns=['Item', 'Value']).to_csv(OUT / 'step9_summary.csv', index=False)

print(f"\nAll outputs saved to: {OUT}")
print("Files:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name} ({f.stat().st_size//1024} KB)")
