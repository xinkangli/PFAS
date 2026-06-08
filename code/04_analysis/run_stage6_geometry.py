"""
Stage 6 — Geometry-aware ablation (v2).

Ablation: M0 (2D only) vs M1 (2D+transport) vs M2 (2D+3D) vs M3 (2D+transport+3D)
3D descriptors extracted via RDKit ETKDG from PFAS-strict SMILES (or pre-computed SDF).

For each model (M0-M3):
  - Chemical-out split
  - Subclass-out split (top subclass with most records)
  - 3 seeds → average

Outputs:
  result/stage6_geometry/ablation_results.csv
  result/tables/table2_ablation.csv
  result/figs/fig_ablation.png
"""
from __future__ import annotations
import sys, json, gc, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'result'))
from _style import NC_PALETTE, neat_axes  # noqa
import _style  # noqa

OUT = ROOT / 'result/stage6_geometry'
FIGS = ROOT / 'result/figs'
TABLES = ROOT / 'result/tables'
for d in [OUT, FIGS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

LOG = ROOT / 'result/logs/stage6.log'
LOG.parent.mkdir(exist_ok=True)


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')


# ============================================================
# 3D descriptor extraction
# ============================================================
def extract_3d_descriptors(smiles_list, n_conf=1, seed=42):
    """
    Returns float32 array [N, 14] of 3D descriptors.
    Missing (conformer generation failure) rows are NaN.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D
    from rdkit.Chem.rdMolDescriptors import (
        CalcPMI1, CalcPMI2, CalcPMI3,
        CalcNPR1, CalcNPR2,
        CalcRadiusOfGyration,
        CalcAsphericity,
        CalcEccentricity,
        CalcInertialShapeFactor,
        CalcSpherocityIndex,
    )
    N = len(smiles_list)
    cols = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration',
            'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'SpherocityIndex',
            'SASA', 'F_surface_ratio', 'head_tail_dist', 'linearity']
    X3d = np.full((N, len(cols)), np.nan, dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol, params)
            if res == -1:
                continue
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()

            pmi1 = CalcPMI1(mol)
            pmi2 = CalcPMI2(mol)
            pmi3 = CalcPMI3(mol)
            npr1 = CalcNPR1(mol)
            npr2 = CalcNPR2(mol)
            rog = CalcRadiusOfGyration(mol)
            asp = CalcAsphericity(mol)
            ecc = CalcEccentricity(mol)
            isf = CalcInertialShapeFactor(mol)
            sph = CalcSpherocityIndex(mol)

            # SASA estimate (sum atomic radii × coords)
            try:
                from rdkit.Chem import rdFreeSASA
                radii = rdFreeSASA.classifyAtoms(mol)
                sasa = rdFreeSASA.CalcSASA(mol, radii)
            except Exception:
                sasa = np.nan

            # F-surface ratio: CF/CF2/CF3 atom SASA vs total
            f_ratio = np.nan
            try:
                from rdkit.Chem import rdFreeSASA
                radii = rdFreeSASA.classifyAtoms(mol)
                atom_sasa = rdFreeSASA.CalcSASA(mol, radii, perAtomSASA=True) if hasattr(rdFreeSASA, 'CalcSASA') else []
                # rough: count F atoms / total heavy atoms
                f_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
                f_ratio = float(f_atoms) / max(mol.GetNumHeavyAtoms(), 1)
            except Exception:
                f_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
                f_ratio = float(f_atoms) / max(mol.GetNumHeavyAtoms(), 1)

            # Head-tail distance: heavy oxygen/N/S atoms vs terminal F atoms
            pos = conf.GetPositions()
            head_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (8, 7, 16) and a.GetAtomicNum() != 1]
            tail_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 9]
            ht_dist = np.nan
            if head_ids and tail_ids:
                head_c = pos[head_ids].mean(0)
                tail_c = pos[tail_ids].mean(0)
                ht_dist = float(np.linalg.norm(head_c - tail_c))

            linearity = float(pmi3 / pmi1) if pmi1 > 1e-6 else np.nan

            X3d[i] = [pmi1, pmi2, pmi3, npr1, npr2, rog, asp, ecc, isf, sph,
                      sasa if not np.isnan(sasa) else np.nan, f_ratio, ht_dist, linearity]
        except Exception:
            continue
    return X3d, cols


# ============================================================
# Load base features (re-use stage5 logic)
# ============================================================
log('loading model2 data and features')
import openpyxl
import featurize as _ft

meta2 = json.load(open(ROOT / 'model_data/model2_out/model2_meta.json'))
feat_names = meta2['feature_names']
bio_vocab = meta2['bio_vocab']
bio_num_cols = ['exposure_days', 'exposure_concentration_numeric', 'temperature_c']
transport_cols = meta2['transport_cols']
y_mean = np.asarray(meta2['y_mean'])
y_std = np.asarray(meta2['y_std'])
in_mean = np.asarray(meta2['in_mean'])
in_std = np.asarray(meta2['in_std'])
fill_med = np.asarray(meta2['fill_median'])
endpoints = meta2['endpoints']
hp = meta2['hyperparams']
hidden = tuple(hp['hidden'])
dropout = hp['dropout']
ep_weights = [1.0] * len(endpoints)  # uniform weights

wb2 = openpyxl.load_workbook(ROOT / 'model_data/Model2_Bioaccumulation.xlsx', data_only=True)
sheet = wb2['X_Y_joined']
header = [c.value for c in next(sheet.iter_rows(min_row=1, max_row=1))]
rows = list(sheet.iter_rows(min_row=2, values_only=True))
df = pd.DataFrame(rows, columns=header)
df = df[df['log_value'].notna()].reset_index(drop=True)
df['endpoint_type'] = df['endpoint_type'].astype(str).str.strip()
_ep_remap = {'BAF':'logBAF','BCF':'logBCF','BCFD':'logBCFD','TMF':'logTMF',
             'logBAF':'logBAF','logBCF':'logBCF','logBCFD':'logBCFD','logTMF':'logTMF'}
df['endpoint_type'] = df['endpoint_type'].map(lambda x: _ep_remap.get(x, x))

# ClassyFire
master = pd.read_csv(ROOT / '03_transport/PFAS_master_structure.csv', low_memory=False)
cf = pd.read_csv(ROOT / '01_structure/classyfire/classyfire_PFAS_pool_subset.csv')
if 'chem_cf_subclass_name' in cf.columns:
    cf = cf.rename(columns={'chem_cf_subclass_name': 'cf_subclass'})
mast_ik = master[['SMILES', 'InChIKey']].dropna().drop_duplicates('SMILES')
cf_slim = cf[['InChIKey', 'cf_subclass']].dropna().drop_duplicates('InChIKey')
mast_cf = mast_ik.merge(cf_slim, on='InChIKey', how='left')[['SMILES', 'cf_subclass']]
_cf_map = mast_cf.drop_duplicates('SMILES').set_index('SMILES')['cf_subclass']
df['cf_subclass'] = df['SMILES'].map(_cf_map)

# Structure features
wb1 = openpyxl.load_workbook(ROOT / 'model_data/Model1_Transport.xlsx', data_only=True)
fstats_sheet = wb1['F_stats_dict']
fstats_header = [c.value for c in next(fstats_sheet.iter_rows(min_row=1, max_row=1))]
fstats_rows = list(fstats_sheet.iter_rows(min_row=2, values_only=True))
fstats = pd.DataFrame(fstats_rows, columns=fstats_header)

ftz = _ft.Featurizer()
X_struct, valid = ftz.transform(df['SMILES'].tolist(), fstats_df=fstats, fstats_key='SMILES')
df = df.loc[valid].reset_index(drop=True)
X_struct = X_struct[valid]

# Transport features — safe join (avoid row explosion)
m1_pred = pd.read_csv(ROOT / 'model_data/model1_out/model1_predictions.csv')
_tcols = [c for c in transport_cols if c in m1_pred.columns]
m1_slim = m1_pred[['SMILES'] + _tcols].drop_duplicates('SMILES').set_index('SMILES')
for col in _tcols:
    df[col] = df['SMILES'].map(m1_slim[col])
X_transport = df[transport_cols].values.astype(np.float32)
t_mean = np.nanmean(X_transport, axis=0)
for j in range(X_transport.shape[1]):
    mask = ~np.isfinite(X_transport[:, j])
    X_transport[mask, j] = t_mean[j]

# Bio context
bio_parts = []
for col, vocab in bio_vocab.items():
    if col not in df.columns:
        arr = np.zeros((len(df), len(vocab)), dtype=np.float32)
    else:
        vals = df[col].astype(str).str.strip()
        idx_map = {v: i for i, v in enumerate(vocab)}
        other_idx = idx_map.get('OTHER', len(vocab) - 1)
        arr = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, v in enumerate(vals):
            arr[i, idx_map.get(v, other_idx)] = 1.0
    bio_parts.append(arr)
bio_num_med = meta2.get('bio_num_median', {})
for col in bio_num_cols:
    if col in df.columns:
        v = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
        med = bio_num_med.get(col, np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
        v[~np.isfinite(v)] = float(med)
        bio_parts.append(v.reshape(-1, 1))
X_bio = np.concatenate(bio_parts, axis=1)

log('extracting 3D descriptors for PFAS-strict SMILES (~PFAS subset)')
# Only compute 3D for PFAS strict (has SDF) — ~148 strict, rest NaN
sdf_idx_df = pd.read_csv(ROOT / '01_structure/ecotox_sdf/sdf_PFAS_pool_subset.csv', low_memory=False)
smi_col = 'chem_pcp_can_smiles' if 'chem_pcp_can_smiles' in sdf_idx_df.columns else 'SMILES'
has_sdf = set(sdf_idx_df[smi_col].dropna().astype(str).str.strip().tolist())
log(f'  has_sdf count = {len(has_sdf)}')

# Compute 3D for all unique SMILES in the dataset (not just SDF-matched)
# This is tractable since dataset has ~2k unique SMILES
unique_smiles_for_3d = df['SMILES'].drop_duplicates().tolist()
log(f'  computing 3D for all {len(unique_smiles_for_3d)} unique SMILES in dataset...')
X3d_unique, d3_cols = extract_3d_descriptors(unique_smiles_for_3d)
smi_to_3d = {smi: X3d_unique[i] for i, smi in enumerate(unique_smiles_for_3d)}
X3d_raw = np.vstack([smi_to_3d.get(s, np.full(14, np.nan)) for s in df['SMILES'].tolist()]).astype(np.float32)
log(f'  3D valid: {np.isfinite(X3d_raw[:, 0]).sum()} / {len(X3d_raw)}')
# Standardize 3D
d3_mean = np.nanmean(X3d_raw, axis=0)
d3_std = np.nanstd(X3d_raw, axis=0)
for j in range(X3d_raw.shape[1]):
    m = ~np.isfinite(X3d_raw[:, j])
    X3d_raw[m, j] = d3_mean[j] if np.isfinite(d3_mean[j]) else 0.0
X3d = (X3d_raw - d3_mean) / np.where(d3_std > 1e-8, d3_std, 1.0)
X3d = X3d.astype(np.float32)
np.save(OUT / 'X3d.npy', X3d)

# Y encoding
Y_raw = df['log_value'].values.astype(np.float32)
ep_col = df['endpoint_type'].values
ep_to_idx = {ep: i for i, ep in enumerate(endpoints)}
Y = np.full((len(df), len(endpoints)), np.nan, dtype=np.float32)
mask_Y = np.zeros((len(df), len(endpoints)), dtype=np.float32)
for i, ep in enumerate(ep_col):
    if ep in ep_to_idx:
        j = ep_to_idx[ep]
        Y[i, j] = (Y_raw[i] - y_mean[j]) / (y_std[j] if y_std[j] > 1e-8 else 1.0)
        mask_Y[i, j] = 1.0

Y = np.where(np.isfinite(Y), Y, 0.0)
rel_col = df['reliability'].values if 'reliability' in df.columns else np.ones(len(df))
_rel_map = {1.5: 1.5, 1.0: 1.0, 0.7: 0.7, 0.5: 0.5, 0.3: 0.3}
def _parse_rel(r):
    try:
        return _rel_map.get(float(r), 1.0)
    except (TypeError, ValueError):
        return 1.0
w_sample = np.array([_parse_rel(r) if pd.notna(r) else 1.0 for r in rel_col], dtype=np.float32)

smiles_col = df['SMILES'].values
cf_col = df['cf_subclass'].values

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Model variants X matrices
# ============================================================
def _dyn_std(X_raw):
    X = X_raw.copy().astype(np.float32)
    m = np.nanmean(X, 0); s = np.nanstd(X, 0)
    m = np.where(np.isfinite(m), m, 0.0); s = np.where(np.isfinite(s), s, 1.0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j]); X[bad, j] = m[j]
    return np.clip((X - m) / np.where(s > 1e-8, s, 1.0), -10, 10)

X_2d = _dyn_std(X_struct)
X_transport_std = _dyn_std(X_transport)
X_bio_std = _dyn_std(X_bio)

VARIANTS = {
    'M0 (2D only)':            np.concatenate([X_2d, X_bio_std], axis=1),
    'M1 (2D+Transport)':       np.concatenate([X_2d, X_transport_std, X_bio_std], axis=1),
    'M2 (2D+3D)':              np.concatenate([X_2d, X3d, X_bio_std], axis=1),
    'M3 (2D+Transport+3D)':    np.concatenate([X_2d, X_transport_std, X3d, X_bio_std], axis=1),
}
log('Model variants dims: ' + str({k: v.shape[1] for k, v in VARIANTS.items()}))


# ============================================================
# Training helper
# ============================================================
class BioNet(nn.Module):
    def __init__(self, in_dim, endpoints, hidden=(512, 256, 128), dropout=0.25):
        super().__init__()
        self.endpoints = list(endpoints)
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({
            ep: nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
            for ep in self.endpoints
        })

    def forward(self, x):
        z = self.backbone(x)
        return torch.cat([self.heads[ep](z) for ep in self.endpoints], dim=1)


def masked_loss(pred, target, mask, ep_wts):
    se = (pred - target) ** 2 * mask
    denom = mask.sum(0).clamp(min=1.)
    per_ep = se.sum(0) / denom
    w = torch.tensor(ep_wts, dtype=torch.float32, device=pred.device)
    w = w / w.sum().clamp(min=1e-8)
    return (per_ep * w).sum()


def run_one(X_full, tr, val, seed, epochs=300, patience=50):
    torch.manual_seed(seed)
    net = BioNet(X_full.shape[1], endpoints, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)

    Xtr = torch.from_numpy(X_full[tr]).float().to(device)
    Ytr = torch.from_numpy(Y[tr]).float().to(device)
    Mtr = torch.from_numpy(mask_Y[tr]).float().to(device)
    Wtr = torch.from_numpy(w_sample[tr]).float().to(device)
    Xval = torch.from_numpy(X_full[val]).float().to(device)

    best_loss = np.inf
    no_imp = 0
    best_state = None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 128):
            idx = perm[s:s + 128]
            xb, yb, mb, wb = Xtr[idx], Ytr[idx], Mtr[idx], Wtr[idx]
            pred = net(xb)
            loss = masked_loss(pred * wb.unsqueeze(1), yb * wb.unsqueeze(1), mb, ep_weights)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vp = net(Xval).cpu().numpy()
        vl = np.mean([(((vp[:, k] - Y[val, k]) ** 2) * mask_Y[val, k]).sum() /
                      max(mask_Y[val, k].sum(), 1.) for k in range(len(endpoints))])
        if vl < best_loss - 1e-6:
            best_loss = vl; no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xval).cpu().numpy()
    return pred


def metrics_ep(pred, true_Y, mask_Y_val, ep_k):
    m = mask_Y_val[:, ep_k].astype(bool)
    if m.sum() < 5:
        return None
    p = pred[m, ep_k] * y_std[ep_k] + y_mean[ep_k]
    t = true_Y[m, ep_k] * y_std[ep_k] + y_mean[ep_k]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rho, _ = spearmanr(p, t) if len(p) > 3 else (np.nan, np.nan)
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
    return {'n': int(m.sum()), 'MAE': mae, 'RMSE': rmse, 'Spearman': rho, 'R2': float(r2)}


# ============================================================
# Define splits (chemical-out + subclass-out)
# ============================================================
N = len(X_2d)
rng = np.random.default_rng(42)
all_idx = np.arange(N)
unique_smi = np.unique(smiles_col)
val_smi_set = set(rng.choice(unique_smi, size=max(1, int(0.15 * len(unique_smi))), replace=False))
chem_val = np.where(np.isin(smiles_col, list(val_smi_set)))[0]
chem_tr = np.setdiff1d(all_idx, chem_val)

# top subclass
cf_known = cf_col[~pd.isnull(cf_col) & (cf_col != '') & (cf_col != 'nan')]
top_sub = pd.Series(cf_known).value_counts().index[0] if len(cf_known) > 0 else None
if top_sub is not None and (cf_col == top_sub).sum() >= 5:
    sub_val = np.where(cf_col == top_sub)[0]
    sub_tr = np.where(cf_col != top_sub)[0]
    do_subclass = True
else:
    do_subclass = False

SEEDS = [42, 123, 777]
abl_rows = []

for model_name, X_full in VARIANTS.items():
    log(f'\n=== {model_name} (dim={X_full.shape[1]}) ===')
    # Chemical-out
    preds_seeds = [run_one(X_full, chem_tr, chem_val, s) for s in SEEDS]
    pred_avg = np.mean(preds_seeds, axis=0)
    for k, ep in enumerate(endpoints):
        m = metrics_ep(pred_avg, Y[chem_val], mask_Y[chem_val], k)
        if m:
            abl_rows.append({'model': model_name, 'split': 'chemical-out',
                             'endpoint': ep, **m})
            log(f'  chem-out {ep}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} Spearman={m["Spearman"]:.4f}')
    # Subclass-out
    if do_subclass:
        preds_seeds_s = [run_one(X_full, sub_tr, sub_val, s) for s in SEEDS]
        pred_avg_s = np.mean(preds_seeds_s, axis=0)
        for k, ep in enumerate(endpoints):
            m = metrics_ep(pred_avg_s, Y[sub_val], mask_Y[sub_val], k)
            if m:
                abl_rows.append({'model': model_name, 'split': f'subclass-out:{top_sub}',
                                 'endpoint': ep, **m})
                log(f'  sub-out {ep}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f}')

abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(OUT / 'ablation_results.csv', index=False)
log('Ablation results saved')

# ============================================================
# Table 2 (BCF chemical-out comparison)
# ============================================================
t2_rows = []
for mname in VARIANTS:
    row = {'Model': mname}
    sub = abl_df[(abl_df['model'] == mname) & (abl_df['split'] == 'chemical-out')]
    for ep in ['logBCF', 'logBAF']:
        r = sub[sub['endpoint'] == ep]
        if not r.empty:
            row[f'{ep} MAE'] = round(r.iloc[0]['MAE'], 4)
            row[f'{ep} Spearman'] = round(r.iloc[0]['Spearman'], 4)
    t2_rows.append(row)
table2 = pd.DataFrame(t2_rows)
table2.to_csv(TABLES / 'table2_ablation.csv', index=False)
log('Table 2 saved')

# ============================================================
# Figure: grouped bar chart M0-M3 × MAE
# ============================================================
fig_ep = 'logBCF'
sub_abl = abl_df[(abl_df['endpoint'] == fig_ep) & (abl_df['split'] == 'chemical-out')]
if len(sub_abl) >= 2:
    model_order = ['M0 (2D only)', 'M1 (2D+Transport)', 'M2 (2D+3D)', 'M3 (2D+Transport+3D)']
    model_present = [m for m in model_order if m in sub_abl['model'].values]
    maes = [sub_abl[sub_abl['model'] == m]['MAE'].values[0] for m in model_present]
    colors = [NC_PALETTE[i % len(NC_PALETTE)] for i in range(len(model_present))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(model_present))
    bars = ax.bar(x, maes, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_present, rotation=20, ha='right', fontsize=11)
    ax.set_ylabel(f'MAE (log {fig_ep.replace("log", "")})', fontsize=13)
    ax.set_title('Geometry-aware ablation (chemical-out split)', pad=14)
    # annotate values above bars
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    neat_axes(ax)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGS / 'fig_ablation.png', dpi=300)
    plt.savefig(OUT / 'fig_ablation.png', dpi=300)
    plt.close()
    log('Figure saved: fig_ablation.png')

log('Stage 6 done.')
