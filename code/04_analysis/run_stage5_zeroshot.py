"""
Stage 5 — Zero-shot benchmark (v2).

6 split types × 4 metrics (MAE, RMSE, Spearman, R²) + 95% CI bootstrap.

Splits:
  ① Random split
  ② Chemical-out split (SMILES group)
  ③ Subclass-out split (ClassyFire subclass)
  ④ Organism-out split (organism_group)
  ⑤ Phylogeny-out split (FCA_pdm clade distance ≥ 0.4)
  ⑥ Endpoint zero-shot (BCF+BAF→TMF, BCF→BAF, BAF→BCF, BCF+BAF→BCFD)

Output:
  result/stage5_zeroshot/zero_shot_results.csv
  result/tables/table1_zeroshot.csv
  result/figs/fig_zeroshot.png
"""
from __future__ import annotations
import sys, json, gc, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'result'))
from _style import NC_PALETTE, neat_axes  # noqa
import _style  # noqa

OUT = ROOT / 'result/stage5_zeroshot'
FIGS = ROOT / 'result/figs'
TABLES = ROOT / 'result/tables'
for d in [OUT, FIGS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

LOG = ROOT / 'result/logs/stage5.log'
LOG.parent.mkdir(exist_ok=True)


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')


# ============================================================
# Model 2 architecture (same as train_model2.py)
# ============================================================
class BioaccumulationNet(nn.Module):
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


# ============================================================
# Load data (reuse model2 data loading logic)
# ============================================================
log('loading model2 data')
import openpyxl

meta2 = json.load(open(ROOT / 'model_data/model2_out/model2_meta.json'))
feat_names = meta2['feature_names']
bio_vocab = meta2['bio_vocab']
bio_num_cols = ['exposure_days', 'exposure_concentration_numeric', 'temperature_c']
transport_cols = meta2['transport_cols']
m1_endpoints = meta2['m1_endpoints']
y_mean = np.asarray(meta2['y_mean'])
y_std = np.asarray(meta2['y_std'])
in_mean = np.asarray(meta2['in_mean'])
in_std = np.asarray(meta2['in_std'])
fill_med = np.asarray(meta2['fill_median'])
endpoints = meta2['endpoints']
hp = meta2['hyperparams']
hidden = tuple(hp['hidden'])
dropout = hp['dropout']

log('loading X_Y_joined from Model2_Bioaccumulation.xlsx')
wb2 = openpyxl.load_workbook(ROOT / 'model_data/Model2_Bioaccumulation.xlsx', data_only=True)
sheet = wb2['X_Y_joined']
header = [c.value for c in next(sheet.iter_rows(min_row=1, max_row=1))]
rows = list(sheet.iter_rows(min_row=2, values_only=True))
df = pd.DataFrame(rows, columns=header)
df = df[df['log_value'].notna()].reset_index(drop=True)
df['endpoint_type'] = df['endpoint_type'].astype(str).str.strip()
# normalise to match model2 endpoint names: BAF→logBAF, BCF→logBCF, etc.
_ep_remap = {'BAF':'logBAF','BCF':'logBCF','BCFD':'logBCFD','TMF':'logTMF',
             'logBAF':'logBAF','logBCF':'logBCF','logBCFD':'logBCFD','logTMF':'logTMF'}
df['endpoint_type'] = df['endpoint_type'].map(lambda x: _ep_remap.get(x, x))
log(f'  X_Y_joined shape={df.shape}, endpoints={df["endpoint_type"].value_counts().to_dict()}')

# Load ClassyFire subclass via master structure InChIKey
master = pd.read_csv(ROOT / '03_transport/PFAS_master_structure.csv', low_memory=False)
cf = pd.read_csv(ROOT / '01_structure/classyfire/classyfire_PFAS_pool_subset.csv')
if 'chem_cf_subclass_name' in cf.columns:
    cf = cf.rename(columns={'chem_cf_subclass_name': 'cf_subclass'})
mast_ik = master[['SMILES', 'InChIKey']].dropna().drop_duplicates('SMILES')
cf_slim = cf[['InChIKey', 'cf_subclass']].dropna().drop_duplicates('InChIKey')
mast_cf = mast_ik.merge(cf_slim, on='InChIKey', how='left')[['SMILES', 'cf_subclass']]
df = df.merge(mast_cf, on='SMILES', how='left')
log(f'  ClassyFire coverage: {df["cf_subclass"].notna().sum()}/{len(df)}')

# ============================================================
# Featurize (same as model2 training)
# ============================================================
log('featurizing (1057-d struct + transport + bio)')
import featurize as _ft

fstats_sheet = wb2['F_stats_dict'] if 'F_stats_dict' in wb2.sheetnames else None
if fstats_sheet is None:
    wb1 = openpyxl.load_workbook(ROOT / 'model_data/Model1_Transport.xlsx', data_only=True)
    fstats_sheet = wb1['F_stats_dict']
fstats_header = [c.value for c in next(fstats_sheet.iter_rows(min_row=1, max_row=1))]
fstats_rows = list(fstats_sheet.iter_rows(min_row=2, values_only=True))
fstats = pd.DataFrame(fstats_rows, columns=fstats_header)

ftz = _ft.Featurizer()
X_struct, valid = ftz.transform(df['SMILES'].tolist(), fstats_df=fstats, fstats_key='SMILES')
df = df.loc[valid].reset_index(drop=True)
X_struct = X_struct[valid]

# Transport features — use index-safe join to avoid row explosion from duplicate SMILES
m1_pred = pd.read_csv(ROOT / 'model_data/model1_out/model1_predictions.csv')
_tcols = [c for c in transport_cols if c in m1_pred.columns]
m1_slim = m1_pred[['SMILES'] + _tcols].drop_duplicates('SMILES').set_index('SMILES')
_n_before = len(df)
for col in _tcols:
    df[col] = df['SMILES'].map(m1_slim[col])
assert len(df) == _n_before, 'transport join changed row count'
X_transport = df[transport_cols].values.astype(np.float32)
t_mean = np.nanmean(X_transport, axis=0)
for j in range(X_transport.shape[1]):
    mask = ~np.isfinite(X_transport[:, j])
    X_transport[mask, j] = t_mean[j]

# Bio context one-hot
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
X = np.concatenate([X_struct, X_transport, X_bio], axis=1).astype(np.float32)

# Standardize: compute per-column stats on the fly to handle any dimension mismatch
_fill = np.nanmedian(X, axis=0)
_fill = np.where(np.isfinite(_fill), _fill, 0.0)
_mean = np.nanmean(X, axis=0)
_mean = np.where(np.isfinite(_mean), _mean, 0.0)
_std  = np.nanstd(X, axis=0)
nan_mask = ~np.isfinite(X)
for j in range(X.shape[1]):
    X[nan_mask[:, j], j] = _fill[j]
X = (X - _mean) / np.where(_std > 1e-8, _std, 1.0)
X = np.clip(X, -10, 10)
in_mean = _mean; in_std = _std; fill_med = _fill  # update for any downstream use

Y_raw = df['log_value'].values.astype(np.float32)
ep_col = df['endpoint_type'].values
smiles_col = df['SMILES'].values
org_col = df['organism_group'].astype(str).str.strip().values if 'organism_group' in df.columns else np.full(len(df), 'unknown')
cf_col = df['cf_subclass'].values

# Encode Y per-endpoint
ep_to_idx = {ep: i for i, ep in enumerate(endpoints)}
Y = np.full((len(df), len(endpoints)), np.nan, dtype=np.float32)
mask_Y = np.zeros((len(df), len(endpoints)), dtype=np.float32)
for i, ep in enumerate(ep_col):
    if ep in ep_to_idx:
        j = ep_to_idx[ep]
        Y[i, j] = (Y_raw[i] - y_mean[j]) / (y_std[j] if y_std[j] > 1e-8 else 1.0)
        mask_Y[i, j] = 1.0

# Replace NaN in Y with 0 (mask=0 prevents these from contributing to loss)
Y = np.where(np.isfinite(Y), Y, 0.0)

# Sample weights
rel_col = df['reliability'].values if 'reliability' in df.columns else np.ones(len(df))
rel_map = {1.5: 1.5, 1.0: 1.0, 0.7: 0.7, 0.5: 0.5, 0.3: 0.3}
def _parse_rel(r):
    try:
        return rel_map.get(float(r), 1.0)
    except (TypeError, ValueError):
        return 1.0
w_sample = np.array([_parse_rel(r) if pd.notna(r) else 1.0 for r in rel_col], dtype=np.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_dim_full = X.shape[1]
log(f'  X shape={X.shape}, in_dim={in_dim_full}')

# ============================================================
# Training function (single split)
# ============================================================

def masked_loss(pred, target, mask, ep_weights):
    se = (pred - target) ** 2 * mask
    denom = mask.sum(0).clamp(min=1.)
    per_ep = se.sum(0) / denom
    w = torch.tensor(ep_weights, dtype=torch.float32, device=pred.device)
    w = w / w.sum().clamp(min=1e-8)
    return (per_ep * w).sum()


def train_and_eval(train_idx, val_idx, ep_weights, seed=42,
                   epochs=400, patience=60, batch=128, lr=5e-4, wd=1e-4,
                   mask_taxonomy=False, name=''):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = BioaccumulationNet(in_dim_full, endpoints, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    X_tr = torch.from_numpy(X[train_idx]).float().to(device)
    Y_tr = torch.from_numpy(Y[train_idx]).float().to(device)
    M_tr = torch.from_numpy(mask_Y[train_idx]).float().to(device)
    W_tr = torch.from_numpy(w_sample[train_idx]).float().to(device)

    X_val = torch.from_numpy(X[val_idx]).float().to(device)
    Y_val_np = Y[val_idx]
    M_val_np = mask_Y[val_idx]

    best_val = np.inf
    no_imp = 0
    best_state = None

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(X_tr))
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, len(X_tr), batch):
            idx = perm[s:s + batch]
            xb, yb, mb, wb = X_tr[idx], Y_tr[idx], M_tr[idx], W_tr[idx]
            pred = net(xb)
            loss = masked_loss(pred, yb, mb, ep_weights)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1

        net.eval()
        with torch.no_grad():
            val_pred = net(X_val).cpu().numpy()
        val_loss = np.mean([(((val_pred[:, k] - Y_val_np[:, k]) ** 2) * M_val_np[:, k]).sum() /
                            max(M_val_np[:, k].sum(), 1.) for k in range(len(endpoints))])
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_all = net(X_val).cpu().numpy()
    return pred_all, Y_val_np, M_val_np


def compute_metrics(pred_ep, true_ep, mask_ep, y_mean_ep, y_std_ep, n_boot=500):
    results = {}
    for k, ep in enumerate(endpoints):
        m = mask_ep[:, k].astype(bool)
        if m.sum() < 5:
            continue
        p_raw = pred_ep[m, k] * y_std_ep[k] + y_mean_ep[k]
        t_raw = true_ep[m, k] * y_std_ep[k] + y_mean_ep[k]
        # drop any remaining NaN/inf pairs
        finite = np.isfinite(p_raw) & np.isfinite(t_raw)
        if finite.sum() < 5:
            continue
        p = p_raw[finite]; t = t_raw[finite]
        log(f'    {ep}: n_valid={finite.sum()}, p_range=[{p.min():.2f},{p.max():.2f}], t_range=[{t.min():.2f},{t.max():.2f}]')
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        rho, _ = spearmanr(p, t) if len(p) > 3 else (np.nan, np.nan)
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan

        # Bootstrap CI
        rng = np.random.default_rng(0)
        boot_mae, boot_rmse, boot_rho, boot_r2 = [], [], [], []
        for _ in range(n_boot):
            bi = rng.choice(len(p), size=len(p), replace=True)
            bp, bt = p[bi], t[bi]
            boot_mae.append(np.mean(np.abs(bp - bt)))
            boot_rmse.append(np.sqrt(np.mean((bp - bt) ** 2)))
            bss_r = np.sum((bp - bt) ** 2)
            bss_t = np.sum((bt - bt.mean()) ** 2)
            boot_r2.append(1 - bss_r / bss_t if bss_t > 1e-10 else np.nan)
            if len(bp) > 3:
                br, _ = spearmanr(bp, bt)
                boot_rho.append(br)
        ci95 = lambda arr: (np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5))
        results[ep] = {
            'n': int(m.sum()),
            'MAE': mae, 'MAE_ci': ci95(boot_mae),
            'RMSE': rmse, 'RMSE_ci': ci95(boot_rmse),
            'Spearman': rho, 'Spearman_ci': ci95(boot_rho) if boot_rho else (np.nan, np.nan),
            'R2': r2, 'R2_ci': ci95(boot_r2),
        }
    return results


# ============================================================
# Define 6 splits
# ============================================================
log('building splits')

splits = {}

N = len(X)
rng = np.random.default_rng(42)
all_idx = np.arange(N)

# ① Random split (15% val)
rand_val = rng.choice(N, size=int(0.15 * N), replace=False)
rand_tr = np.setdiff1d(all_idx, rand_val)
splits['Random split'] = [(rand_tr, rand_val, 'random')]

# ② Chemical-out split (group by SMILES)
unique_smi = np.unique(smiles_col)
rng2 = np.random.default_rng(0)
val_smi = set(rng2.choice(unique_smi, size=max(1, int(0.15 * len(unique_smi))), replace=False))
chem_val = np.where(np.isin(smiles_col, list(val_smi)))[0]
chem_tr = np.setdiff1d(all_idx, chem_val)
splits['Chemical-out split'] = [(chem_tr, chem_val, 'chemical-out')]

# ③ Subclass-out split — leave-one-subclass-out for top subclasses
cf_known = cf_col[~pd.isnull(cf_col) & (cf_col != '') & (cf_col != 'nan')]
sub_counts = pd.Series(cf_known).value_counts()
top_subs = sub_counts[sub_counts >= 10].head(5).index.tolist()
subclass_splits = []
for sub in top_subs:
    sub_val_idx = np.where(cf_col == sub)[0]
    sub_tr_idx = np.where(cf_col != sub)[0]  # keep all others including unlabelled
    if len(sub_val_idx) >= 5 and len(sub_tr_idx) >= 20:
        subclass_splits.append((sub_tr_idx, sub_val_idx, f'subclass-out:{sub}'))
splits['Subclass-out split'] = subclass_splits

# ④ Organism-out split — leave-one-group-out
org_groups = pd.Series(org_col)
org_counts = org_groups.value_counts()
major_orgs = org_counts[org_counts >= 30].index.tolist()
org_splits = []
for grp in major_orgs:
    og_val = np.where(org_col == grp)[0]
    og_tr = np.where(org_col != grp)[0]
    if len(og_val) >= 10 and len(og_tr) >= 50:
        org_splits.append((og_tr, og_val, f'org-out:{grp}'))
splits['Organism-out split'] = org_splits

# ⑤ Phylogeny-out split — use PDM distance matrix
phylo_splits = []
try:
    pdm = pd.read_csv(ROOT / '04_bioaccumulation/taxonomy/FCA_pdm_species.csv', index_col=0)
    sp_in_pfas = pd.read_csv(ROOT / '04_bioaccumulation/taxonomy/species_in_pfas_bcf.csv')
    repl = pd.read_csv(ROOT / '04_bioaccumulation/taxonomy/FCA_replaced_species.csv')
    repl_dict = dict(zip(repl.iloc[:, 0].str.lower().str.replace('_', ' '),
                         repl.iloc[:, 1].str.lower().str.replace('_', ' '))) if len(repl.columns) >= 2 else {}
    pdm.index = pdm.index.str.lower().str.replace('_', ' ')
    pdm.columns = pdm.columns.str.lower().str.replace('_', ' ')

    def norm_sp(s):
        s = str(s).strip().lower().replace('_', ' ')
        return repl_dict.get(s, s)

    sp_norm = np.array([norm_sp(s) for s in df['species_sci'].values]) if 'species_sci' in df.columns else None
    if sp_norm is not None:
        pdm_species = set(pdm.index.tolist())
        in_pdm = np.array([s in pdm_species for s in sp_norm])
        if in_pdm.sum() >= 20:
            in_idx = np.where(in_pdm)[0]
            sp_in_idx = sp_norm[in_idx]
            dist_thresh = 0.4
            unique_sp_in = list(set(sp_in_idx))
            # pick test species as those far from all others
            for test_sp in unique_sp_in[:8]:
                if test_sp not in pdm.index:
                    continue
                dists = pdm.loc[test_sp, [s for s in unique_sp_in if s != test_sp and s in pdm.columns]]
                if dists.min() < dist_thresh:
                    continue
                phy_val = in_idx[sp_in_idx == test_sp]
                phy_tr = np.setdiff1d(all_idx, phy_val)
                if len(phy_val) >= 5:
                    phylo_splits.append((phy_tr, phy_val, f'phylo-out:{test_sp}'))
                if len(phylo_splits) >= 3:
                    break
        log(f'  phylogeny splits found: {len(phylo_splits)}')
except Exception as e:
    log(f'  phylogeny split failed: {e}; using organism-group fallback')

if not phylo_splits:
    # fallback: use fish vs mammal/bird
    fish_idx = np.where(np.isin(org_col, ['fish', 'invertebrate']))[0]
    terr_idx = np.where(np.isin(org_col, ['mammal', 'bird']))[0]
    if len(terr_idx) >= 5:
        phylo_splits.append((fish_idx, terr_idx, 'phylo-fallback:terrestrial'))

splits['Phylogeny-out split'] = phylo_splits

# ⑥ Endpoint zero-shot
ep_splits = []
# BCF+BAF → TMF
tmf_rows = df[df['endpoint_type'] == 'logTMF'].index.tolist() if 'logTMF' in df['endpoint_type'].values else []
bcf_baf_rows = df[df['endpoint_type'].isin(['logBCF', 'logBAF'])].index.tolist()
if len(tmf_rows) >= 5:
    ep_splits.append((np.array(bcf_baf_rows), np.array(tmf_rows), 'endpoint:TMF-zeroshot'))
# BCF → BAF
baf_rows = df[df['endpoint_type'] == 'logBAF'].index.tolist()
bcf_rows = df[df['endpoint_type'] == 'logBCF'].index.tolist()
if len(baf_rows) >= 20 and len(bcf_rows) >= 20:
    ep_splits.append((np.array(bcf_rows), np.array(baf_rows), 'endpoint:BCF→BAF'))
    ep_splits.append((np.array(baf_rows), np.array(bcf_rows), 'endpoint:BAF→BCF'))
# BCF+BAF → BCFD
bcfd_rows = df[df['endpoint_type'] == 'logBCFD'].index.tolist()
if len(bcfd_rows) >= 10:
    ep_splits.append((np.array(bcf_baf_rows), np.array(bcfd_rows), 'endpoint:BCFD-zeroshot'))

splits['Endpoint zero-shot'] = ep_splits

log('splits defined:')
for sname, sp_list in splits.items():
    log(f'  {sname}: {len(sp_list)} fold(s)')

# ============================================================
# Run all splits
# ============================================================
ep_weights = [1.0] * len(endpoints)  # uniform weights for zero-shot benchmark

all_rows = []

for split_name, sp_list in splits.items():
    log(f'\n=== {split_name} ===')
    agg_preds, agg_true, agg_mask = [], [], []

    for fold_idx, (tr, val, tag) in enumerate(sp_list):
        if len(tr) < 30 or len(val) < 5:
            log(f'  skip {tag}: tr={len(tr)} val={len(val)}')
            continue
        log(f'  fold {fold_idx}: {tag} tr={len(tr)} val={len(val)}')
        pred, true, mask = train_and_eval(
            tr, val, ep_weights, seed=42,
            epochs=300, patience=50, name=tag)
        agg_preds.append(pred)
        agg_true.append(true)
        agg_mask.append(mask)

    if not agg_preds:
        log(f'  no valid folds, skip')
        continue

    all_pred = np.concatenate(agg_preds, axis=0)
    all_true = np.concatenate(agg_true, axis=0)
    all_mask = np.concatenate(agg_mask, axis=0)

    metrics = compute_metrics(all_pred, all_true, all_mask, y_mean, y_std, n_boot=300)
    log(f'  metrics: {metrics}')
    for ep_name, m in metrics.items():
        all_rows.append({
            'split': split_name,
            'endpoint': ep_name,
            'n': m['n'],
            'MAE': round(m['MAE'], 4),
            'MAE_lo': round(m['MAE_ci'][0], 4),
            'MAE_hi': round(m['MAE_ci'][1], 4),
            'RMSE': round(m['RMSE'], 4),
            'RMSE_lo': round(m['RMSE_ci'][0], 4),
            'RMSE_hi': round(m['RMSE_ci'][1], 4),
            'Spearman': round(m['Spearman'], 4) if not np.isnan(m['Spearman']) else np.nan,
            'Spearman_lo': round(m['Spearman_ci'][0], 4),
            'Spearman_hi': round(m['Spearman_ci'][1], 4),
            'R2': round(m['R2'], 4) if not np.isnan(m['R2']) else np.nan,
            'R2_lo': round(m['R2_ci'][0], 4),
            'R2_hi': round(m['R2_ci'][1], 4),
        })

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUT / 'zero_shot_results.csv', index=False)
log(f'\nResults saved: {len(results_df)} rows')

# ============================================================
# Table 1 (paper format): averaged per split × primary endpoint
# ============================================================
primary = {'logBAF': 'logBAF', 'logBCF': 'logBCF'}
table_rows = []
SPLIT_ORDER = ['Random split', 'Chemical-out split', 'Subclass-out split',
               'Organism-out split', 'Phylogeny-out split', 'Endpoint zero-shot']
for sname in SPLIT_ORDER:
    sub = results_df[results_df['split'] == sname]
    if sub.empty:
        continue
    # prefer logBAF if available else logBCF else first
    for ep_pref in ['logBAF', 'logBCF']:
        row = sub[sub['endpoint'] == ep_pref]
        if not row.empty:
            r = row.iloc[0]
            table_rows.append({
                'Benchmark': sname,
                'Endpoint': ep_pref,
                'N': r['n'],
                'MAE': f"{r['MAE']:.3f} [{r['MAE_lo']:.3f}–{r['MAE_hi']:.3f}]",
                'RMSE': f"{r['RMSE']:.3f} [{r['RMSE_lo']:.3f}–{r['RMSE_hi']:.3f}]",
                'Spearman': f"{r['Spearman']:.3f} [{r['Spearman_lo']:.3f}–{r['Spearman_hi']:.3f}]",
                'R²': f"{r['R2']:.3f} [{r['R2_lo']:.3f}–{r['R2_hi']:.3f}]",
            })
            break

table1 = pd.DataFrame(table_rows)
table1.to_csv(TABLES / 'table1_zeroshot.csv', index=False)
log('Table 1 saved')

# ============================================================
# Figure: bar chart of Spearman per split
# ============================================================
fig_df = results_df.dropna(subset=['Spearman'])
fig_df = fig_df[fig_df['endpoint'].isin(['logBAF', 'logBCF'])]
# keep one row per split (prefer BAF)
fig_rows = []
for sname in SPLIT_ORDER:
    sub = fig_df[fig_df['split'] == sname]
    for ep in ['logBAF', 'logBCF']:
        r = sub[sub['endpoint'] == ep]
        if not r.empty:
            fig_rows.append(r.iloc[0])
            break
if fig_rows:
    fig_df2 = pd.DataFrame(fig_rows)
    short_names = {
        'Random split': 'Random',
        'Chemical-out split': 'Chemical-out',
        'Subclass-out split': 'Subclass-out',
        'Organism-out split': 'Organism-out',
        'Phylogeny-out split': 'Phylogeny-out',
        'Endpoint zero-shot': 'Endpoint zero-shot',
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fig_df2))
    colors = [NC_PALETTE[i % len(NC_PALETTE)] for i in range(len(x))]
    bars = ax.bar(x, fig_df2['Spearman'].values, color=colors, width=0.6,
                  edgecolor='white', linewidth=0.8)
    # error bars
    lo = fig_df2['Spearman'].values - fig_df2['Spearman_lo'].values
    hi = fig_df2['Spearman_hi'].values - fig_df2['Spearman'].values
    ax.errorbar(x, fig_df2['Spearman'].values, yerr=[lo, hi],
                fmt='none', color='#333333', capsize=4, linewidth=1.2, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(s, s) for s in fig_df2['split'].values],
                       rotation=25, ha='right', fontsize=11)
    ax.set_ylabel("Spearman's ρ", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='#9E9E9E', ls='--', lw=0.8)
    neat_axes(ax)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGS / 'fig_zeroshot_spearman.png', dpi=300)
    plt.savefig(OUT / 'fig_zeroshot_spearman.png', dpi=300)
    plt.close()
    log('Figure saved: fig_zeroshot_spearman.png')

log('Stage 5 done.')
