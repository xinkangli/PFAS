"""
Stage 4 — Latent space analysis (v2 NC-style).

Steps:
  1. Rebuild featurizer + model 1 from saved weights.
  2. Forward X_pool_full (subset to PFAS pool ~117k SMILES) and grab the
     128-d backbone embedding (E1 = transport latent).
  3. Build E0 (structure-only PCA-64 from raw 1057-d features) for comparison.
  4. UMAP both to 2D, save UMAP coords + 4 main figures + quant metrics.

Outputs (under result/stage4_latent/):
  latent_E1.npy              [N, 128]
  umap_E0.npy / umap_E1.npy  [N, 2]
  meta.csv                   per-SMILES: cf_subclass, persistence_label,
                             pred_logKow, pred_logBCF, pred_logBAF, n_F, ...
  fig4a_logKow.png
  fig4b_logBCF.png
  fig4c_persistence.png
  fig4d_subclass.png
  quant_metrics.csv          Spearman / silhouette / kNN-consistency
"""
from __future__ import annotations
import json, sys, os, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import umap

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'result'))
from _style import TIMES, NC_PALETTE, SEQ_CMAP, neat_axes, PERSIST_C  # noqa
import _style  # noqa  (apply rcParams)

OUT  = ROOT/'result/stage4_latent'
FIGS = ROOT/'result/figs'
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
LOG  = ROOT/'result/logs/stage4.log'
LOG.parent.mkdir(parents=True, exist_ok=True)
def log(msg, end='\n'):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, end=end, flush=True)
    with open(LOG, 'a') as f: f.write(s + end)

import featurize as _ft  # type: ignore
from model import MultiTaskTransportNet  # type: ignore

# ============== 1. Load saved model 1 ==============
log('load model 1')
meta = json.load(open(ROOT/'model_data/model1_out/model1_meta.json'))
endpoints = meta['endpoints']
feat_names = meta['feature_names']
in_mean = np.asarray(meta['in_mean'])
in_std  = np.asarray(meta['in_std'])
fill_med = np.asarray(meta['fill_median'])
y_mean = np.asarray(meta['target_mean'])
y_std  = np.asarray(meta['target_std'])
hp = meta['hyperparams']
hidden = tuple(hp['hidden'])
dropout = hp['dropout']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MultiTaskTransportNet(in_dim=len(feat_names), endpoints=endpoints,
                            hidden=hidden, dropout=dropout)
state = torch.load(ROOT/'model_data/model1_out/model1_state.pt', map_location=device)
net.load_state_dict(state)
net.eval().to(device)
log(f'  model loaded; in_dim={len(feat_names)}, endpoints={endpoints}')

# ============== 2. Featurize SMILES pool ==============
# Use the 127k pool, but for tractable UMAP we'll take strict PFAS + a
# random sample from the rest.
log('load pool SMILES')
preds = pd.read_csv(ROOT/'model_data/model1_out/model1_predictions.csv')
preds = preds[preds['smiles_valid'] == True].reset_index(drop=True)
log(f'  pool valid SMILES = {len(preds)}')

# Tag strict PFAS
master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
strict_smi = set(master.loc[master['pfas_class'].notna(), 'SMILES']
                 .dropna().astype(str).str.strip())
preds['__smi__'] = preds['SMILES'].astype(str).str.strip()
preds['is_strict'] = preds['__smi__'].isin(strict_smi)
log(f'  strict PFAS in valid pool: {preds["is_strict"].sum()}')

# Sample ~30k for UMAP: keep all strict + random from rest
N_SAMPLE = 30000
strict_idx = preds.index[preds['is_strict']].tolist()
other_idx  = preds.index[~preds['is_strict']].tolist()
rng = np.random.default_rng(42)
keep_other = rng.choice(other_idx, size=min(N_SAMPLE - len(strict_idx), len(other_idx)),
                        replace=False)
keep = sorted(set(strict_idx) | set(keep_other.tolist()))
sub = preds.loc[keep].reset_index(drop=True)
log(f'  UMAP set: {len(sub)} (strict={int(sub["is_strict"].sum())} + sampled rest)')

# Build features (1057-d) — re-run the featurizer; it needs F-stats join
log('featurize SMILES (1057-d)')
import openpyxl
wb = openpyxl.load_workbook(ROOT/'model_data/Model1_Transport.xlsx', data_only=True)
fstats_sheet = wb['F_stats_dict']
header = [c.value for c in next(fstats_sheet.iter_rows(min_row=1, max_row=1))]
fstats_rows = list(fstats_sheet.iter_rows(min_row=2, values_only=True))
fstats = pd.DataFrame(fstats_rows, columns=header)
log(f'  F_stats rows={len(fstats)} cols={list(fstats.columns)}')
fstats_dict = (fstats.dropna(subset=['SMILES']).set_index('SMILES')[
    ['n_F','n_CF1','n_CF2','n_CF3','n_CF4']].astype(float).to_dict('index'))
ftz = _ft.Featurizer()
X_arr, valid = ftz.transform(sub['SMILES'].tolist(), fstats_df=fstats, fstats_key='SMILES')
log(f'  X shape = {X_arr.shape}, valid = {valid.sum()}')
sub = sub.loc[valid].reset_index(drop=True)
X = X_arr[valid]

# Standardise like training
X_proc = X.copy()
nan_mask = ~np.isfinite(X_proc)
if nan_mask.any():
    X_proc[nan_mask] = np.broadcast_to(fill_med, X_proc.shape)[nan_mask]
X_proc = (X_proc - in_mean) / np.where(in_std > 1e-8, in_std, 1.0)

# ============== 3. Extract 128-d transport latent (E1) ==============
log('extract latent E1 (backbone last layer)')
hooks = {}
def hook_fn(_m, _i, out):  hooks['z'] = out.detach().cpu().numpy()
h = net.backbone.register_forward_hook(hook_fn)

BATCH = 4096
zs = []
with torch.no_grad():
    for i in range(0, len(X_proc), BATCH):
        xb = torch.from_numpy(X_proc[i:i+BATCH]).float().to(device)
        _ = net.backbone(xb)
        zs.append(hooks['z'])
h.remove()
E1 = np.concatenate(zs, axis=0)
log(f'  E1 shape = {E1.shape}')
np.save(OUT/'latent_E1.npy', E1)

# E0: structure-only PCA-64 of standardised X
log('PCA-64 on structure (E0)')
pca = PCA(n_components=min(64, X_proc.shape[1]), random_state=42)
E0 = pca.fit_transform(X_proc)
log(f'  E0 shape = {E0.shape}, explained var = {pca.explained_variance_ratio_[:5].sum():.3f}')

# ============== 4. UMAP both ==============
log('UMAP E1')
um1 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                metric='euclidean', random_state=42, n_jobs=-1)
U1 = um1.fit_transform(E1)
np.save(OUT/'umap_E1.npy', U1)
log('UMAP E0')
um0 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                metric='euclidean', random_state=42, n_jobs=-1)
U0 = um0.fit_transform(E0)
np.save(OUT/'umap_E0.npy', U0)

# ============== 5. Build colouring metadata ==============
log('build colour metadata')
meta_df = pd.DataFrame({'SMILES': sub['SMILES'].values})
_preds_slim = preds[['SMILES','pred_logKow','pred_water_solubility','is_strict']].drop_duplicates('SMILES')
meta_df = meta_df.merge(_preds_slim, on='SMILES', how='left')
m2_pred = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv',
                      usecols=['SMILES','pred_logBAF','pred_logBCF','pred_logBCFD'])
m2_pred = m2_pred.drop_duplicates('SMILES')
meta_df = meta_df.merge(m2_pred, on='SMILES', how='left')
meta_df = meta_df.iloc[:len(sub)].reset_index(drop=True)  # safety: keep exactly N rows

# clip outliers (some mixtures yield insane predictions)
for c in ['pred_logBCF','pred_logBAF','pred_logKow','pred_water_solubility']:
    if c in meta_df:
        s = meta_df[c]
        q1, q99 = s.quantile([0.01, 0.99])
        meta_df[c+'_clip'] = s.clip(q1, q99)

# ClassyFire subclass
cf = pd.read_csv(ROOT/'01_structure/classyfire/classyfire_PFAS_pool_subset.csv')
cf = cf.rename(columns={'chem_cf_subclass_name':'cf_subclass',
                        'chem_cf_class_name':'cf_class'}) if 'chem_cf_subclass_name' in cf.columns else cf
# join via InChIKey through PFAS_master_structure
mast = master[['SMILES','InChIKey']].dropna().drop_duplicates('InChIKey')
cf = cf.merge(mast, on='InChIKey', how='left').dropna(subset=['SMILES'])
cf_slim = cf[['SMILES','cf_subclass','cf_class']].drop_duplicates('SMILES')
meta_df = meta_df.merge(cf_slim, on='SMILES', how='left')

# persistence label by abbreviation -> SMILES is hard, but compounds.csv has cas
pers = pd.read_csv(ROOT/'05_persistence_transformation/persistence.csv')
comp = pd.read_csv(ROOT/'05_persistence_transformation/compounds.csv', on_bad_lines='skip')
pers_merge = pers.merge(comp[['compound_id','cas']], on='compound_id', how='left')
pers_by_cas = (pers_merge.groupby('cas')['persistence_label'].first().reset_index())
mast2 = master[['SMILES','CAS']].dropna()
mast2['cas'] = mast2['CAS'].astype(str).str.upper().str.strip()
pers_by_cas['cas'] = pers_by_cas['cas'].astype(str).str.upper().str.strip()
pers_smi = mast2.merge(pers_by_cas, on='cas', how='inner')[['SMILES','persistence_label']]
meta_df = meta_df.merge(pers_smi.drop_duplicates('SMILES'), on='SMILES', how='left')

meta_df.to_csv(OUT/'meta.csv', index=False)
log(f'  meta saved, persistence labelled = {meta_df["persistence_label"].notna().sum()}, '
    f'cf_subclass labelled = {meta_df["cf_subclass"].notna().sum()}')


# ============== 6. Plot 4 main figures (use E1 / U1) ==============
def scatter_continuous(U, vals, title, cbar_label, fname, cmap=SEQ_CMAP):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    mask = np.isfinite(vals)
    sc = ax.scatter(U[mask, 0], U[mask, 1], c=vals[mask], s=4, alpha=0.7,
                    cmap=cmap, edgecolors='none', rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label(cbar_label, fontsize=13)
    ax.set_xlabel('UMAP dim 1');  ax.set_ylabel('UMAP dim 2')
    ax.set_title(title, pad=14)
    neat_axes(ax)
    plt.tight_layout()
    plt.savefig(FIGS/fname); plt.savefig(OUT/fname)
    plt.close()

def scatter_categorical(U, labels, title, fname, top_k=8):
    series = pd.Series(labels)
    counts = series.value_counts()
    top = counts.head(top_k).index.tolist()
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # background grey for non-labelled
    other_mask = (~series.isin(top)) & series.notna()
    nan_mask   = series.isna()
    ax.scatter(U[nan_mask.values, 0], U[nan_mask.values, 1], c='#E0E0E0', s=2.5,
               alpha=0.4, edgecolors='none', rasterized=True, label='_nolegend_')
    ax.scatter(U[other_mask.values, 0], U[other_mask.values, 1], c='#9E9E9E', s=3,
               alpha=0.5, edgecolors='none', rasterized=True, label='other')
    for i, lab in enumerate(top):
        m = (series == lab).values
        ax.scatter(U[m, 0], U[m, 1], c=NC_PALETTE[i % len(NC_PALETTE)], s=10,
                   alpha=0.9, edgecolors='white', linewidths=0.3, label=str(lab))
    ax.set_xlabel('UMAP dim 1'); ax.set_ylabel('UMAP dim 2')
    ax.set_title(title, pad=14)
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=10, title='', title_fontsize=11)
    neat_axes(ax)
    plt.tight_layout()
    plt.savefig(FIGS/fname); plt.savefig(OUT/fname)
    plt.close()


log('plot fig 4a-d')
scatter_continuous(U1, meta_df['pred_logKow_clip'].values,
    'Transport latent space coloured by predicted log K$_{ow}$',
    'predicted log K$_{ow}$', 'fig4a_logKow.png')
scatter_continuous(U1, meta_df['pred_logBCF_clip'].values,
    'Transport latent space coloured by predicted log BCF',
    'predicted log BCF', 'fig4b_logBCF.png')

# 4c: persistence — categorical
fig, ax = plt.subplots(figsize=(7.5, 6.0))
labels = meta_df['persistence_label'].fillna('unlabelled').values
nan_mask = (labels == 'unlabelled')
ax.scatter(U1[nan_mask, 0], U1[nan_mask, 1], c='#E0E0E0', s=2.5, alpha=0.4,
           edgecolors='none', rasterized=True, label='unlabelled')
order = ['vP','P','not-P_self']
for i, lab in enumerate(order):
    m = (labels == lab)
    if not m.any(): continue
    ax.scatter(U1[m, 0], U1[m, 1], c=PERSIST_C.get(lab, NC_PALETTE[i]),
               s=22, alpha=0.95, edgecolors='white', linewidths=0.4, label=lab)
ax.set_xlabel('UMAP dim 1'); ax.set_ylabel('UMAP dim 2')
ax.set_title('Transport latent space coloured by persistence label', pad=14)
ax.legend(frameon=False, loc='upper right', fontsize=11)
neat_axes(ax)
plt.tight_layout(); plt.savefig(FIGS/'fig4c_persistence.png'); plt.savefig(OUT/'fig4c_persistence.png'); plt.close()

# 4d: ClassyFire subclass on strict PFAS only
sub_mask = meta_df['is_strict'].values & meta_df['cf_subclass'].notna().values
fig, ax = plt.subplots(figsize=(8.5, 6.0))
ax.scatter(U1[~sub_mask, 0], U1[~sub_mask, 1], c='#E0E0E0', s=2.5, alpha=0.4,
           edgecolors='none', rasterized=True)
strict_labels = meta_df.loc[sub_mask, 'cf_subclass']
counts = strict_labels.value_counts()
top = counts.head(8).index.tolist()
for i, lab in enumerate(top):
    m = (meta_df['cf_subclass'] == lab).values
    ax.scatter(U1[m, 0], U1[m, 1], c=NC_PALETTE[i % len(NC_PALETTE)], s=22,
               alpha=0.95, edgecolors='white', linewidths=0.4, label=str(lab))
other = sub_mask & (~meta_df['cf_subclass'].isin(top).values)
ax.scatter(U1[other, 0], U1[other, 1], c='#9E9E9E', s=10, alpha=0.6,
           edgecolors='none', label='other strict PFAS')
ax.set_xlabel('UMAP dim 1'); ax.set_ylabel('UMAP dim 2')
ax.set_title('Transport latent space coloured by ClassyFire subclass (strict PFAS)', pad=14)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
neat_axes(ax)
plt.tight_layout(); plt.savefig(FIGS/'fig4d_subclass.png'); plt.savefig(OUT/'fig4d_subclass.png'); plt.close()

# ============== 7. Quantitative metrics: E0 vs E1 ==============
log('quantitative metrics')

def latent_distance_corr(E, target, n_pairs=5000, seed=0):
    rng = np.random.default_rng(seed)
    finite = np.where(np.isfinite(target))[0]
    if len(finite) < 50: return np.nan
    pairs = rng.choice(finite, size=(n_pairs, 2))
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    dE = np.linalg.norm(E[pairs[:,0]] - E[pairs[:,1]], axis=1)
    dY = np.abs(target[pairs[:,0]] - target[pairs[:,1]])
    rho, _ = spearmanr(dE, dY)
    return rho

def silhouette_by(E, labels):
    s = pd.Series(labels)
    keep = s.notna() & (s.astype(str).str.lower() != 'unlabelled')
    s = s[keep]; E = E[keep.values]
    counts = s.value_counts()
    big = counts[counts >= 8].index.tolist()
    if len(big) < 2: return np.nan
    keep2 = s.isin(big)
    if keep2.sum() < 20: return np.nan
    if keep2.sum() > 8000:  # subsample
        rng = np.random.default_rng(0)
        idx = rng.choice(np.where(keep2)[0], size=8000, replace=False)
        return float(silhouette_score(E[idx], s.iloc[idx].values))
    return float(silhouette_score(E[keep2.values], s[keep2].values))

def knn_consistency(E, labels, k=15):
    s = pd.Series(labels)
    keep = s.notna() & (s.astype(str).str.lower() != 'unlabelled')
    if keep.sum() < 50: return np.nan
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(E[keep.values])
    _, ind = nbrs.kneighbors(E[keep.values])
    same = (s[keep].values[ind[:,1:]] == s[keep].values[:,None]).mean()
    return float(same)

def all_metrics(E, name):
    rows = []
    for c in ['pred_logKow', 'pred_logBCF', 'pred_logBAF']:
        rho = latent_distance_corr(E, meta_df[c].values, n_pairs=8000)
        rows.append({'embedding':name, 'metric':f'Spearman(d_E, |Δ {c}|)', 'value':rho})
    rows.append({'embedding':name,
                 'metric':'silhouette by ClassyFire subclass (strict)',
                 'value': silhouette_by(E, meta_df['cf_subclass'].values)})
    rows.append({'embedding':name,
                 'metric':'silhouette by persistence label',
                 'value': silhouette_by(E, meta_df['persistence_label'].values)})
    rows.append({'embedding':name,
                 'metric':'kNN-15 consistency by subclass',
                 'value': knn_consistency(E, meta_df['cf_subclass'].values)})
    rows.append({'embedding':name,
                 'metric':'kNN-15 consistency by persistence',
                 'value': knn_consistency(E, meta_df['persistence_label'].values)})
    return rows

rows = []
rows += all_metrics(E0, 'E0 structure-only (PCA-64)')
rows += all_metrics(E1, 'E1 transport latent (128d)')
qm = pd.DataFrame(rows)
qm.to_csv(OUT/'quant_metrics.csv', index=False)
log('  metrics:')
log(qm.to_string(index=False))

log('stage 4 done')
