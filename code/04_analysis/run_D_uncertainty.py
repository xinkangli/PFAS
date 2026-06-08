"""
Analysis D: Applicability domain & uncertainty analysis
UMAP colored by MC-Dropout uncertainty + per-class uncertainty distribution
Outputs: fig_D_uncertainty_umap.png, fig_D_uncertainty_class.png, table_D_uncertainty.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import NC_PALETTE, SEQ_CMAP, neat_axes, TIMES  # noqa

OUT  = ROOT / 'result/discussion'
S4   = ROOT / 'result/stage4_latent'
OUT.mkdir(exist_ok=True)

print('loading UMAP coords + meta ...')
U1   = np.load(S4/'umap_E1.npy')
meta = pd.read_csv(S4/'meta.csv')
N    = len(U1)
assert len(meta) == N

print('loading model2 predictions uncertainty ...')
m2 = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m2 = m2[m2['smiles_valid'] == True][['SMILES','unc_logBAF','unc_logBCF',
                                      'pred_logBAF','pred_logBCF']].copy()
m2['SMILES'] = m2['SMILES'].astype(str).str.strip()
meta['SMILES'] = meta['SMILES'].astype(str).str.strip()

# join uncertainty into meta (many-to-one, keep first match)
m2_dedup = m2.drop_duplicates('SMILES')
meta = meta.merge(m2_dedup, on='SMILES', how='left')
meta = meta.iloc[:N].reset_index(drop=True)

# load training data SMILES to mark them on UMAP
print('loading training SMILES ...')
df2 = pd.read_excel(ROOT/'model_data/Model2_Bioaccumulation.xlsx',
                    sheet_name='X_Y_joined', engine='openpyxl')
train_smis = set(df2['SMILES'].dropna().astype(str).str.strip().tolist())
meta['is_train'] = meta['SMILES'].isin(train_smis)
print(f'  training points on UMAP: {meta["is_train"].sum()}')

# pfas_class from master
master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
master = master[['SMILES','pfas_class']].dropna(subset=['SMILES']).drop_duplicates('SMILES')
master['SMILES'] = master['SMILES'].astype(str).str.strip()
meta = meta.merge(master, on='SMILES', how='left')
meta = meta.iloc[:N].reset_index(drop=True)

CLASS_MAP = {'PFCA':'PFCA','PFSA':'PFSA','FASA':'FASA',
             'FTOH':'FTOH','Ether':'Ether-PFAS'}
def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower():
            return v
    return 'Other'
meta['norm_class'] = meta['pfas_class'].apply(norm_cls)

# ── table ─────────────────────────────────────────────────────────────────────
CLASS_ORDER = ['PFCA','PFSA','FASA','FTOH','Ether-PFAS','Other']
rows = []
for cls in CLASS_ORDER:
    sub = meta[meta['norm_class'] == cls]
    unc = sub['unc_logBAF'].dropna()
    if len(unc) == 0:
        continue
    rows.append({
        'Functional class': cls,
        'n': len(unc),
        'unc_median': round(float(unc.median()), 3),
        'unc_mean':   round(float(unc.mean()), 3),
        'unc_q75':    round(float(unc.quantile(0.75)), 3),
        'unc_q95':    round(float(unc.quantile(0.95)), 3),
        'pct_unc>0.5': round(100*(unc > 0.5).sum()/len(unc), 1),
    })
unc_df = pd.DataFrame(rows)
unc_df.to_csv(OUT/'table_D_uncertainty.csv', index=False)
print(unc_df.to_string(index=False))

# ── Figure D1: UMAP colored by unc_logBAF ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: UMAP density / unc
unc_vals = meta['unc_logBAF'].values
unc_clip = np.clip(unc_vals, 0, np.nanpercentile(unc_vals[np.isfinite(unc_vals)], 98))
finite = np.isfinite(unc_clip)

ax = axes[0]
# unlabelled background
ax.scatter(U1[~finite, 0], U1[~finite, 1],
           c='#E8E8E8', s=2, alpha=0.3, edgecolors='none',
           rasterized=True, label='No uncertainty data')
sc = ax.scatter(U1[finite, 0], U1[finite, 1],
                c=unc_clip[finite], s=3, alpha=0.7,
                cmap='YlOrRd', edgecolors='none', rasterized=True,
                vmin=0, vmax=np.nanpercentile(unc_clip[finite], 97))
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Prediction uncertainty σ(logBAF)', fontsize=11)

# Overlay training points
train_mask = meta['is_train'].values
ax.scatter(U1[train_mask, 0], U1[train_mask, 1],
           c=NC_PALETTE[0], s=12, alpha=0.8, marker='*',
           edgecolors='white', linewidths=0.3, zorder=5,
           label=f'Training data (n={train_mask.sum()})')
ax.set_xlabel('UMAP dim 1', labelpad=5)
ax.set_ylabel('UMAP dim 2', labelpad=5)
ax.set_title('(a) Applicability domain: MC-Dropout uncertainty\n(transport latent space, n=30k)', pad=10)
ax.legend(loc='lower right', frameon=True, fontsize=9.5,
          framealpha=0.9, edgecolor='#cccccc')
neat_axes(ax)

# Right: per-class boxplot
ax2 = axes[1]
data_list = []
labels_list = []
for cls in CLASS_ORDER:
    sub = meta[meta['norm_class'] == cls]['unc_logBAF'].dropna()
    if len(sub) < 5:
        continue
    data_list.append(sub.clip(0, 1.5).values)
    labels_list.append(cls)

bp = ax2.boxplot(data_list, patch_artist=True, vert=True,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='o', markersize=2.5,
                                 alpha=0.4, markeredgewidth=0))
for patch, cls in zip(bp['boxes'], labels_list):
    i = CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)-1
    patch.set_facecolor(NC_PALETTE[i % len(NC_PALETTE)])
    patch.set_alpha(0.85)

ax2.axhline(0.5, color='#474747', lw=1.2, ls='--', alpha=0.8,
            label='High uncertainty (σ > 0.5)')
ax2.set_xticks(range(1, len(labels_list)+1))
ax2.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=10)
ax2.set_ylabel('Prediction uncertainty σ(logBAF)', labelpad=5)
ax2.set_title('(b) Uncertainty distribution by functional class', pad=10)
ax2.legend(loc='upper right', frameon=False, fontsize=9.5)
neat_axes(ax2)

fig.tight_layout(pad=2.0, w_pad=3.0)
plt.savefig(OUT/'fig_D_uncertainty.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig_D_uncertainty.png saved')

# ── Figure D2: high-uncertainty fraction bar chart ────────────────────────────
fig2, ax3 = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(unc_df))
bars = ax3.bar(x, unc_df['pct_unc>0.5'],
               color=[NC_PALETTE[i % len(NC_PALETTE)] for i in range(len(unc_df))],
               width=0.6, edgecolor='white', linewidth=0.5)
# annotate values on bars
for i, (bar, val) in enumerate(zip(bars, unc_df['pct_unc>0.5'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9.5)
ax3.set_xticks(x)
ax3.set_xticklabels(unc_df['Functional class'], rotation=30, ha='right')
ax3.set_ylabel('Fraction with σ(logBAF) > 0.5 (%)', labelpad=5)
ax3.set_title('High-uncertainty fraction by functional class\n(applicability domain data gaps)', pad=10)
ax3.set_ylim(0, max(unc_df['pct_unc>0.5']) * 1.25)
neat_axes(ax3)
fig2.tight_layout(pad=2.0)
plt.savefig(OUT/'fig_D_uncertainty_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig_D_uncertainty_bar.png saved')
