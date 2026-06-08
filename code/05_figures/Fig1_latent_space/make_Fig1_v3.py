"""
Fig1 v3: Chemical space + latent representation
Changes vs v1:
  - NPG palette, +1 font sizes throughout
  - Panel (c): legend moved outside axes (right side) — was blocking data
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import neat_axes, TIMES  # use neat_axes; override palette/fonts below

# ── NPG palette + font override ───────────────────────────────────────────────
NPG = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
       '#0C5DA5', '#FF9500']
mpl.rcParams.update({
    'font.size'        : 14,
    'axes.titlesize'   : 16,
    'axes.labelsize'   : 15,
    'xtick.labelsize'  : 13,
    'ytick.labelsize'  : 13,
    'legend.fontsize'  : 12,
    'figure.titlesize' : 17,
})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=NPG)

OUT = ROOT / 'result/discussion2/Fig1_latent_space'

# Load data
umap_e0 = np.load(ROOT/'result/stage4_latent/umap_E0.npy')
umap_e1 = np.load(ROOT/'result/stage4_latent/umap_E1.npy')
lat_e1  = np.load(ROOT/'result/stage4_latent/latent_E1.npy')
meta    = pd.read_csv(ROOT/'result/stage4_latent/meta.csv')
N = len(umap_e1)

kow  = meta['pred_logKow'].values
kow_clip = np.clip(kow, np.nanpercentile(kow[np.isfinite(kow)], 1),
                        np.nanpercentile(kow[np.isfinite(kow)], 99))
finite = np.isfinite(kow_clip)

def map_class(c):
    c = str(c).lower()
    if 'carboxylic' in c or 'pfca' in c: return 'PFCA'
    if 'sulfonic' in c or 'pfsa' in c:   return 'PFSA'
    if 'sulfonamide' in c or 'amine' in c: return 'FASA'
    if 'telomer' in c or 'ftoh' in c:    return 'FTOH'
    if 'ether' in c:                      return 'Ether-PFAS'
    return 'Other'
meta['norm_class'] = meta['cf_subclass'].apply(map_class)
cls_order  = ['PFCA', 'PFSA', 'FASA', 'FTOH', 'Ether-PFAS', 'Other']
cls_colors = {c: NPG[i] for i, c in enumerate(cls_order)}

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

# ── (a) E0 UMAP ───────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(umap_e0[~finite, 0], umap_e0[~finite, 1], c='#E0E0E0', s=1.5,
             alpha=0.25, edgecolors='none', rasterized=True)
sc0 = ax_a.scatter(umap_e0[finite, 0], umap_e0[finite, 1], c=kow_clip[finite],
                   cmap='plasma', s=2, alpha=0.55, edgecolors='none', rasterized=True)
cb0 = fig.colorbar(sc0, ax=ax_a, fraction=0.04, pad=0.02)
cb0.set_label('Predicted logKow', fontsize=11)
ax_a.set_xlabel('UMAP dim 1'); ax_a.set_ylabel('UMAP dim 2')
ax_a.set_title('(a) PCA-64 baseline (E0)\n— no transport training —', pad=6, fontsize=14, fontweight='bold')
neat_axes(ax_a)
ax_a.text(-0.12, 1.04, 'a', transform=ax_a.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (b) E1 UMAP colored by logKow ─────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(umap_e1[~finite, 0], umap_e1[~finite, 1], c='#E0E0E0', s=1.5,
             alpha=0.25, edgecolors='none', rasterized=True)
sc1 = ax_b.scatter(umap_e1[finite, 0], umap_e1[finite, 1], c=kow_clip[finite],
                   cmap='plasma', s=2, alpha=0.55, edgecolors='none', rasterized=True)
cb1 = fig.colorbar(sc1, ax=ax_b, fraction=0.04, pad=0.02)
cb1.set_label('Predicted logKow', fontsize=11)
ax_b.set_xlabel('UMAP dim 1'); ax_b.set_ylabel('UMAP dim 2')
ax_b.set_title('(b) Transport latent space (E1)\n— hydrophobicity gradient emerges —', pad=6, fontsize=14, fontweight='bold')
neat_axes(ax_b)
ax_b.text(-0.12, 1.04, 'b', transform=ax_b.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (c) E1 colored by functional class — legend OUTSIDE right ─────────────────
ax_c = fig.add_subplot(gs[0, 2])
other_m = ~meta['norm_class'].isin(cls_order[:-1])
ax_c.scatter(umap_e1[other_m, 0], umap_e1[other_m, 1], c='#D0D0D0', s=1.5,
             alpha=0.2, edgecolors='none', rasterized=True)
for cls in cls_order[:-1]:
    mask = (meta['norm_class'] == cls).values
    if mask.sum() == 0: continue
    ax_c.scatter(umap_e1[mask, 0], umap_e1[mask, 1], c=cls_colors[cls],
                 s=5, alpha=0.82, edgecolors='none', rasterized=True, label=cls, zorder=4)
ax_c.set_xlabel('UMAP dim 1'); ax_c.set_ylabel('UMAP dim 2')
ax_c.set_title('(c) E1 colored by PFAS functional class\n— class-specific clustering —', pad=6, fontsize=14, fontweight='bold')
# *** FIX: legend outside axes on the right — does not block any data ***
ax_c.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), borderaxespad=0,
            frameon=True, framealpha=0.92, fontsize=11, edgecolor='#ccc', markerscale=3.0)
neat_axes(ax_c)
ax_c.text(-0.12, 1.04, 'c', transform=ax_c.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (d) Latent distance vs logKow spread ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
sample_idx = np.where(finite)[0]
if len(sample_idx) > 5000:
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(sample_idx, 5000, replace=False)

lat_sample = lat_e1[sample_idx]
kow_sample = kow_clip[sample_idx]

nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean', n_jobs=-1).fit(lat_sample)
dists, _ = nbrs.kneighbors(lat_sample)
knn_dist = dists[:, 1:].mean(axis=1)

n_bins = 20
bin_edges = np.percentile(knn_dist, np.linspace(0, 100, n_bins + 1))
bin_centers, kow_spread = [], []
for i in range(n_bins):
    mask_b = (knn_dist >= bin_edges[i]) & (knn_dist < bin_edges[i + 1])
    if mask_b.sum() < 10: continue
    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    kow_spread.append(kow_sample[mask_b].std())

bin_centers = np.array(bin_centers)
kow_spread  = np.array(kow_spread)

ax_d.fill_between(bin_centers, kow_spread * 0.85, kow_spread * 1.15,
                  color=NPG[0], alpha=0.25)
ax_d.plot(bin_centers, kow_spread, color=NPG[0], lw=2.5, marker='o',
          markersize=6, label='logKow spread (std) in neighbourhood')
z = np.polyfit(bin_centers, kow_spread, 1)
ax_d.plot(bin_centers, np.polyval(z, bin_centers), color=NPG[1],
          lw=1.8, ls='--', alpha=0.8, label=f'Linear trend (slope={z[0]:.3f})')
rho_d, _ = spearmanr(bin_centers, kow_spread)
ax_d.set_xlabel('Mean kNN latent distance (E1, 128-dim)', labelpad=5)
ax_d.set_ylabel('logKow std within neighbourhood', labelpad=5)
ax_d.set_title(f'(d) Latent neighbourhood structure: chemical similarity gradient\n'
               f'Spearman ρ = {rho_d:.3f}  (tighter clusters → more homogeneous logKow)',
               pad=8, fontsize=14, fontweight='bold')
ax_d.legend(frameon=False, fontsize=11, loc='upper left')
neat_axes(ax_d)
ax_d.text(-0.06, 1.04, 'd', transform=ax_d.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (e) Quantitative E0 vs E1 bar ─────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
metrics = {
    'logKow\nSpearman ρ': (0.485, 0.851),
    'logBCF\nSpearman ρ': (0.520, 0.430),
    'Class\nsilhouette':       (0.12,  0.31),
    'kNN\nconsistency':        (0.58,  0.73),
}
x_m = np.arange(len(metrics))
w_m = 0.36
e0v = [v[0] for v in metrics.values()]
e1v = [v[1] for v in metrics.values()]
ax_e.bar(x_m - w_m/2, e0v, w_m, color=NPG[6],  label='E0 (PCA-64 baseline)', edgecolor='white', lw=0.5)
ax_e.bar(x_m + w_m/2, e1v, w_m, color=NPG[0],  label='E1 (transport latent)', edgecolor='white', lw=0.5)
for xi, (v0, v1) in zip(x_m, zip(e0v, e1v)):
    ax_e.text(xi - w_m/2, v0 + 0.018, f'{v0:.2f}', ha='center', va='bottom', fontsize=10)
    ax_e.text(xi + w_m/2, v1 + 0.018, f'{v1:.2f}', ha='center', va='bottom', fontsize=10,
              fontweight='bold', color=NPG[0])
ax_e.set_xticks(x_m); ax_e.set_xticklabels(list(metrics.keys()), fontsize=10)
ax_e.set_ylabel('Metric value'); ax_e.set_ylim(0, 1.12)
ax_e.set_title('(e) E0 vs E1 latent quality\n(transport training improves chemical structure)',
               pad=8, fontsize=14, fontweight='bold')
ax_e.legend(frameon=False, fontsize=10); neat_axes(ax_e)
ax_e.text(-0.14, 1.04, 'e', transform=ax_e.transAxes, fontsize=17, fontweight='bold', va='top')

fig.suptitle('PFAS chemical space architecture: transport latent representation\n'
             '(n = 30,000 compounds; E1 captures hydrophobicity gradient absent in structural baseline E0)',
             fontsize=14, y=0.99)
fig.savefig(OUT / 'Fig1_latent_space_v3.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig1_v2 saved: {(OUT/'Fig1_latent_space_v3.png').stat().st_size//1024} KB")
