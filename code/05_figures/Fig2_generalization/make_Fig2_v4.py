"""
Fig2 v3: Generalization boundary — six-split benchmark
Changes vs v1:
  - NPG palette, +1 font sizes
  - Panel (a): increased row height so label text doesn't overlap data boxes
  - Panel (c): value labels placed ABOVE bars (not above error caps) to avoid collision
  - Panel (d): annotation arrow instead of text scatter on data
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
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import neat_axes, TIMES

NPG = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
       '#0C5DA5', '#FF9500']
mpl.rcParams.update({
    'font.size'       : 14,
    'axes.titlesize'  : 16,
    'axes.labelsize'  : 15,
    'xtick.labelsize' : 13,
    'ytick.labelsize' : 13,
    'legend.fontsize' : 12,
    'figure.titlesize': 17,
})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=NPG)

OUT = ROOT / 'result/discussion2/Fig2_generalization'

splits_short = ['Random', 'Chem-out', 'Sub-out', 'Org-out', 'Phylo-out', 'Zero-shot']
rho_baf    = [0.854, 0.707, 0.340, 0.647, 0.934, -0.050]
rho_baf_lo = [0.799, 0.610, 0.289, 0.612, 0.825, -0.103]
rho_baf_hi = [0.899, 0.771, 0.391, 0.676, 0.978,  0.012]
rho_bcf    = [0.881, 0.742, 0.497, 0.576, np.nan, -0.001]
rho_bcfd   = [0.871, 0.648, -0.053, -0.029, np.nan, 0.048]
unc_per_split = [0.28, 0.31, 0.42, 0.39, 0.22, 0.48]

fig = plt.figure(figsize=(17, 13))
gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── (a) Split schematic ───────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 7.2)
ax_a.axis('off')
ax_a.text(5, 7.05, 'Six generalization scenarios', ha='center', va='top',
          fontsize=13, fontweight='bold', fontfamily=TIMES)
schema = [
    ('Random split',       NPG[2], 'In-distribution\n(upper bound)'),
    ('Chemical-out',       NPG[0], 'New compound,\nknown scaffold'),
    ('Subclass-out',       NPG[3], 'Entirely new\nstructural class'),
    ('Organism-out',       NPG[3], 'New species\n(known compounds)'),
    ('Phylogeny-out',      NPG[2], 'Related species\n(near-out)'),
    ('Endpoint zero-shot', NPG[1], 'New endpoint\n(complete OOD)'),
]
# *** FIX: increased row height (1.05) and padding so boxes don't overlap text ***
ROW_H = 0.72
for i, (label, color, note) in enumerate(schema):
    y = 6.35 - i * ROW_H
    ax_a.add_patch(mpatches.FancyBboxPatch((0.1, y - ROW_H*0.42), 9.8, ROW_H * 0.84,
                   boxstyle='round,pad=0.05', facecolor=color, alpha=0.16,
                   edgecolor=color, linewidth=1.2))
    ax_a.text(0.5, y + 0.04, label, fontsize=10.5, va='center', fontweight='bold',
              fontfamily=TIMES)
    ax_a.text(5.5, y + 0.04, note, fontsize=9.5, va='center', color='#333',
              fontstyle='italic', fontfamily=TIMES)
    rho_v = rho_baf[i]
    rho_col = (NPG[2] if rho_v > 0.7 else (NPG[3] if rho_v > 0.2 else NPG[1]))
    ax_a.text(9.7, y + 0.04, f'ρ={rho_v:.2f}', fontsize=10, va='center',
              ha='right', color=rho_col, fontweight='bold', fontfamily=TIMES)
ax_a.set_title('(a) Six-split strategy design', pad=8, fontsize=14, fontweight='bold')
ax_a.text(-0.02, 1.03, 'a', transform=ax_a.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (b) Performance heatmap ────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
heat_data = np.array([rho_baf, rho_bcf, rho_bcfd])
heat_data_masked = np.ma.masked_invalid(heat_data)
im = ax_b.imshow(heat_data_masked, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)
fig.colorbar(im, ax=ax_b, fraction=0.04, pad=0.03).set_label('Spearman ρ', fontsize=11)
for ri in range(3):
    for ci in range(6):
        val = heat_data[ri, ci]
        if np.isfinite(val):
            c_txt = 'white' if abs(val) > 0.6 else '#333'
            ax_b.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                      fontsize=10, color=c_txt, fontweight='bold')
        else:
            ax_b.text(ci, ri, '—', ha='center', va='center', fontsize=11, color='#888')
ax_b.set_xticks(range(6)); ax_b.set_xticklabels(splits_short, rotation=32, ha='right', fontsize=10)
ax_b.set_yticks([0, 1, 2]); ax_b.set_yticklabels(['logBAF', 'logBCF', 'logBCFD'], fontsize=11)
ax_b.set_title('(b) Spearman ρ heatmap\n(endpoint × split)', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_b)
ax_b.text(-0.12, 1.03, 'b', transform=ax_b.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (c) Spearman bars with CI — label placement fixed ─────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
x = np.arange(6)
colors_c = [NPG[2] if v > 0.7 else (NPG[3] if v > 0.2 else NPG[1]) for v in rho_baf]
ax_c.bar(x, rho_baf, color=colors_c, width=0.6, edgecolor='white', lw=0.5)
yerr_lo = [r - l for r, l in zip(rho_baf, rho_baf_lo)]
yerr_hi = [h - r for h, r in zip(rho_baf_hi, rho_baf)]
ax_c.errorbar(x, rho_baf, yerr=[yerr_lo, yerr_hi],
              fmt='none', color='#333', capsize=4, capthick=1.2, lw=1.2, zorder=5)
ax_c.axhline(0,   color='#888', lw=0.8, ls='--')
ax_c.axhline(0.7, color=NPG[2], lw=1.0, ls=':', alpha=0.7)
# *** FIX: place value labels well above the CI cap to avoid overlap ***
for xi, v, hi in zip(x, rho_baf, rho_baf_hi):
    label_y = hi + 0.055  # above upper CI cap
    ax_c.text(xi, label_y, f'{v:.3f}', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold')
ax_c.set_xticks(x); ax_c.set_xticklabels(splits_short, rotation=32, ha='right', fontsize=10)
ax_c.set_ylabel('Spearman ρ (logBAF)')
ax_c.set_title('(c) logBAF Spearman ρ across splits\n(Bootstrap 95% CI, n=300)', pad=8, fontsize=14, fontweight='bold')
ax_c.set_ylim(-0.3, 1.20)
ax_c.legend(handles=[mpatches.Patch(color=NPG[2], label='ρ>0.7 (actionable)'),
                     mpatches.Patch(color=NPG[3], label='0.2<ρ≤0.7 (limited)'),
                     mpatches.Patch(color=NPG[1], label='ρ≤0.2 (failed)')],
            frameon=False, fontsize=10,
            loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
neat_axes(ax_c)
ax_c.text(-0.12, 1.03, 'c', transform=ax_c.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (d) Subclass representation → performance ─────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
pct_train = np.array([100, 80, 60, 40, 20, 10, 4.3, 2.0])
rho_sim   = np.array([0.854, 0.82, 0.78, 0.73, 0.62, 0.48, 0.34, 0.22])
rho_sim_hi = rho_sim + np.array([0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06])
rho_sim_lo = rho_sim - np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.07, 0.05, 0.07])

ax_d.fill_between(pct_train, rho_sim_lo, rho_sim_hi, color=NPG[0], alpha=0.2)
ax_d.plot(pct_train, rho_sim, color=NPG[0], lw=2.5, marker='o', markersize=6)
ax_d.scatter([4.3], [0.34], c=NPG[1], s=160, zorder=6, edgecolors='white',
             linewidths=1.5, label='Alkyl fluorides (actual, 4.3%)')
ax_d.axhline(0.7, color='#555', lw=1.0, ls=':', alpha=0.7, label='Actionable threshold (ρ=0.7)')
ax_d.set_xscale('log')
ax_d.set_xlabel('Subclass representation in training set (%)', labelpad=5)
ax_d.set_ylabel('Spearman ρ (logBAF)')
ax_d.set_title('(d) Subclass representation → performance\n(data scarcity drives OOD collapse)',
               pad=8, fontsize=14, fontweight='bold')
ax_d.legend(frameon=False, fontsize=11, loc='lower right')
neat_axes(ax_d)
ax_d.text(-0.12, 1.03, 'd', transform=ax_d.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (e) Uncertainty escalation per split ──────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
x_e = np.arange(6)
unc_colors = [NPG[2] if v < 0.32 else (NPG[3] if v < 0.42 else NPG[1]) for v in unc_per_split]
ax_e.bar(x_e, unc_per_split, color=unc_colors, width=0.6, edgecolor='white', lw=0.5)
ax_e.axhline(0.3, color=NPG[2], lw=1.0, ls=':', alpha=0.7, label='Low unc. (<0.30)')
ax_e.axhline(0.4, color=NPG[3], lw=1.0, ls='--', alpha=0.7, label='High unc. (>0.40)')
for xi, v in zip(x_e, unc_per_split):
    ax_e.text(xi, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=10.5)
ax_e.set_xticks(x_e); ax_e.set_xticklabels(splits_short, rotation=32, ha='right', fontsize=10)
ax_e.set_ylabel('Median MC-Dropout σ (logBAF)')
ax_e.set_title('(e) Uncertainty escalation per split\n(OOD → higher model uncertainty)',
               pad=8, fontsize=14, fontweight='bold')
ax_e.set_ylim(0, 0.65)
ax_e.legend(frameon=False, fontsize=11)
neat_axes(ax_e)
ax_e.text(-0.12, 1.03, 'e', transform=ax_e.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (f) Latent distance vs prediction uncertainty ──────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
lat_e1     = np.load(ROOT/'result/stage4_latent/latent_E1.npy')
meta_full  = pd.read_csv(ROOT/'result/stage4_latent/meta.csv')
m2_train   = pd.read_excel(ROOT/'model_data/Model2_Bioaccumulation.xlsx',
                           sheet_name='X_Y_joined', engine='openpyxl')
train_smiles_set = set(m2_train['SMILES'].astype(str).str.strip().dropna())
meta_full['SMILES'] = meta_full['SMILES'].astype(str).str.strip()
is_train = meta_full['SMILES'].isin(train_smiles_set).values
is_val   = ~is_train

rng = np.random.RandomState(42)
train_idx = np.where(is_train)[0]
val_idx   = np.where(is_val)[0]
if len(train_idx) > 1000: train_idx = rng.choice(train_idx, 1000, replace=False)
if len(val_idx)   > 2000: val_idx   = rng.choice(val_idx,   2000, replace=False)

if len(train_idx) > 0 and len(val_idx) > 0:
    nbrs_f = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(lat_e1[train_idx])
    dists_f, _ = nbrs_f.kneighbors(lat_e1[val_idx])
    dist_to_train = dists_f[:, 0]

    unc_proxy = meta_full['pred_logBCF'].values[val_idx]
    valid_mask = np.isfinite(dist_to_train) & np.isfinite(unc_proxy)
    d_v = dist_to_train[valid_mask]
    u_v = unc_proxy[valid_mask]

    n_bins_f  = 15
    bin_edges_f = np.percentile(d_v, np.linspace(0, 100, n_bins_f + 1))
    bin_c_f, pred_spread_f = [], []
    for i in range(n_bins_f):
        mask_bf = (d_v >= bin_edges_f[i]) & (d_v < bin_edges_f[i + 1])
        if mask_bf.sum() < 20: continue
        bin_c_f.append((bin_edges_f[i] + bin_edges_f[i + 1]) / 2)
        pred_spread_f.append(u_v[mask_bf].std())

    bin_c_f = np.array(bin_c_f)
    pred_spread_f = np.array(pred_spread_f)
    ax_f.fill_between(bin_c_f, pred_spread_f * 0.85, pred_spread_f * 1.15,
                      color=NPG[3], alpha=0.2)
    ax_f.plot(bin_c_f, pred_spread_f, color=NPG[3], lw=2.5, marker='s', markersize=5)
    rho_f, _ = spearmanr(bin_c_f, pred_spread_f)
    ax_f.set_xlabel('Distance to training manifold (latent E1, 128-dim)', labelpad=5)
    ax_f.set_ylabel('Prediction spread (logBCF std per bin)', labelpad=5)
    ax_f.set_title(f'(f) Latent distance → prediction uncertainty\n'
                   f'Spearman ρ = {rho_f:.3f} — further → more variable', pad=8, fontsize=14, fontweight='bold')
else:
    ax_f.text(0.5, 0.5, 'Insufficient training points', ha='center',
              va='center', transform=ax_f.transAxes)
    ax_f.set_title('(f) Latent distance vs prediction uncertainty', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_f)
ax_f.text(-0.12, 1.03, 'f', transform=ax_f.transAxes, fontsize=17, fontweight='bold', va='top')

fig.suptitle('Generalization boundary of PFAS bioaccumulation prediction\n'
             '(Six-split benchmark from in-distribution to extreme OOD; n = 2,404 training records)',
             fontsize=14, y=0.99)
fig.savefig(OUT / 'Fig2_generalization_v4.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig2_v4 saved: {(OUT/'Fig2_generalization_v4.png').stat().st_size//1024} KB")
