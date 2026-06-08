"""
Fig6 v3: External validation — NPG palette, +1 fonts, subplot titles fontsize=14 bold
Saves: Fig6_EV1_v3.png, Fig6_EV2_v3.png in Fig6_external_validation/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT / 'result'))
from _style import neat_axes, TIMES

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

WB  = ROOT / 'result/waibuyanzheng'
OUT = ROOT / 'result/discussion2/Fig6_external_validation'

ext_valid  = pd.read_csv(WB / 'step4_validation_set.csv')
metrics_df = pd.read_csv(WB / 'step5_regression_metrics.csv')
cls_df     = pd.read_csv(WB / 'step6_threshold_classification.csv')
B_THRESH   = 3.3;  VB_THRESH = 4.3

baf_ext  = ext_valid[ext_valid['endpoint'] == 'logBAF'].copy()
tmf_ext  = ext_valid[ext_valid['endpoint'] == 'logTMF'].copy()
bmf_ext  = ext_valid[ext_valid['endpoint'] == 'logBMF'].copy()
bcf_baf  = ext_valid[ext_valid['endpoint'].isin(['logBAF','logBCF'])].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# EV1: four-panel validation
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

# (a) Direct logBAF
ax = axes[0]
obs  = baf_ext['log_value'].values
pred = baf_ext['pred_value'].values
fin  = np.isfinite(obs) & np.isfinite(pred)
obs, pred = obs[fin], pred[fin]
if len(obs) >= 3:
    rho, pval = spearmanr(obs, pred)
    mae = np.mean(np.abs(obs - pred))
    ax.scatter(obs, pred, c=NPG[0], s=65, alpha=0.88, edgecolors='white', lw=0.7, zorder=4)
    lim = [min(obs.min(), pred.min()) - 0.3, max(obs.max(), pred.max()) + 0.3]
    ax.plot(lim, lim, 'k--', lw=1.2, alpha=0.6, label='y = x (perfect)')
    ax.axhline(B_THRESH, color=NPG[3], lw=1.0, ls=':', alpha=0.7)
    ax.axvline(B_THRESH, color=NPG[3], lw=1.0, ls=':', alpha=0.7, label='B threshold (3.3)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Observed logBAF (external literature)', labelpad=5)
    ax.set_ylabel('Predicted logBAF (Model 2)', labelpad=5)
    ax.set_title(f'(a) External logBAF validation\nn = {len(obs)}, Spearman ρ = {rho:.3f}, MAE = {mae:.2f}',
                 pad=8, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    for i, row in baf_ext[fin].iterrows():
        ax.annotate(row['PFAS'], (row['log_value'], row['pred_value']),
                    fontsize=8, color='#444', xytext=(3, 3), textcoords='offset points')
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_title('(a) External logBAF validation', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax)

# (b) TMF cross-scale
ax = axes[1]
tmf_obs  = tmf_ext['log_value'].values
tmf_pred = tmf_ext['pred_value'].values
fin_t    = np.isfinite(tmf_obs) & np.isfinite(tmf_pred)
tmf_obs, tmf_pred = tmf_obs[fin_t], tmf_pred[fin_t]
if len(tmf_obs) >= 5:
    rho_t, pval_t = spearmanr(tmf_obs, tmf_pred)
    colors_t = [NPG[1] if v > 0 else NPG[0] for v in tmf_obs]
    ax.scatter(tmf_pred, tmf_obs, c=colors_t, s=48, alpha=0.78,
               edgecolors='white', lw=0.5, zorder=4, rasterized=True)
    ax.axhline(0, color='#777', lw=1.0, ls='--', alpha=0.6, label='logTMF = 0')
    z = np.polyfit(tmf_pred, tmf_obs, 1)
    x_line = np.linspace(tmf_pred.min(), tmf_pred.max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), color=NPG[1], lw=2.0, ls='-', alpha=0.75)
    ax.set_xlabel('Predicted logBAF (individual scale)', labelpad=5)
    ax.set_ylabel('Observed logTMF (food-web scale)', labelpad=5)
    ax.set_title(f'(b) Cross-scale TMF validation\nn = {len(tmf_obs)}, Spearman ρ = {rho_t:.3f}, p = {pval_t:.2e}',
                 pad=8, fontsize=14, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=NPG[1], label='Biomagnifying (logTMF > 0)'),
                       mpatches.Patch(color=NPG[0], label='Diluting (logTMF < 0)')],
              loc='upper left', frameon=False, fontsize=11)
neat_axes(ax)

# (c) BMF rank correlation
ax = axes[2]
bmf_obs  = bmf_ext['log_value'].values
bmf_pred = bmf_ext['pred_value'].values
fin_b    = np.isfinite(bmf_obs) & np.isfinite(bmf_pred)
bmf_obs, bmf_pred = bmf_obs[fin_b], bmf_pred[fin_b]
if len(bmf_obs) >= 5:
    rho_b, pval_b = spearmanr(bmf_obs, bmf_pred)
    sc = ax.hexbin(bmf_pred, bmf_obs, gridsize=25, cmap='Blues', mincnt=1,
                   linewidths=0.1, bins='log')
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Count (log10)', fontsize=11)
    z_b   = np.polyfit(bmf_pred, bmf_obs, 1)
    x_b   = np.linspace(bmf_pred.min(), bmf_pred.max(), 50)
    ax.plot(x_b, np.polyval(z_b, x_b), color=NPG[1], lw=2.2, ls='--', alpha=0.82)
    ax.set_xlabel('Predicted logBAF (individual scale)', labelpad=5)
    ax.set_ylabel('Observed logBMF (food-web predator/prey)', labelpad=5)
    ax.set_title(f'(c) Food-web BMF rank correlation\nn = {len(bmf_obs)}, Spearman ρ = {rho_b:.3f}, p = {pval_b:.2e}',
                 pad=8, fontsize=14, fontweight='bold')
    ax.text(0.04, 0.94, 'Cross-scale rank validation',
            transform=ax.transAxes, fontsize=10, va='top', color='#555',
            bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.85, pad=3))
neat_axes(ax)

# (d) OECD classification confusion matrix
ax = axes[3]
if len(bcf_baf) > 0:
    obs_b  = (bcf_baf['log_value'].values  >= B_THRESH).astype(int)
    pred_b = (bcf_baf['pred_value'].values >= B_THRESH).astype(int)
    cm = confusion_matrix(obs_b, pred_b)
    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
    for ri in range(2):
        for ci in range(2):
            v = cm[ri, ci]
            ax.text(ci, ri, f'{v}', ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if v > cm.max()/2 else '#333')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted\nnon-B', 'Predicted B'], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Observed\nnon-B', 'Observed B'], fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label('Count', fontsize=11)
    if len(cls_df) > 0:
        b_row = cls_df[cls_df['Threshold'].str.contains('3.3')].iloc[0]
        stats_text = (f'Precision = {b_row["Precision"]:.2f}\n'
                      f'Recall = {b_row["Recall"]:.2f}\n'
                      f'F1 = {b_row["F1"]:.2f}\n'
                      f'Accuracy = {b_row["Accuracy"]:.2f}')
        ax.text(1.38, 0.5, stats_text, transform=ax.transAxes,
                fontsize=11, va='center', ha='left',
                bbox=dict(facecolor='#f8f8f8', edgecolor='#ccc', pad=5))
    ax.set_title(f'(d) OECD B threshold classification\nn = {len(bcf_baf)} external BCF/BAF',
                 pad=8, fontsize=14, fontweight='bold')
    neat_axes(ax)

fig.suptitle('External literature validation of Model 2 bioaccumulation predictions',
             fontsize=15, y=0.98)
fig.tight_layout(pad=2.5, w_pad=3.5, h_pad=3.5)
plt.savefig(OUT / 'Fig6_EV1_v3.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig6 EV1 saved: {(OUT/'Fig6_EV1_v3.png').stat().st_size//1024} KB")

# ═══════════════════════════════════════════════════════════════════════════════
# EV2: coverage profile
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6.5))

ax_a = axes2[0]
tmf_smiles = tmf_ext[['PFAS','pred_value','log_value']].dropna().copy()
tmf_sorted = tmf_smiles.sort_values('pred_value')
colors_tmf = [NPG[1] if v > 0 else NPG[0] for v in tmf_sorted['log_value']]
ax_a.barh(range(len(tmf_sorted)), tmf_sorted['pred_value'],
          color=NPG[6], height=0.7, alpha=0.7, label='Predicted logBAF')
ax_a.scatter(tmf_sorted['log_value'], range(len(tmf_sorted)),
             c=colors_tmf, s=52, zorder=5, edgecolors='white', lw=0.5)
ax_a.axvline(B_THRESH, color=NPG[3], lw=1.3, ls='--', alpha=0.8, label='B threshold (3.3)')
ax_a.axvline(0,        color='#777', lw=0.8, ls=':', alpha=0.6, label='logTMF = 0')
ax_a.set_xlabel('Predicted logBAF / Observed logTMF', labelpad=5)
ax_a.set_title(f'(a) TMF validation: ordered by predicted logBAF\n(n = {len(tmf_sorted)} food-web × PFAS records)',
               pad=8, fontsize=14, fontweight='bold')
ax_a.legend(loc='lower right', frameon=False, fontsize=11)
neat_axes(ax_a)

ax_b = axes2[1]
pfas_agg = ext_valid.groupby('PFAS').agg(
    n_records=('endpoint', 'count'),
    n_BAF=('endpoint', lambda x: (x=='logBAF').sum()),
    n_TMF=('endpoint', lambda x: (x=='logTMF').sum()),
    n_BMF=('endpoint', lambda x: (x=='logBMF').sum()),
    pred_logBAF=('pred_value', 'median'),
).reset_index().sort_values('pred_logBAF', ascending=True)

y_pos = np.arange(len(pfas_agg))
for i, (_, row) in enumerate(pfas_agg.iterrows()):
    c = NPG[1] if row['pred_logBAF'] >= VB_THRESH else (
        NPG[3] if row['pred_logBAF'] >= B_THRESH else NPG[6])
    ax_b.barh(i, row['pred_logBAF'], color=c, height=0.55, alpha=0.82)
for i, (_, row) in enumerate(pfas_agg.iterrows()):
    ax_b.text(row['pred_logBAF'] + 0.06, i,
              f"n={int(row['n_records'])} ({int(row['n_BAF'])}B/{int(row['n_TMF'])}T/{int(row['n_BMF'])}M)",
              va='center', fontsize=8, color='#444')
ax_b.axvline(B_THRESH,  color=NPG[3], lw=1.3, ls='--', alpha=0.8)
ax_b.axvline(VB_THRESH, color=NPG[1], lw=1.3, ls='-.', alpha=0.8)
ax_b.set_yticks(y_pos); ax_b.set_yticklabels(pfas_agg['PFAS'], fontsize=10)
ax_b.set_xlabel('Predicted logBAF (Model 2)', labelpad=5)
ax_b.set_title('(b) Validated PFAS: predicted risk level\n(B=BAF, T=TMF, M=BMF records)',
               pad=8, fontsize=14, fontweight='bold')
ax_b.legend(handles=[mpatches.Patch(color=NPG[1], label='vB (≥4.3)'),
                     mpatches.Patch(color=NPG[3], label='B (≥3.3)'),
                     mpatches.Patch(color=NPG[6], label='non-B (<3.3)')],
            loc='lower right', frameon=False, fontsize=11)
neat_axes(ax_b)

fig2.suptitle('PFAS external validation dataset: coverage and risk profile',
              fontsize=15, y=0.99)
fig2.tight_layout(pad=2.0, w_pad=3.5)
plt.savefig(OUT / 'Fig6_EV2_v3.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig6 EV2 saved: {(OUT/'Fig6_EV2_v3.png').stat().st_size//1024} KB")
