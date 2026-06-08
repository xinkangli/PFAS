"""
Analysis B: Regulatory gap — how many of 127k PFAS exceed B/vB thresholds
Outputs: fig_B_regulatory.png, table_B_regulatory.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import NC_PALETTE, neat_axes, TIMES  # noqa

OUT = ROOT / 'result/discussion'
OUT.mkdir(exist_ok=True)

B_THRESH  = 3.3   # OECD B
VB_THRESH = 4.3   # OECD vB

print('loading predictions ...')
m2 = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m2 = m2[m2['smiles_valid'] == True].copy()
m2['SMILES'] = m2['SMILES'].astype(str).str.strip()

# load pfas_class
master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
master = master[['SMILES','pfas_class']].dropna(subset=['SMILES'])
master['SMILES'] = master['SMILES'].astype(str).str.strip()
master = master.drop_duplicates('SMILES')
m2 = m2.merge(master, on='SMILES', how='left')

# load OECD 2019 list to flag "regulated"
oecd = pd.read_csv(ROOT/'01_structure/OECD_PFAS_list_2019.csv', encoding='latin-1')
oecd_smiles = set(oecd['SMILES'].dropna().astype(str).str.strip().tolist()) \
              if 'SMILES' in oecd.columns else set()
# try CAS
oecd_cas = set(oecd['CAS'].dropna().astype(str).str.strip().tolist()) \
           if 'CAS' in oecd.columns else set()
m2['on_oecd_list'] = m2['SMILES'].isin(oecd_smiles)
print(f'  OECD list match: {m2["on_oecd_list"].sum()}')

# Normalise class
CLASS_MAP = {'PFCA': 'PFCA', 'PFSA': 'PFSA', 'FASA': 'FASA',
             'FTOH': 'FTOH', 'Ether': 'Ether-PFAS'}
def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower():
            return v
    return 'Other'

m2['norm_class'] = m2['pfas_class'].apply(norm_cls)
CLASS_ORDER = ['PFCA', 'PFSA', 'FASA', 'FTOH', 'Ether-PFAS', 'Other']
CLASS_COLORS = {c: NC_PALETTE[i] for i, c in enumerate(CLASS_ORDER)}

# ── bioaccumulation categories ────────────────────────────────────────────────
# Use max(logBAF, logBCF) as conservative estimate
m2['logB_max'] = m2[['pred_logBAF','pred_logBCF']].max(axis=1)
m2['B_cat'] = pd.cut(m2['logB_max'],
                     bins=[-np.inf, B_THRESH, VB_THRESH, np.inf],
                     labels=['non-B (< 3.3)', 'B (3.3-4.3)', 'vB (>= 4.3)'])
m2['B_cat'] = m2['B_cat'].astype(str)

# ── table ─────────────────────────────────────────────────────────────────────
rows = []
for cls in CLASS_ORDER:
    sub = m2[m2['norm_class'] == cls]
    n_total = len(sub)
    if n_total == 0:
        continue
    n_b  = (sub['logB_max'] >= B_THRESH).sum()
    n_vb = (sub['logB_max'] >= VB_THRESH).sum()
    n_unreg_b = ((sub['logB_max'] >= B_THRESH) & (~sub['on_oecd_list'])).sum()
    rows.append({
        'Functional class': cls,
        'Total compounds': n_total,
        'B count (>=3.3)': int(n_b),
        'B fraction (%)': round(100*n_b/n_total, 1),
        'vB count (>=4.3)': int(n_vb),
        'vB fraction (%)': round(100*n_vb/n_total, 1),
        'Unregulated B compounds': int(n_unreg_b),
    })
reg_df = pd.DataFrame(rows)
reg_df.to_csv(OUT/'table_B_regulatory.csv', index=False)
print(reg_df.to_string(index=False))

# ── figure: 2-panel ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Panel 1: stacked bar chart (% by category)
cat_colors = {
    'non-B (< 3.3)':   NC_PALETTE[6],
    'B (3.3-4.3)':   NC_PALETTE[3],
    'vB (>= 4.3)':    NC_PALETTE[1],
}
pivot = (m2.groupby(['norm_class','B_cat'])
           .size()
           .unstack(fill_value=0)
           .reindex(CLASS_ORDER, fill_value=0))
# percent
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
cat_order = ['non-B (< 3.3)', 'B (3.3-4.3)', 'vB (>= 4.3)']
bottom = np.zeros(len(pivot_pct))
x = np.arange(len(CLASS_ORDER))
for cat in cat_order:
    if cat not in pivot_pct.columns:
        continue
    vals = pivot_pct[cat].reindex(CLASS_ORDER, fill_value=0).values
    bars = ax1.bar(x, vals, bottom=bottom, color=cat_colors[cat],
                   label=cat, width=0.65, edgecolor='white', linewidth=0.5)
    # add percentage label inside bar if large enough
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 6:
            ax1.text(xi, b + v/2, f'{v:.0f}%',
                     ha='center', va='center', fontsize=8.5,
                     color='white', fontweight='bold')
    bottom += vals

ax1.set_xticks(x)
ax1.set_xticklabels(CLASS_ORDER, rotation=30, ha='right', fontsize=10)
ax1.set_ylabel('Compounds (%)', labelpad=6)
ax1.set_title('(a) (a) Bioaccumulation risk by functional class', pad=10)
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right', frameon=False, fontsize=10,
           bbox_to_anchor=(1.0, 1.0))
neat_axes(ax1)

# Panel 2: density scatter — pred_logBCF vs unc_logBCF
m2_valid = m2[(m2['pred_logBCF'].notna()) & (m2['unc_logBCF'].notna())].copy()
# clip extremes for display
xclip = m2_valid['pred_logBCF'].clip(-1, 10)
yclip = m2_valid['unc_logBCF'].clip(0, 2)

sc = ax2.hexbin(xclip, yclip, gridsize=60, cmap='YlOrRd',
                mincnt=1, linewidths=0.1, bins='log')
cb = fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.03)
cb.set_label('Count (log10)', fontsize=11)

# threshold lines
ax2.axvline(B_THRESH,  color=NC_PALETTE[3], lw=1.5, ls='--', label='B threshold (3.3)')
ax2.axvline(VB_THRESH, color=NC_PALETTE[1], lw=1.5, ls='-.', label='vB threshold (4.3)')
ax2.axhline(0.5, color=NC_PALETTE[6], lw=1.2, ls=':', label='High uncertainty (0.5)')

ax2.set_xlabel('Predicted logBCF', labelpad=6)
ax2.set_ylabel('Uncertainty (MC-Dropout σ)', labelpad=6)
ax2.set_title('(b) (b) Predicted logBCF vs uncertainty', pad=10)
ax2.legend(loc='upper left', frameon=False, fontsize=9.5)
neat_axes(ax2)

fig.tight_layout(pad=2.5, w_pad=3.5)
plt.savefig(OUT/'fig_B_regulatory.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig_B_regulatory.png saved')

# summary stats
total_valid = (m2['smiles_valid'] == True).sum() if 'smiles_valid' in m2.columns else len(m2)
n_B  = (m2['logB_max'] >= B_THRESH).sum()
n_vB = (m2['logB_max'] >= VB_THRESH).sum()
n_unreg_B = ((m2['logB_max'] >= B_THRESH) & (~m2['on_oecd_list'])).sum()
print(f'\nSummary: {len(m2)} valid predictions')
print(f'  B (≥3.3):  {n_B} ({100*n_B/len(m2):.1f}%)')
print(f'  vB (≥4.3): {n_vB} ({100*n_vB/len(m2):.1f}%)')
print(f'  Unregulated B: {n_unreg_B}')
