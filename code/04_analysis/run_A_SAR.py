"""
Analysis A: PFAS bioaccumulation structure-activity relationship (SAR)
Chain length × functional class × predicted logBAF/logBCF
All labels in English for NC submission.
Outputs: fig_A_SAR.png, table_A_SAR.csv
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import NC_PALETTE, neat_axes, TIMES  # noqa

OUT = ROOT / 'result/discussion'
OUT.mkdir(exist_ok=True)

# ── chain length estimation from SMILES ──────────────────────────────────────
def estimate_chain_len(smi: str) -> int | None:
    """Count longest C-F chain via CF2/CF3 groups."""
    s = str(smi)
    # count C(F)(F) fragments as proxy for CF2 units
    cf2 = s.count('C(F)(F)')   # includes CF2 and CF3
    cf3 = s.count('C(F)(F)(F)')
    # net CF2 = total CF2+CF3 occurrences minus CF3 occurrences (already counted in CF2)
    total = cf2  # CF3 is a subset of CF2 pattern
    return total if total >= 3 else None

# ── load data ─────────────────────────────────────────────────────────────────
print('loading Model2 training data ...')
df2 = pd.read_excel(ROOT/'model_data/Model2_Bioaccumulation.xlsx',
                    sheet_name='X_Y_joined', engine='openpyxl')
df2['log_value'] = pd.to_numeric(df2['log_value'], errors='coerce')
df2['chain_len']  = pd.to_numeric(df2['chain_len'], errors='coerce')
df2 = df2[df2['log_value'].notna()].copy()
ep_map = {'BAF': 'logBAF', 'BCF': 'logBCF', 'BCFD': 'logBCFD'}
df2['endpoint'] = df2['endpoint_type'].map(ep_map).fillna(df2['endpoint_type'])

print('loading predictions + pfas_class ...')
m2p = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m2p = m2p[m2p['smiles_valid'] == True].copy()
m2p['SMILES'] = m2p['SMILES'].astype(str).str.strip()

master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
master = master[['SMILES','pfas_class']].dropna(subset=['SMILES']).copy()
master['SMILES'] = master['SMILES'].astype(str).str.strip()
master = master.dropna(subset=['pfas_class']).drop_duplicates('SMILES')
m2p = m2p.merge(master, on='SMILES', how='left')

# Estimate chain length from SMILES for all pool compounds
print('estimating chain lengths from SMILES ...')
m2p['chain_len'] = m2p['SMILES'].apply(estimate_chain_len)
m2p_cl = m2p[m2p['chain_len'].notna() & m2p['pfas_class'].notna()].copy()
m2p_cl['chain_len'] = m2p_cl['chain_len'].astype(int)
m2p_cl = m2p_cl[m2p_cl['chain_len'].between(3, 18)]
print(f'  pool predictions with chain_len: {len(m2p_cl)}')

# Normalise class
CLASS_MAP = {'PFCA':'PFCA','PFSA':'PFSA','FASA':'FASA',
             'FTOH':'FTOH','Ether':'Ether-PFAS'}
def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower():
            return v
    return None

m2p_cl['norm_class'] = m2p_cl['pfas_class'].apply(norm_cls)
pool_sar = m2p_cl[m2p_cl['norm_class'].notna()].copy()
print(f'  class-labelled with chain_len: {len(pool_sar)}')
print(pool_sar['norm_class'].value_counts())

# ── SAR stats from pool predictions ──────────────────────────────────────────
def sar_stats(df, val_col, class_col='norm_class'):
    return (df.groupby([class_col,'chain_len'])[val_col]
              .agg(median='median',
                   q25=lambda x: x.quantile(0.25),
                   q75=lambda x: x.quantile(0.75),
                   n='count')
              .reset_index())

pred_baf = sar_stats(pool_sar, 'pred_logBAF')
pred_bcf = sar_stats(pool_sar, 'pred_logBCF')

# ── SAR from training data (experimental) ────────────────────────────────────
if 'family' in df2.columns:
    df2['func_class'] = df2['family'].astype(str)
else:
    df2['func_class'] = df2.get('pfas_class', pd.Series('Other', index=df2.index)).astype(str)
df2['norm_class'] = df2['func_class'].apply(norm_cls)
exp_baf = df2[(df2['endpoint']=='logBAF') & df2['norm_class'].notna() & df2['chain_len'].notna()]
exp_bcf = df2[(df2['endpoint']=='logBCF') & df2['norm_class'].notna() & df2['chain_len'].notna()]

# ── save SAR table ────────────────────────────────────────────────────────────
CLASS_ORDER = ['PFCA','PFSA','FASA','FTOH','Ether-PFAS']
rows = []
for cls in CLASS_ORDER:
    b = pred_baf[pred_baf['norm_class']==cls].sort_values('chain_len')
    c = pred_bcf[pred_bcf['norm_class']==cls].sort_values('chain_len')
    for cl in sorted(set(b['chain_len'].tolist() + c['chain_len'].tolist())):
        row = {'class': cls, 'chain_len': int(cl)}
        rb = b[b['chain_len']==cl]
        rc = c[c['chain_len']==cl]
        if len(rb): row.update({f'pred_logBAF_{k}': round(float(rb.iloc[0][k]),3)
                                 for k in ['median','q25','q75','n']})
        if len(rc): row.update({f'pred_logBCF_{k}': round(float(rc.iloc[0][k]),3)
                                 for k in ['median','q25','q75','n']})
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'table_A_SAR.csv', index=False)
print(f'SAR table rows: {len(rows)}')

# ── figure ────────────────────────────────────────────────────────────────────
CLASS_COLORS = {cls: NC_PALETTE[i] for i, cls in enumerate(CLASS_ORDER)}

# Only plot classes with >=3 chain length points
plot_classes = [c for c in CLASS_ORDER
                if len(pred_baf[pred_baf['norm_class']==c]) >= 3]
n_cls = len(plot_classes)
ncols = min(n_cls, 3)
nrows = (n_cls + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2*nrows),
                          squeeze=False)
axes_flat = axes.flatten()

for idx, cls in enumerate(plot_classes):
    ax = axes_flat[idx]
    color = CLASS_COLORS[cls]

    # Predicted BAF curve
    pb = pred_baf[pred_baf['norm_class']==cls].sort_values('chain_len')
    if len(pb) >= 2:
        ax.plot(pb['chain_len'], pb['median'], color=color, lw=2.2,
                zorder=3, label='Predicted logBAF (median)')
        ax.fill_between(pb['chain_len'], pb['q25'], pb['q75'],
                        color=color, alpha=0.18, zorder=2)
    # Predicted BCF curve
    pc = pred_bcf[pred_bcf['norm_class']==cls].sort_values('chain_len')
    if len(pc) >= 2:
        ax.plot(pc['chain_len'], pc['median'], color=color, lw=2.0,
                ls='--', alpha=0.75, zorder=3, label='Predicted logBCF (median)')

    # Experimental scatter
    eb = exp_baf[exp_baf['norm_class']==cls].groupby('chain_len')['log_value'].median().reset_index()
    if len(eb):
        ax.scatter(eb['chain_len'], eb['log_value'], color=color, s=55,
                   zorder=5, edgecolors='white', linewidths=0.7,
                   label='Observed logBAF (median)')
    ec = exp_bcf[exp_bcf['norm_class']==cls].groupby('chain_len')['log_value'].median().reset_index()
    if len(ec):
        ax.scatter(ec['chain_len'], ec['log_value'], color=color, s=55,
                   marker='s', zorder=5, edgecolors='white', linewidths=0.7,
                   label='Observed logBCF (median)')

    # Regulatory thresholds
    ax.axhline(3.3, color='#555555', lw=1.0, ls=':', alpha=0.8, zorder=1)
    ax.axhline(4.3, color='#555555', lw=1.0, ls='-.', alpha=0.8, zorder=1)
    # place labels at right edge, just outside the data to avoid overlap
    xlim_r = ax.get_xlim()[1] if ax.get_xlim()[1] > 5 else 18.3
    ax.text(xlim_r, 3.35, 'B', va='bottom', ha='left', fontsize=8.5, color='#555555')
    ax.text(xlim_r, 4.35, 'vB', va='bottom', ha='left', fontsize=8.5, color='#555555')

    ax.set_xlim(2.5, 18.5)
    ax.set_xlabel('Perfluorocarbon chain length (C)', labelpad=5)
    ax.set_ylabel('log BAF / BCF', labelpad=5)
    ax.set_title(cls, pad=8)
    ax.set_xticks(range(3, 19, 2))
    neat_axes(ax)
    if idx == 0:
        ax.legend(loc='upper left', frameon=False, fontsize=8.5,
                  handlelength=1.4, borderpad=0.2)

# hide unused axes
for i in range(len(plot_classes), len(axes_flat)):
    axes_flat[i].set_visible(False)

# shared legend for thresholds
thresh_lines = [
    plt.Line2D([0],[0], color='#555555', ls=':', lw=1.0, label='B threshold (logBCF=3.3)'),
    plt.Line2D([0],[0], color='#555555', ls='-.', lw=1.0, label='vB threshold (logBCF=4.3)'),
]
fig.legend(handles=thresh_lines, loc='lower center', ncol=2,
           frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle('PFAS Bioaccumulation SAR: Chain Length × Functional Class',
             fontsize=13, y=1.02)
fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)
plt.savefig(OUT/'fig_A_SAR.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig_A_SAR.png saved')
