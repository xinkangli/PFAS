"""
Analysis C: Multi-property risk landscape
logKow × logBAF density map for all 127k PFAS, with persistence overlay
Outputs: fig_C_landscape.png, table_C_triplethreat.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import NC_PALETTE, PERSIST_C, neat_axes, TIMES  # noqa

OUT = ROOT / 'result/discussion'
OUT.mkdir(exist_ok=True)

print('loading predictions ...')
m1 = pd.read_csv(ROOT/'model_data/model1_out/model1_predictions.csv')
m2 = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m1 = m1[m1['smiles_valid'] == True][['SMILES','pred_logKow','unc_logKow']].copy()
m2 = m2[m2['smiles_valid'] == True][['SMILES','pred_logBAF','pred_logBCF','unc_logBAF']].copy()
merged = m1.merge(m2, on='SMILES', how='inner')
print(f'  merged: {len(merged)}')

# clip to sensible range (99th percentile)
kow_q99 = merged['pred_logKow'].quantile(0.99)
baf_q99 = merged['pred_logBAF'].quantile(0.99)
kow_q01 = merged['pred_logKow'].quantile(0.01)
baf_q01 = merged['pred_logBAF'].quantile(0.01)
df = merged[(merged['pred_logKow'].between(kow_q01, kow_q99)) &
            (merged['pred_logBAF'].between(baf_q01, baf_q99))].copy()
print(f'  after clip: {len(df)}')

# persistence labels
print('loading persistence labels ...')
pers = pd.read_csv(ROOT/'05_persistence_transformation/persistence.csv')
comp = pd.read_csv(ROOT/'05_persistence_transformation/compounds.csv', on_bad_lines='skip')
pers_merge = pers.merge(comp[['compound_id','cas']], on='compound_id', how='left')
pers_by_cas = pers_merge.groupby('cas')['persistence_label'].first().reset_index()
master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
mast2 = master[['SMILES','CAS','pfas_class']].dropna(subset=['SMILES']).copy()
mast2['cas'] = mast2['CAS'].astype(str).str.upper().str.strip()
pers_by_cas['cas'] = pers_by_cas['cas'].astype(str).str.upper().str.strip()
pers_smi = mast2.merge(pers_by_cas, on='cas', how='inner')[
    ['SMILES','persistence_label']].drop_duplicates('SMILES')
df = df.merge(pers_smi, on='SMILES', how='left')
df = df.merge(mast2[['SMILES','pfas_class']].drop_duplicates('SMILES'),
              on='SMILES', how='left')
print(f'  persistence labelled: {df["persistence_label"].notna().sum()}')

# ── triple-threat: logKow>5, logBAF>3.3, vP ─────────────────────────────────
B_THRESH  = 3.3
KOW_THRESH = 5.0
triple = df[
    (df['pred_logKow']  >= KOW_THRESH) &
    (df['pred_logBAF']  >= B_THRESH) &
    (df['persistence_label'] == 'vP')
].copy()
print(f'  triple-threat: {len(triple)}')
tt_out = triple[['SMILES','pred_logKow','pred_logBAF','pred_logBCF',
                  'unc_logBAF','pfas_class','persistence_label']].copy()
for c in ['pred_logKow','pred_logBAF','pred_logBCF','unc_logBAF']:
    tt_out[c] = tt_out[c].round(3)
tt_out.to_csv(OUT/'table_C_triplethreat.csv', index=False)
print(f'triple-threat table: {len(tt_out)} compounds')

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7))
gs = GridSpec(1, 2, width_ratios=[3, 1.1], wspace=0.35)
ax_main = fig.add_subplot(gs[0])
ax_side = fig.add_subplot(gs[1])

# Main panel: hexbin density
hb = ax_main.hexbin(df['pred_logKow'], df['pred_logBAF'],
                    gridsize=70, cmap='Blues', mincnt=1,
                    linewidths=0.1, bins='log', alpha=0.85)
cb = fig.colorbar(hb, ax=ax_main, fraction=0.04, pad=0.03)
cb.set_label('Count (log10)', fontsize=11)

# Overlay persistence-labelled compounds
for label in ['vP', 'P', 'not-P_self', 'not-P']:
    sub = df[df['persistence_label'] == label]
    if len(sub) == 0:
        continue
    c = PERSIST_C.get(label, NC_PALETTE[4])
    ms = 55 if label == 'vP' else 28
    ax_main.scatter(sub['pred_logKow'], sub['pred_logBAF'],
                    c=c, s=ms, alpha=0.9, edgecolors='white',
                    linewidths=0.4, zorder=5, label=label)

# Threshold lines
ax_main.axhline(B_THRESH, color='#474747', lw=1.2, ls='--', alpha=0.8)
ax_main.axvline(KOW_THRESH, color='#474747', lw=1.2, ls=':', alpha=0.8)
# annotate quadrants (offset from axes to avoid overlap)
xmin, xmax = ax_main.get_xlim()
ymin, ymax = ax_main.get_ylim()

# safe text positions: upper-right / lower-left corners well inside axes
ax_main.text(KOW_THRESH + 0.3, B_THRESH + 0.2,
             'High-risk zone\n(logKow>5, logBAF>3.3)',
             fontsize=9, color='#333333', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#cccccc', alpha=0.85))

ax_main.set_xlabel('Predicted logKow', labelpad=6)
ax_main.set_ylabel('Predicted logBAF', labelpad=6)
ax_main.set_title('(a) (a) PFAS risk landscape (n=117k)', pad=10)

# Legend for persistence (outside plot area)
handles = [mpatches.Patch(color=PERSIST_C[l], label=l)
           for l in ['vP','P','not-P_self','not-P'] if l in df['persistence_label'].values]
handles += [
    plt.Line2D([0],[0], color='#474747', ls='--', lw=1.2, label='B threshold (logBAF=3.3)'),
    plt.Line2D([0],[0], color='#474747', ls=':', lw=1.2, label='logKow=5.0'),
]
ax_main.legend(handles=handles, loc='lower right', frameon=True,
               fontsize=9.5, framealpha=0.9, edgecolor='#cccccc',
               borderpad=0.5, handlelength=1.4)
neat_axes(ax_main)

# Side panel: stacked horizontal bar — % B/vB by class
CLASS_ORDER = ['PFCA','PFSA','FASA','FTOH','Ether-PFAS','Other']
CLASS_MAP = {'PFCA':'PFCA','PFSA':'PFSA','FASA':'FASA',
             'FTOH':'FTOH','Ether':'Ether-PFAS'}
def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower():
            return v
    return 'Other'
df['norm_class'] = df['pfas_class'].apply(norm_cls)

# B fraction by class
side_rows = []
for cls in CLASS_ORDER:
    sub = df[df['norm_class'] == cls]
    n = len(sub)
    if n == 0:
        continue
    side_rows.append({
        'class': cls,
        'n': n,
        'pct_B': 100*(sub['pred_logBAF'] >= B_THRESH).sum()/n,
        'pct_vB': 100*(sub['pred_logBAF'] >= 4.3).sum()/n,
    })
side_df = pd.DataFrame(side_rows)

y = np.arange(len(side_df))
ax_side.barh(y, side_df['pct_B'], color=NC_PALETTE[3], label='B class (>=3.3)',
             height=0.55, edgecolor='white')
ax_side.barh(y, side_df['pct_vB'], color=NC_PALETTE[1], label='vB class (>=4.3)',
             height=0.55, edgecolor='white')
ax_side.set_yticks(y)
ax_side.set_yticklabels(side_df['class'], fontsize=10)
ax_side.set_xlabel('Fraction exceeding threshold (%)', labelpad=5)
ax_side.set_title('(b) (b) B/vB fraction by class', pad=10)
ax_side.legend(loc='lower right', frameon=False, fontsize=9)
neat_axes(ax_side)

# Annotate n values
for i, row in side_df.iterrows():
    ax_side.text(side_df['pct_B'].max() + 2, list(y)[list(side_df['class']).index(row['class'])],
                 f"n={row['n']:,}", va='center', fontsize=8.5, color='#474747')

fig.tight_layout(pad=2.0)
plt.savefig(OUT/'fig_C_landscape.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig_C_landscape.png saved')
