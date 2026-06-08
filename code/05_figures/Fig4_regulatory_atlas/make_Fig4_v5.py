"""
Fig4 v5: Regulatory risk atlas
Changes vs v4:
  - Panel (b): legend moved outside to the right of the panel.
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
from matplotlib.lines import Line2D

ROOT = Path('/DATA/lxk/lxkzero/claude/qd/data')
sys.path.insert(0, str(ROOT/'result'))
from _style import neat_axes, TIMES, PERSIST_C

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

OUT    = ROOT / 'result/discussion2/Fig4_regulatory_atlas'
B_THRESH  = 3.3; VB_THRESH = 4.3; KOW_THRESH = 5.0

# Load predictions
m1 = pd.read_csv(ROOT/'model_data/model1_out/model1_predictions.csv')
m1 = m1[m1['smiles_valid'] == True][['SMILES', 'pred_logKow']].copy()
m2 = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m2 = m2[m2['smiles_valid'] == True].copy()
m2['SMILES'] = m2['SMILES'].astype(str).str.strip()
m1['SMILES'] = m1['SMILES'].astype(str).str.strip()
merged = m1.merge(m2, on='SMILES', how='inner')

pers  = pd.read_csv(ROOT/'05_persistence_transformation/persistence.csv')
comp  = pd.read_csv(ROOT/'05_persistence_transformation/compounds.csv', on_bad_lines='skip')
pers_m = pers.merge(comp[['compound_id', 'cas']], on='compound_id', how='left')
pers_by_cas = pers_m.groupby('cas')['persistence_label'].first().reset_index()
master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
mast2  = master[['SMILES', 'CAS', 'pfas_class']].dropna(subset=['SMILES']).copy()
mast2['SMILES'] = mast2['SMILES'].astype(str).str.strip()
mast2['cas']    = mast2['CAS'].astype(str).str.upper().str.strip()
pers_by_cas['cas'] = pers_by_cas['cas'].astype(str).str.upper().str.strip()
pers_smi = mast2.merge(pers_by_cas, on='cas', how='inner')[['SMILES', 'persistence_label']].drop_duplicates('SMILES')
merged   = merged.merge(pers_smi, on='SMILES', how='left')

CLASS_MAP = {'PFCA':'PFCA','PFSA':'PFSA','FASA':'FASA','FTOH':'FTOH','Ether':'Ether-PFAS'}
def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower(): return v
    return 'Other'

merged = merged.merge(mast2[['SMILES', 'pfas_class']].drop_duplicates('SMILES'), on='SMILES', how='left')
merged['norm_class'] = merged['pfas_class'].apply(norm_cls)

kow_q = (merged['pred_logKow'].quantile(0.01), merged['pred_logKow'].quantile(0.99))
baf_q = (merged['pred_logBAF'].quantile(0.01), merged['pred_logBAF'].quantile(0.99))
df = merged[(merged['pred_logKow'].between(*kow_q)) & (merged['pred_logBAF'].between(*baf_q))].copy()

oecd = pd.read_csv(ROOT/'01_structure/OECD_PFAS_list_2019.csv', encoding='latin-1')
oecd_smiles = set(oecd['SMILES'].dropna().astype(str).str.strip()) if 'SMILES' in oecd.columns else set()
m2['on_oecd']   = m2['SMILES'].isin(oecd_smiles)
m2['logB_max']  = m2[['pred_logBAF', 'pred_logBCF']].max(axis=1)
m2 = m2.merge(mast2[['SMILES', 'pfas_class']].drop_duplicates('SMILES'), on='SMILES', how='left')
m2['norm_class'] = m2['pfas_class'].apply(norm_cls)

ext_valid    = pd.read_csv(ROOT/'result/waibuyanzheng/step4_validation_set.csv')
ext_smiles_baf = set(ext_valid[ext_valid['endpoint'].isin(['logBAF','logBCF'])]['SMILES'].dropna().astype(str))

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

# ── (a) Main risk landscape ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0:2])
hb = ax_a.hexbin(df['pred_logKow'], df['pred_logBAF'], gridsize=65, cmap='Blues',
                 mincnt=1, linewidths=0.1, bins='log', alpha=0.85)
fig.colorbar(hb, ax=ax_a, fraction=0.04, pad=0.02).set_label('Count (log10)', fontsize=11)

for label in ['vP', 'P', 'not-P_self', 'not-P']:
    sub = df[df['persistence_label'] == label]
    if len(sub) == 0: continue
    c  = PERSIST_C.get(label, NPG[4])
    ms = 65 if label == 'vP' else 26
    ax_a.scatter(sub['pred_logKow'], sub['pred_logBAF'], c=c, s=ms, alpha=0.9,
                 edgecolors='white', lw=0.4, zorder=5, label=f'Persistence={label}')

ext_on_land = df[df['SMILES'].isin(ext_smiles_baf)]
if len(ext_on_land) > 0:
    ax_a.scatter(ext_on_land['pred_logKow'], ext_on_land['pred_logBAF'],
                 c='black', s=85, marker='*', zorder=7, edgecolors='white',
                 lw=0.5, label=f'Ext. validated ({len(ext_on_land)})')

rect = mpatches.FancyBboxPatch((KOW_THRESH, B_THRESH),
                               df['pred_logKow'].max() - KOW_THRESH + 0.2,
                               df['pred_logBAF'].max() - B_THRESH + 0.2,
                               boxstyle='round,pad=0.1', linewidth=2,
                               edgecolor=NPG[1], facecolor='none', linestyle='--', zorder=8)
ax_a.add_patch(rect)
ax_a.text(KOW_THRESH + 0.2, B_THRESH + 0.15,
          'Triple-threat zone\n(logKow>5, logBAF>3.3, vP)',
          fontsize=9.5, color=NPG[1], va='bottom',
          bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.88, pad=3))
ax_a.axhline(B_THRESH,   color='#555', lw=1.2, ls='--', alpha=0.8)
ax_a.axvline(KOW_THRESH, color='#555', lw=1.2, ls=':',  alpha=0.8)
ax_a.set_xlabel('Predicted logKow', labelpad=5)
ax_a.set_ylabel('Predicted logBAF', labelpad=5)
ax_a.set_title('(a) PFAS multi-property risk atlas (n = 117k)\n'
               '(logKow × logBAF; colored by persistence; stars = externally validated)',
               pad=8, fontsize=14, fontweight='bold')
handles_a = [mpatches.Patch(color=PERSIST_C[l], label=l)
             for l in ['vP','P','not-P_self','not-P'] if l in df['persistence_label'].values]
handles_a += [mpatches.Patch(color='black', label='Externally validated'),
              Line2D([0],[0], color='#555', ls='--', lw=1.2, label='B threshold (3.3)'),
              Line2D([0],[0], color='#555', ls=':', lw=1.2, label='logKow = 5.0')]
ax_a.legend(handles=handles_a, loc='lower right', frameon=True, framealpha=0.9,
            fontsize=10, edgecolor='#ccc')
neat_axes(ax_a)
ax_a.text(-0.04, 1.04, 'a', transform=ax_a.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (b) B/vB stacked bar ───────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
CLS_ORDER  = ['PFCA','PFSA','FASA','FTOH','Ether-PFAS']
cat_colors = {'non-B': NPG[6], 'B': NPG[3], 'vB': NPG[1]}
for ci, cls in enumerate(CLS_ORDER):
    sub = m2[m2['norm_class'] == cls]
    n   = len(sub)
    if n == 0: continue
    n_vb = int((sub['logB_max'] >= VB_THRESH).sum())
    n_b  = int(((sub['logB_max'] >= B_THRESH) & (sub['logB_max'] < VB_THRESH)).sum())
    n_nb = n - n_b - n_vb
    # Use integer counts to derive widths so segments tile exactly to 100%
    counts = [n_nb, n_b, n_vb]
    colors = [cat_colors['non-B'], cat_colors['B'], cat_colors['vB']]
    labels = ['non-B', 'B', 'vB']
    bot = 0.0
    for cnt, pc, lbl in zip(counts, colors, labels):
        pv = 100.0 * cnt / n
        if cnt == 0:
            continue
        ax_b.barh(ci, pv, left=bot, color=pc, height=0.65,
                  edgecolor='white', lw=1.5)
        if pv > 8:
            ax_b.text(bot + pv/2, ci, f'{pv:.0f}%', ha='center', va='center',
                      fontsize=9.5, color='white', fontweight='bold')
        bot += pv
    # Force last segment to land exactly at 100 to close any fp gap
    if bot < 100.0:
        ax_b.barh(ci, 100.0 - bot, left=bot, color=colors[-1],
                  height=0.65, edgecolor='white', lw=1.5)
ax_b.set_yticks(range(len(CLS_ORDER))); ax_b.set_yticklabels(CLS_ORDER, fontsize=11)
ax_b.set_xlabel('Compounds (%)')
ax_b.set_title('(b) B/vB distribution by class', pad=8, fontsize=14, fontweight='bold')
handles_b = [mpatches.Patch(color=cat_colors[k], label=k) for k in ['vB','B','non-B']]
ax_b.legend(handles=handles_b, frameon=True, framealpha=0.9, fontsize=10,
            loc='upper left', bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0, edgecolor='#ccc')
neat_axes(ax_b)
ax_b.text(-0.18, 1.04, 'b', transform=ax_b.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (c) Unregulated B-class counts ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
unreg_data = {'PFCA':213,'PFSA':108,'FASA':212,'FTOH':126,'Ether-PFAS':75,'Other':6147}
x_c     = np.arange(len(unreg_data))
colors_c = [NPG[i] for i in range(len(unreg_data))]
ax_c.bar(x_c, list(unreg_data.values()), color=colors_c, width=0.6, edgecolor='white', lw=0.5)
for xi, v in zip(x_c, unreg_data.values()):
    ax_c.text(xi, v + 35, f'{v:,}', ha='center', va='bottom', fontsize=10)
ax_c.set_xticks(x_c); ax_c.set_xticklabels(list(unreg_data.keys()), rotation=32, ha='right', fontsize=11)
ax_c.set_ylabel('Unregulated B-class compounds')
ax_c.set_title('(c) Regulatory gap: unregulated B-class\n(total: 6,881 not on OECD 2019 list)',
               pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_c)
ax_c.text(-0.14, 1.04, 'c', transform=ax_c.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (d) Uncertainty distribution high-risk vs low-risk — FIXED overlaps ───────
ax_d = fig.add_subplot(gs[1, 1])
high_risk = df[(df['pred_logKow'] >= KOW_THRESH) & (df['pred_logBAF'] >= B_THRESH)]
low_risk  = df[(df['pred_logKow'] <  KOW_THRESH) | (df['pred_logBAF'] <  B_THRESH)]

unc_col_key = 'unc_logBAF' if 'unc_logBAF' in df.columns else None
if unc_col_key:
    hi_unc = high_risk[unc_col_key].dropna().clip(0, 1.5)
    lo_unc = low_risk[unc_col_key].dropna().clip(0, 1.5)
    ax_d.hist(lo_unc, bins=40, density=True, alpha=0.60, color=NPG[0],
              label=f'Low-risk zone (n={len(lo_unc):,})', histtype='stepfilled')
    ax_d.hist(hi_unc, bins=40, density=True, alpha=0.60, color=NPG[1],
              label=f'High-risk zone (n={len(hi_unc):,})', histtype='stepfilled')
    # *** FIX: threshold line label via annotate with arrow off data area ***
    ax_d.axvline(0.5, color='#333', lw=1.5, ls='--', zorder=5)
    ax_d.annotate('σ = 0.5\n(high unc.)', xy=(0.5, ax_d.get_ylim()[1] if ax_d.get_ylim()[1] > 0 else 1),
                  xytext=(0.62, 0.88), textcoords='axes fraction',
                  fontsize=10.5, color='#333', va='top',
                  arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax_d.set_xlabel('MC-Dropout σ (logBAF)', labelpad=5)
    ax_d.set_ylabel('Density')
    # *** FIX: legend at lower-right where density → 0 (away from peak & black line) ***
    ax_d.legend(frameon=True, framealpha=0.94, fontsize=11,
                loc='upper center', bbox_to_anchor=(0.5, -0.22),
                ncol=1, edgecolor='#ccc')
else:
    ax_d.text(0.5, 0.5, 'unc_logBAF not in df', ha='center', va='center',
              transform=ax_d.transAxes)
ax_d.set_title('(d) Uncertainty distribution:\nhigh-risk vs low-risk zone', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_d)
ax_d.text(-0.14, 1.04, 'd', transform=ax_d.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (e) Triple-threat table ────────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
tt = pd.read_csv(ROOT/'result/discussion/table_C_triplethreat.csv')
tt['class'] = tt['pfas_class'].fillna('').apply(norm_cls)
ax_e.axis('off')
col_labels = ['Class', 'logKow', 'logBAF', 'logBCF', 'sigma']
col_data   = [[norm_cls(row['pfas_class']),
               f"{row['pred_logKow']:.2f}",
               f"{row['pred_logBAF']:.2f}",
               f"{row['pred_logBCF']:.2f}",
               f"{row['unc_logBAF']:.2f}"]
              for _, row in tt.iterrows()]
table = ax_e.table(cellText=col_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10)
table.scale(1.15, 1.55)
for (ri, ci), cell in table._cells.items():
    cell.set_edgecolor('#cccccc')
    if ri == 0:
        cell.set_facecolor('#3C5488'); cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')
    elif ri % 2 == 1:
        cell.set_facecolor('#f0f4fa')
    else:
        cell.set_facecolor('white')
ax_e.set_title('(e) Triple-threat PFAS\n(logKow≥5, logBAF≥3.3, persistence=vP)',
               pad=8, fontsize=14, fontweight='bold')
ax_e.text(-0.08, 1.04, 'e', transform=ax_e.transAxes, fontsize=17, fontweight='bold', va='top')

fig.suptitle('PFAS regulatory risk atlas: from prediction to policy action\n'
             '(117,018 valid predictions; 6,881 unregulated high-bioaccumulation compounds identified)',
             fontsize=14, y=0.99)
fig.savefig(OUT / 'Fig4_regulatory_atlas_v5.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig4_v5 saved: {(OUT/'Fig4_regulatory_atlas_v5.png').stat().st_size//1024} KB")
