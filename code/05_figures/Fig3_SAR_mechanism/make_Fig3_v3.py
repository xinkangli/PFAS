"""
Fig3 v3: Mechanistic SAR — chain length × headgroup × uncertainty
Changes vs v1:
  - NPG palette, +1 font sizes
  - Panel (a): threshold labels moved to right margin, not on data area
  - Panel (e): title not overlapping colorbar tick labels
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

OUT = ROOT / 'result/discussion2/Fig3_SAR_mechanism'

sar   = pd.read_csv(ROOT/'result/discussion/table_A_SAR.csv')
train = pd.read_excel(ROOT/'model_data/Model2_Bioaccumulation.xlsx',
                      sheet_name='X_Y_joined', engine='openpyxl')
train['log_value']  = pd.to_numeric(train['log_value'],  errors='coerce')
train['chain_len']  = pd.to_numeric(train['chain_len'],  errors='coerce')

master = pd.read_csv(ROOT/'03_transport/PFAS_master_structure.csv', low_memory=False)
master = master[['SMILES', 'pfas_class']].dropna(subset=['SMILES']).drop_duplicates('SMILES')
master['SMILES'] = master['SMILES'].astype(str).str.strip()
CLASS_MAP = {'PFCA':'PFCA','PFSA':'PFSA','FASA':'FASA','FTOH':'FTOH','Ether':'Ether-PFAS'}

def norm_cls(c):
    c = str(c)
    for k, v in CLASS_MAP.items():
        if k.lower() in c.lower(): return v
    return 'Other'

def estimate_chain_len(smi):
    return str(smi).count('C(F)(F)')

master['norm_class'] = master['pfas_class'].apply(norm_cls)

m2p = pd.read_csv(ROOT/'model_data/model2_out/model2_predictions.csv')
m2p = m2p[m2p['smiles_valid'] == True].copy()
m2p['SMILES'] = m2p['SMILES'].astype(str).str.strip()
m2p = m2p.merge(master[['SMILES', 'norm_class']], on='SMILES', how='left')
m2p['chain_len'] = m2p['SMILES'].apply(estimate_chain_len)
m2p_cl = m2p[
    m2p['chain_len'].between(3, 18) &
    m2p['norm_class'].notna() &
    m2p['norm_class'].isin(['PFCA','PFSA','FASA','FTOH','Ether-PFAS'])
].copy()

B_THRESH = 3.3; VB_THRESH = 4.3
CLS_ORDER  = ['PFCA','PFSA','FASA','FTOH','Ether-PFAS']
CLS_COLORS = {c: NPG[i] for i, c in enumerate(CLS_ORDER)}

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.40)

# ── (a) SAR curves ─────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0:2])
plot_cls = ['PFCA', 'PFSA', 'FTOH']
for cls in plot_cls:
    sub = sar[sar['class'] == cls].sort_values('chain_len')
    if len(sub) < 3: continue
    c = CLS_COLORS.get(cls, NPG[5])
    ax_a.plot(sub['chain_len'], sub['pred_logBAF_median'], color=c, lw=2.8, zorder=4,
              label=f'{cls} (predicted median logBAF)')
    ax_a.fill_between(sub['chain_len'], sub['pred_logBAF_q25'], sub['pred_logBAF_q75'],
                      color=c, alpha=0.15, zorder=2)
    tr_sub = train[
        train.get('pfas_class', pd.Series([''] * len(train))).astype(str).str.contains(
            cls, case=False, na=False) &
        (train.get('endpoint_type', pd.Series([''] * len(train))).astype(str) == 'BAF') &
        train['chain_len'].notna()
    ]
    if len(tr_sub) > 0:
        exp_med = tr_sub.groupby('chain_len')['log_value'].median().reset_index()
        ax_a.scatter(exp_med['chain_len'], exp_med['log_value'], color=c, s=65, zorder=6,
                     edgecolors='white', lw=0.8, marker='D', alpha=0.9)

# *** FIX: threshold lines drawn to xmax; labels placed at right edge with transform ***
ax_a.axhline(B_THRESH,  color='#555', lw=1.0, ls=':',  alpha=0.8)
ax_a.axhline(VB_THRESH, color='#555', lw=1.0, ls='--', alpha=0.8)
# Use annotate with no arrow to position labels in right margin, clipped=False
ax_a.annotate('B (3.3)',  xy=(19.5, B_THRESH),  xycoords='data',
               fontsize=10, color='#555', va='bottom', ha='left',
               annotation_clip=False)
ax_a.annotate('vB (4.3)', xy=(19.5, VB_THRESH), xycoords='data',
               fontsize=10, color='#555', va='bottom', ha='left',
               annotation_clip=False)

ax_a.set_xlim(2.5, 19.0); ax_a.set_xticks(range(3, 19, 2))
ax_a.set_xlabel('Perfluorocarbon chain length (C)', labelpad=5)
ax_a.set_ylabel('log BAF', labelpad=5)
ax_a.set_title('(a) SAR: chain length drives bioaccumulation across functional classes\n'
               '(shaded = IQR; diamonds = observed training data medians)', pad=8, fontsize=14, fontweight='bold')
handles_a = [mpatches.Patch(color=CLS_COLORS[c], label=c) for c in plot_cls]
handles_a += [plt.Line2D([0],[0], color='#555', ls=':', lw=1.2, label='OECD B threshold (3.3)'),
              plt.Line2D([0],[0], color='#555', ls='--', lw=1.2, label='OECD vB threshold (4.3)')]
ax_a.legend(handles=handles_a, frameon=False, fontsize=11, loc='upper left')
neat_axes(ax_a)
ax_a.text(-0.04, 1.04, 'a', transform=ax_a.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (b) Headgroup effect at C7–C9 ─────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
c8_data = []
for cls in CLS_ORDER:
    sub = m2p_cl[(m2p_cl['norm_class'] == cls) & (m2p_cl['chain_len'].between(7, 9))]
    c8_data.append(sub['pred_logBAF'].values if len(sub) >= 5 else np.array([np.nan]))

bplot = ax_b.boxplot([d for d in c8_data if len(d) > 1], patch_artist=True,
                     medianprops=dict(color='white', lw=2.2),
                     whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
                     flierprops=dict(marker='o', markersize=2.5, alpha=0.4, mew=0))
valid_cls = [cls for cls, d in zip(CLS_ORDER, c8_data) if len(d) > 1]
for patch, cls in zip(bplot['boxes'], valid_cls):
    patch.set_facecolor(CLS_COLORS.get(cls, NPG[5]))
    patch.set_alpha(0.82)
ax_b.axhline(B_THRESH,  color='#555', lw=1.0, ls=':',  alpha=0.8)
ax_b.axhline(VB_THRESH, color='#555', lw=1.0, ls='--', alpha=0.8)
ax_b.set_xticks(range(1, len(valid_cls) + 1))
ax_b.set_xticklabels(valid_cls, rotation=28, ha='right', fontsize=11)
ax_b.set_ylabel('Predicted logBAF', labelpad=5)
ax_b.set_title('(b) Headgroup effect at C7–C9\n(PFSA > PFCA > Ether-PFAS)', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_b)
ax_b.text(-0.14, 1.04, 'b', transform=ax_b.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (c) Chain length at B threshold crossing ───────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
threshold_cl = {}
for cls in CLS_ORDER:
    sub_sar = sar[sar['class'] == cls].sort_values('chain_len')
    if len(sub_sar) == 0: continue
    crossed = sub_sar[sub_sar['pred_logBAF_median'] >= B_THRESH]
    if len(crossed) > 0:
        threshold_cl[cls] = int(crossed.iloc[0]['chain_len'])

cls_thr   = list(threshold_cl.keys())
cl_vals   = [threshold_cl[c] for c in cls_thr]
colors_thr = [CLS_COLORS.get(c, NPG[5]) for c in cls_thr]
ax_c.bar(range(len(cls_thr)), cl_vals, color=colors_thr, width=0.6, edgecolor='white', lw=0.5)
for xi, (cls, v) in enumerate(zip(cls_thr, cl_vals)):
    ax_c.text(xi, v + 0.12, f'C{v}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax_c.set_xticks(range(len(cls_thr))); ax_c.set_xticklabels(cls_thr, fontsize=14, fontweight='bold')
ax_c.set_ylabel('Carbon chain length at B threshold (C)')
ax_c.set_title('(c) Minimum chain length to exceed OECD B\n(PFSA crosses earliest at C6–7)',
               pad=8, fontsize=14, fontweight='bold')
ax_c.set_ylim(0, max(cl_vals) * 1.28)
neat_axes(ax_c)
ax_c.text(-0.14, 1.04, 'c', transform=ax_c.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (d) B-exceedance fraction vs chain length ──────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
chain_lens = range(4, 17)
for cls in ['PFCA', 'PFSA', 'FTOH']:
    pct_b = []
    for cl in chain_lens:
        sub_cl = m2p_cl[(m2p_cl['norm_class'] == cls) & (m2p_cl['chain_len'] == cl)]
        pct_b.append(100 * (sub_cl['pred_logBAF'] >= B_THRESH).mean() if len(sub_cl) >= 3 else np.nan)
    pct_arr = np.array(pct_b)
    valid   = np.isfinite(pct_arr)
    ax_d.plot(np.array(list(chain_lens))[valid], pct_arr[valid],
              color=CLS_COLORS[cls], lw=2.5, marker='o', markersize=5.5, label=cls)

ax_d.axhline(50, color='#666', lw=0.8, ls='--', alpha=0.6, label='50% majority')
ax_d.set_xlabel('Carbon chain length (C)', labelpad=5)
ax_d.set_ylabel('Fraction exceeding B threshold (%)', labelpad=5)
ax_d.set_title('(d) B-exceedance fraction vs chain length\n(PFSA and FTOH escalate faster than PFCA)',
               pad=8, fontsize=14, fontweight='bold')
ax_d.legend(frameon=False, fontsize=11)
ax_d.set_ylim(-5, 108)
neat_axes(ax_d)
ax_d.text(-0.14, 1.04, 'd', transform=ax_d.transAxes, fontsize=17, fontweight='bold', va='top')

# ── (e) Fluorination × chain length 2D heatmap ────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
m2p_cl = m2p_cl.copy()
m2p_cl['n_F_per_C'] = m2p_cl['SMILES'].apply(
    lambda s: str(s).count('F') / max(str(s).count('C'), 1))
cl_bins = pd.cut(m2p_cl['chain_len'], bins=[2,5,8,11,14,18],
                 labels=['C3-5','C6-8','C9-11','C12-14','C15-18'])
fl_bins = pd.cut(m2p_cl['n_F_per_C'], bins=[0,0.5,1.0,1.5,2.0,3.0],
                 labels=['<0.5','0.5-1','1-1.5','1.5-2','>2'])
m2p_cl['cl_bin'] = cl_bins.values
m2p_cl['fl_bin'] = fl_bins.values
heat_int = m2p_cl.groupby(['fl_bin','cl_bin'])['pred_logBAF'].median().unstack(fill_value=np.nan)
if heat_int.shape[0] > 0 and heat_int.shape[1] > 0:
    im_e = ax_e.imshow(heat_int.values, cmap='YlOrRd', aspect='auto',
                       vmin=1, vmax=5, interpolation='nearest')
    cb_e = fig.colorbar(im_e, ax=ax_e, fraction=0.04, pad=0.02)
    cb_e.set_label('Predicted logBAF', fontsize=11)
    ax_e.set_xticks(range(len(heat_int.columns)))
    ax_e.set_xticklabels(heat_int.columns, rotation=28, ha='right', fontsize=10)
    ax_e.set_yticks(range(len(heat_int.index)))
    ax_e.set_yticklabels(heat_int.index, fontsize=10)
    ax_e.set_xlabel('Chain length bin', labelpad=5)
    ax_e.set_ylabel('F/C fluorination ratio', labelpad=5)
    ax_e.set_title('(e) Feature interaction: chain length × fluorination\n'
                   '(long chain + high F/C = highest bioaccumulation)', pad=8, fontsize=14, fontweight='bold')
neat_axes(ax_e)
ax_e.text(-0.14, 1.04, 'e', transform=ax_e.transAxes, fontsize=17, fontweight='bold', va='top')

fig.suptitle('Mechanistic structure–activity relationships for PFAS bioaccumulation\n'
             '(chain length, headgroup, fluorination ratio as joint determinants; n=2,661 class-annotated compounds)',
             fontsize=14, y=0.99)
fig.savefig(OUT / 'Fig3_SAR_mechanism_v3.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig3_v2 saved: {(OUT/'Fig3_SAR_mechanism_v3.png').stat().st_size//1024} KB")
