"""
SI_supplementary: 用 NPG 配色重绘消融图和 zero-shot 图
输出文件直接覆盖 SI_supplementary/ 中的同名 png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

SI_OUT = ROOT / 'result/discussion2/SI_supplementary'

# ─── SI Fig 1: Geometry-aware ablation (M0–M3 × MAE) ──────────────────────────
abl_path = ROOT / 'result/stage6_geometry/ablation_results.csv'
if abl_path.exists():
    abl_df = pd.read_csv(abl_path)
    fig_ep  = 'logBCF'
    sub_abl = abl_df[(abl_df['endpoint'] == fig_ep) & (abl_df['split'] == 'chemical-out')]
    model_order   = ['M0 (2D only)', 'M1 (2D+Transport)', 'M2 (2D+3D)', 'M3 (2D+Transport+3D)']
    model_present = [m for m in model_order if m in sub_abl['model'].values]
    if model_present:
        maes   = [sub_abl[sub_abl['model'] == m]['MAE'].values[0] for m in model_present]
        colors = [NPG[i] for i in range(len(model_present))]

        fig, ax = plt.subplots(figsize=(8, 5))
        x    = np.arange(len(model_present))
        bars = ax.bar(x, maes, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_present, rotation=22, ha='right', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'MAE (log {fig_ep.replace("log", "")})', fontsize=14)
        ax.set_title('Supplementary Fig. S1 | Geometry-aware feature ablation\n'
                     '(chemical-out split; lower MAE = better)', pad=10)
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        neat_axes(ax)
        plt.tight_layout(pad=1.8)
        plt.savefig(SI_OUT / 'fig_ablation.png', dpi=300)
        plt.close()
        print(f"SI Fig1 saved  ({(SI_OUT/'fig_ablation.png').stat().st_size//1024} KB)")
    else:
        print("No model_present for ablation figure")
else:
    print(f"[SKIP] ablation_results.csv not found at {abl_path}")

# ─── SI Fig 2: Zero-shot Spearman bar ─────────────────────────────────────────
zs_path   = ROOT / 'result/stage5_zeroshot/zero_shot_results.csv'
if zs_path.exists():
    results_df = pd.read_csv(zs_path)
    SPLIT_ORDER = [
        'Random split', 'Chemical-out split', 'Subclass-out split',
        'Organism-out split', 'Phylogeny-out split', 'Endpoint zero-shot',
    ]
    fig_df  = results_df.dropna(subset=['Spearman'])
    fig_df  = fig_df[fig_df['endpoint'].isin(['logBAF', 'logBCF'])]
    fig_rows = []
    for sname in SPLIT_ORDER:
        sub = fig_df[fig_df['split'] == sname]
        for ep in ['logBAF', 'logBCF']:
            r = sub[sub['endpoint'] == ep]
            if not r.empty:
                fig_rows.append(r.iloc[0])
                break
    if fig_rows:
        fig_df2 = pd.DataFrame(fig_rows)
        short_names = {
            'Random split'         : 'Random',
            'Chemical-out split'   : 'Chemical-out',
            'Subclass-out split'   : 'Subclass-out',
            'Organism-out split'   : 'Organism-out',
            'Phylogeny-out split'  : 'Phylogeny-out',
            'Endpoint zero-shot'   : 'Endpoint zero-shot',
        }
        # color by performance tier
        rho_vals = fig_df2['Spearman'].values
        colors = [NPG[2] if v > 0.7 else (NPG[3] if v > 0.2 else NPG[1]) for v in rho_vals]

        fig, ax = plt.subplots(figsize=(9, 5.2))
        x    = np.arange(len(fig_df2))
        bars = ax.bar(x, rho_vals, color=colors, width=0.60, edgecolor='white', linewidth=0.8)

        # 95% CI error bars
        lo = rho_vals - fig_df2['Spearman_lo'].values
        hi = fig_df2['Spearman_hi'].values - rho_vals
        ax.errorbar(x, rho_vals, yerr=[lo, hi],
                    fmt='none', color='#333333', capsize=4.5, linewidth=1.3, zorder=5)

        # value labels above CI caps
        for xi, v, h in zip(x, rho_vals, fig_df2['Spearman_hi'].values):
            ax.text(xi, h + 0.04, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([short_names.get(s, s) for s in fig_df2['split'].values],
                           rotation=28, ha='right', fontsize=14, fontweight='bold')
        ax.set_ylabel("Spearman's ρ", fontsize=14)
        ax.set_ylim(-0.2, 1.25)
        ax.axhline(0,   color='#888', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(0.7, color=NPG[2], lw=1.0, ls=':', alpha=0.8, label='ρ=0.7 (actionable)')
        ax.set_title('Supplementary Fig. S2 | Spearman ρ across six generalization splits\n'
                     '(Bootstrap 95% CI, n=300 resamples)', pad=10)

        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(color=NPG[2], label='ρ>0.7 (actionable)'),
                           mpatches.Patch(color=NPG[3], label='0.2<ρ≤0.7 (limited)'),
                           mpatches.Patch(color=NPG[1], label='ρ≤0.2 (failed)')],
                  frameon=False, fontsize=11, loc='upper right')
        neat_axes(ax)
        plt.tight_layout(pad=1.8)
        plt.savefig(SI_OUT / 'fig_zeroshot_spearman.png', dpi=300)
        plt.close()
        print(f"SI Fig2 saved  ({(SI_OUT/'fig_zeroshot_spearman.png').stat().st_size//1024} KB)")
    else:
        print("No rows for zeroshot figure")
else:
    print(f"[SKIP] zero_shot_results.csv not found at {zs_path}")
