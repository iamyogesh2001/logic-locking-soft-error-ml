"""
=============================================================
  LOCKED BUT LEAKY — Figure Regeneration Script
  Reads existing CSV results — NO re-simulation
  600 DPI, 14pt fonts, publication ready

  Run:
    source ~/paper_env/bin/activate
    python3 ~/Downloads/regenerate_figures.py

  Output: ~/figures_hq/
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Setup ────────────────────────────────────────────────────
os.makedirs(os.path.expanduser('~/figures_hq'), exist_ok=True)

DPI      = 600
FONTSIZE = 14
CMAP     = plt.cm.tab10.colors

plt.rcParams.update({
    'font.size':        FONTSIZE,
    'axes.titlesize':   FONTSIZE,
    'axes.labelsize':   FONTSIZE,
    'xtick.labelsize':  FONTSIZE - 1,
    'ytick.labelsize':  FONTSIZE - 1,
    'legend.fontsize':  FONTSIZE - 2,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ── Load data ────────────────────────────────────────────────
print("Loading results...")
baselines  = pd.read_csv(os.path.expanduser('~/results/baselines.csv'))
df_locked  = pd.read_csv(os.path.expanduser('~/results/locked_results.csv'))
df_ml      = pd.read_csv(os.path.expanduser('~/results/ml_placement_comparison.csv'))
feat_imp   = pd.read_csv(os.path.expanduser('~/results/feature_importance.csv'))

baselines  = dict(zip(baselines['circuit'], baselines['sdc']))
CIRCUITS   = ['c432','c499','c880','c1355','c1908',
              'c2670','c3540','c5315','c6288','c7552']
SCHEMES    = ['XOR','XNOR','Mixed']
K16        = 16

print(f"  Loaded {len(df_locked)} locked results")
print(f"  Loaded {len(df_ml)} ML placement rows")


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Baseline SDC + Locked Boxplots
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    'SDC Rate: Unlocked vs. Locked Circuits (ISCAS-85 Benchmarks)',
    fontsize=FONTSIZE + 1, fontweight='bold', y=1.01
)

# Left — baseline bar chart
ax = axes[0]
bv = [baselines[c] * 100 for c in CIRCUITS]
bars = ax.barh(CIRCUITS, bv, color='steelblue', alpha=0.85,
               edgecolor='k', linewidth=0.6)
for i, v in enumerate(bv):
    ax.text(v + 0.8, i, f'{v:.1f}%', va='center',
            fontsize=FONTSIZE - 2, color='black')
ax.set_xlabel('SDC Rate (%)')
ax.set_title('(a) Baseline — Unlocked Circuits')
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Right — locked boxplots
ax2   = axes[1]
offs  = [-0.28, 0, 0.28]
for idx, scheme in enumerate(SCHEMES):
    for ci, cname in enumerate(CIRCUITS):
        mask = ((df_locked['circuit'] == cname) &
                (df_locked['scheme']  == scheme) &
                (df_locked['K']       == K16))
        vals = df_locked[mask]['sdc'].values * 100
        if len(vals) == 0:
            continue
        ax2.boxplot(
            vals,
            positions=[ci + offs[idx]],
            widths=0.22,
            patch_artist=True,
            boxprops=dict(facecolor=CMAP[idx], alpha=0.75),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
            flierprops=dict(marker='.', markersize=3),
            showfliers=True
        )

ax2.set_xticks(range(len(CIRCUITS)))
ax2.set_xticklabels(CIRCUITS, rotation=30, ha='right')
ax2.set_ylabel('SDC Rate (%)')
ax2.set_title(f'(b) Locked — XOR / XNOR / Mixed, K={K16}, 20 Placements Each')
ax2.legend(
    handles=[mpatches.Patch(color=CMAP[i], alpha=0.75, label=s)
             for i, s in enumerate(SCHEMES)],
    loc='upper left'
)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.tight_layout()
out = os.path.expanduser('~/figures_hq/fig1_sdc_baseline_locked.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print("  fig1 saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — SDC Gap (max - min) Across Placements
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
x, w = np.arange(len(CIRCUITS)), 0.25

for idx, scheme in enumerate(SCHEMES):
    gaps = []
    for cname in CIRCUITS:
        mask = ((df_locked['circuit'] == cname) &
                (df_locked['scheme']  == scheme) &
                (df_locked['K']       == K16))
        vals = df_locked[mask]['sdc'].values * 100
        gaps.append(max(vals) - min(vals) if len(vals) > 1 else 0)

    bars = ax.bar(x + (idx - 1) * w, gaps, w,
                  color=CMAP[idx], alpha=0.85,
                  edgecolor='k', linewidth=0.5, label=scheme)
    for bar, g in zip(bars, gaps):
        if g > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    g + 0.2, f'{g:.1f}',
                    ha='center', va='bottom',
                    fontsize=FONTSIZE - 4, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(CIRCUITS)
ax.set_ylabel('SDC Rate Gap: max − min Placement (pp)')
ax.set_title(
    f'Reliability Gap Between Best and Worst Key-Gate Placement\n'
    f'(K={K16}, 20 Random Placements per Scheme — Identical Security Strength)',
    fontsize=FONTSIZE
)
ax.legend(title='Locking Scheme')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, 28)

# Annotation on c499
c_idx = CIRCUITS.index('c499')
ax.annotate(
    'Up to 23.2 pp gap\n(same security)',
    xy=(c_idx + 0 * w, 23.2),
    xytext=(c_idx + 1.2, 25.5),
    fontsize=FONTSIZE - 3,
    color='darkred', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5)
)

plt.tight_layout()
out = os.path.expanduser('~/figures_hq/fig2_sdc_gap.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print("  fig2 saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Four-Way Comparison
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
sub = df_ml[df_ml['K'] == K16].copy()
sub = sub.set_index('circuit').reindex(CIRCUITS).reset_index()

x, w = np.arange(len(CIRCUITS)), 0.2

b0 = ax.bar(x - 1.5*w,
            [baselines[c]*100 for c in CIRCUITS],
            w, color='steelblue', alpha=0.85,
            edgecolor='k', linewidth=0.5,
            label='Unlocked baseline')
b1 = ax.bar(x - 0.5*w,
            sub['sdc_random'].values * 100,
            w, color='orange', alpha=0.85,
            edgecolor='k', linewidth=0.5,
            label='Random placement (avg)')
b2 = ax.bar(x + 0.5*w,
            sub['sdc_security'].values * 100,
            w, color='tomato', alpha=0.85,
            edgecolor='k', linewidth=0.5,
            label='Security-greedy placement')
b3 = ax.bar(x + 1.5*w,
            sub['sdc_ml'].values * 100,
            w, color='limegreen', alpha=0.9,
            edgecolor='k', linewidth=0.5,
            label='ML-guided placement (ours)')

ax.set_xticks(x)
ax.set_xticklabels(CIRCUITS)
ax.set_ylabel('SDC Rate (%)')
ax.set_title(
    f'SDC Rate Comparison: Unlocked / Random / Security-Greedy / ML-Guided (K={K16})\n'
    f'ML-guided placement achieves up to 26.5% SDC reduction vs. random baseline',
    fontsize=FONTSIZE
)
ax.legend(ncol=2, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, 105)

# Annotate best reduction
c_idx = CIRCUITS.index('c499')
ax.annotate(
    '26.5% reduction\nvs random',
    xy=(c_idx + 1.5*w, sub[sub['circuit']=='c499']['sdc_ml'].values[0]*100),
    xytext=(c_idx + 2.8, 72),
    fontsize=FONTSIZE - 3,
    color='darkgreen', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5)
)

plt.tight_layout()
out = os.path.expanduser('~/figures_hq/fig3_four_way_comparison.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print("  fig3 saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Feature Importance
# ══════════════════════════════════════════════════════════════

# Clean feature names for publication
name_map = {
    'signal_prob':    'Signal Probability',
    'output_reach':   'Output Reach',
    'level_norm':     'Normalized Level (Depth)',
    'cone_size':      'Fanin Cone Size',
    'sp_skew':        'Signal Probability Skew',
    'level':          'Topological Level',
    'fan_out':        'Fanout',
    'fan_in':         'Fanin',
    'is_nand':        'Gate Type: NAND',
    'is_and':         'Gate Type: AND',
    'is_xor':         'Gate Type: XOR',
    'is_nor':         'Gate Type: NOR',
    'is_or':          'Gate Type: OR',
    'is_not':         'Gate Type: NOT',
    'is_xnor':        'Gate Type: XNOR',
}

fi = feat_imp.head(10).copy()
fi['feature_clean'] = fi['feature'].map(name_map).fillna(fi['feature'])

fig, ax = plt.subplots(figsize=(10, 6))
colors = [CMAP[i % len(CMAP)] for i in range(len(fi))]
ax.barh(
    fi['feature_clean'].values[::-1],
    fi['importance'].values[::-1],
    color=colors[::-1],
    alpha=0.85, edgecolor='k', linewidth=0.5
)
ax.set_xlabel('LightGBM Feature Importance (Split Gain)')
ax.set_title(
    'Top-10 Gate Features Predicting SDC Contribution\n'
    'of Key-Gate Placement (LightGBM, Leave-One-Circuit-Out CV)',
    fontsize=FONTSIZE
)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.expanduser('~/figures_hq/fig4_feature_importance.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print("  fig4 saved.")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  ALL FIGURES SAVED TO ~/figures_hq/")
print("="*55)
print("  fig1_sdc_baseline_locked.png")
print("  fig2_sdc_gap.png")
print("  fig3_four_way_comparison.png")
print("  fig4_feature_importance.png")
print(f"\n  Resolution : 600 DPI")
print(f"  Font size  : {FONTSIZE}pt")
print(f"  Format     : PNG (convert to PDF in LaTeX)")
