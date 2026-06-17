"""
=============================================================
  LOCKED BUT LEAKY — Classification Metrics for ML Section
  Kaushik: run this script to get TP, FP, TN, FN, F1, 
  Precision, Recall, Confusion Matrix
  
  Reads: ~/results/gate_features.csv
  No re-simulation needed.

  Run:
    source ~/paper_env/bin/activate
    pip install xgboost --quiet
    python3 ~/Downloads/classification_metrics.py
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
import lightgbm as lgb

os.makedirs(os.path.expanduser('~/figures_hq'), exist_ok=True)
os.makedirs(os.path.expanduser('~/results'), exist_ok=True)

# ── Load data ────────────────────────────────────────────────
print("Loading gate_features.csv...")
df = pd.read_csv(os.path.expanduser('~/results/gate_features.csv'))
print(f"  {len(df)} gates across {df['circuit'].nunique()} circuits")

# ── Feature columns ──────────────────────────────────────────
FCOLS = ['fan_in','fan_out','level','level_norm','cone_size',
         'output_reach','signal_prob','sp_skew',
         'is_xor','is_xnor','is_and','is_nand','is_or','is_nor','is_not']

# ── Binary label: median split ────────────────────────────────
# Dangerous = top 50% delta_sdc per circuit (label=1)
# Safe      = bottom 50% delta_sdc per circuit (label=0)
df['label'] = 0
for circ in df['circuit'].unique():
    mask   = df['circuit'] == circ
    median = df.loc[mask, 'delta_sdc'].median()
    df.loc[mask & (df['delta_sdc'] >= median), 'label'] = 1

print(f"\n  Label distribution:")
print(f"  Dangerous (1): {df['label'].sum()} gates ({df['label'].mean()*100:.1f}%)")
print(f"  Safe      (0): {(df['label']==0).sum()} gates ({(df['label']==0).mean()*100:.1f}%)")

X      = df[FCOLS].values
y      = df['label'].values
groups = df['circuit'].values

# ── LOCO-CV Classification ────────────────────────────────────
print("\n" + "="*60)
print("  LOCO-CV Classification Results")
print("="*60)

logo  = LeaveOneGroupOut()
all_preds  = np.zeros_like(y)
all_probs  = np.zeros(len(y))

per_circuit = []

for train_idx, test_idx in logo.split(X, y, groups):
    circ_name = groups[test_idx][0]
    
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        class_weight='balanced', verbose=-1
    )
    model.fit(X[train_idx], y[train_idx])
    
    preds = model.predict(X[test_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    
    all_preds[test_idx] = preds
    all_probs[test_idx] = probs
    
    # per circuit metrics
    cm  = confusion_matrix(y[test_idx], preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    f1_mac = f1_score(y[test_idx], preds, average='macro', zero_division=0)
    f1_mic = f1_score(y[test_idx], preds, average='micro', zero_division=0)
    prec   = precision_score(y[test_idx], preds, average='macro', zero_division=0)
    rec    = recall_score(y[test_idx], preds, average='macro', zero_division=0)
    
    per_circuit.append({
        'circuit': circ_name,
        'TP': int(tp), 'FP': int(fp),
        'TN': int(tn), 'FN': int(fn),
        'Precision': round(prec, 3),
        'Recall':    round(rec,  3),
        'F1 Macro':  round(f1_mac, 3),
        'F1 Micro':  round(f1_mic, 3),
    })

# ── Print per-circuit table ───────────────────────────────────
df_metrics = pd.DataFrame(per_circuit)
print(f"\n  {'Circuit':<10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} "
      f"{'Prec':>7} {'Rec':>7} {'F1 Mac':>8} {'F1 Mic':>8}")
print("  " + "-"*65)
for _, row in df_metrics.iterrows():
    print(f"  {row['circuit']:<10} {row['TP']:>5} {row['FP']:>5} "
          f"{row['TN']:>5} {row['FN']:>5} "
          f"{row['Precision']:>7.3f} {row['Recall']:>7.3f} "
          f"{row['F1 Macro']:>8.3f} {row['F1 Micro']:>8.3f}")

# Overall
print("\n" + "="*60)
print("  OVERALL (aggregated across all circuits)")
print("="*60)
cm_total = confusion_matrix(y, all_preds)
tn_t, fp_t, fn_t, tp_t = cm_total.ravel()
f1_mac_total = f1_score(y, all_preds, average='macro')
f1_mic_total = f1_score(y, all_preds, average='micro')
prec_total   = precision_score(y, all_preds, average='macro')
rec_total    = recall_score(y, all_preds, average='macro')

print(f"  TP: {tp_t}  FP: {fp_t}  TN: {tn_t}  FN: {fn_t}")
print(f"  Precision (macro): {prec_total:.3f}")
print(f"  Recall    (macro): {rec_total:.3f}")
print(f"  F1 Macro         : {f1_mac_total:.3f}")
print(f"  F1 Micro         : {f1_mic_total:.3f}")

print(f"\n  Full classification report:")
print(classification_report(y, all_preds,
      target_names=['Safe (0)', 'Dangerous (1)']))

# ── Save metrics CSV ─────────────────────────────────────────
df_metrics.to_csv(
    os.path.expanduser('~/results/classification_metrics.csv'),
    index=False)
print("\n  Saved: ~/results/classification_metrics.csv")

# ── Figure 1: Confusion Matrix ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_total,
                               display_labels=['Safe', 'Dangerous'])
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix — LOCO-CV\n'
             'Gate Dangerousness Classification (Median Split)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.expanduser(
    '~/figures_hq/confusion_matrix.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved: ~/figures_hq/confusion_matrix.png")

# ── Figure 2: Per-circuit F1 Macro bar chart ─────────────────
fig, ax = plt.subplots(figsize=(10, 4))
circuits = df_metrics['circuit'].tolist()
x = np.arange(len(circuits))
w = 0.35
ax.bar(x - w/2, df_metrics['F1 Macro'], w,
       color='steelblue', alpha=0.85, edgecolor='k', lw=0.5,
       label='F1 Macro')
ax.bar(x + w/2, df_metrics['F1 Micro'], w,
       color='tomato', alpha=0.85, edgecolor='k', lw=0.5,
       label='F1 Micro')
for i, (mac, mic) in enumerate(zip(df_metrics['F1 Macro'],
                                    df_metrics['F1 Micro'])):
    ax.text(i - w/2, mac + 0.01, f'{mac:.2f}',
            ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, mic + 0.01, f'{mic:.2f}',
            ha='center', va='bottom', fontsize=8)
ax.axhline(f1_mac_total, color='steelblue', lw=1.5,
           linestyle='--', label=f'Avg Macro={f1_mac_total:.3f}')
ax.axhline(f1_mic_total, color='tomato', lw=1.5,
           linestyle='--', label=f'Avg Micro={f1_mic_total:.3f}')
ax.set_xticks(x)
ax.set_xticklabels(circuits, fontsize=10)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.1)
ax.set_title('Per-Circuit F1 Score — LOCO-CV Classification\n'
             'Dangerous vs. Safe Key-Gate Identification',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.expanduser(
    '~/figures_hq/f1_per_circuit.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved: ~/figures_hq/f1_per_circuit.png")

# ── Figure 3: Predicted vs Actual scatter ────────────────────
df_reg = pd.read_csv(os.path.expanduser('~/results/gate_features.csv'))
FCOLS_REG = FCOLS

# retrain regressor for scatter
from sklearn.model_selection import LeaveOneGroupOut
preds_reg = np.zeros(len(df_reg))
y_reg     = df_reg['delta_sdc'].values
g_reg     = df_reg['circuit'].values

for tr, te in LeaveOneGroupOut().split(
        df_reg[FCOLS_REG].values, y_reg, g_reg):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                           num_leaves=31, random_state=42, verbose=-1)
    m.fit(df_reg[FCOLS_REG].values[tr], y_reg[tr])
    preds_reg[te] = m.predict(df_reg[FCOLS_REG].values[te])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
circuits_list = sorted(df_reg['circuit'].unique())
for ax, circ in zip(axes.flatten(), circuits_list):
    mask = g_reg == circ
    ax.scatter(y_reg[mask], preds_reg[mask],
               alpha=0.4, s=15, color='steelblue', edgecolors='none')
    mn = min(y_reg[mask].min(), preds_reg[mask].min())
    mx = max(y_reg[mask].max(), preds_reg[mask].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.2, label='Ideal')
    ax.set_title(circ, fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual Δ SDC', fontsize=8)
    ax.set_ylabel('Predicted Δ SDC', fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle('Predicted vs. Actual Δ SDC per Circuit (LOCO-CV)\n'
             'Each point = one gate held out during training',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.expanduser(
    '~/figures_hq/predicted_vs_actual.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved: ~/figures_hq/predicted_vs_actual.png")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  DONE. Give these numbers to Kaushik:")
print("="*60)
print(f"  TP : {tp_t}")
print(f"  FP : {fp_t}")
print(f"  TN : {tn_t}")
print(f"  FN : {fn_t}")
print(f"  Precision (macro): {prec_total:.3f}")
print(f"  Recall    (macro): {rec_total:.3f}")
print(f"  F1 Macro         : {f1_mac_total:.3f}")
print(f"  F1 Micro         : {f1_mic_total:.3f}")
print(f"\n  3 new figures in ~/figures_hq/")
print(f"  - confusion_matrix.png")
print(f"  - f1_per_circuit.png")
print(f"  - predicted_vs_actual.png")
