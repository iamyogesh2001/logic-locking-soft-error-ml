"""
=============================================================
  LOCKED BUT LEAKY — ML Retraining Script
  Reads existing gate_features.csv — no re-simulation needed
  Run: python3 ~/Downloads/retrain_ml.py
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────
print("Loading gate_features.csv...")
df = pd.read_csv(os.path.expanduser('~/results/gate_features.csv'))
print(f"  {len(df)} rows, {df['circuit'].nunique()} circuits")
print(f"  Circuits: {sorted(df['circuit'].unique())}")

# ── Feature engineering ──────────────────────────────────────
# Original features
BASE_FEATS = ['fan_in','fan_out','level','level_norm','cone_size',
              'output_reach','signal_prob','sp_skew',
              'is_xor','is_xnor','is_and','is_nand','is_or','is_nor','is_not']

# New interaction features
df['cone_x_reach']    = df['cone_size']    * df['output_reach']
df['sp_x_fanout']     = df['signal_prob']  * df['fan_out']
df['level_x_reach']   = df['level']        * df['output_reach']
df['cone_x_sp']       = df['cone_size']    * df['signal_prob']
df['reach_x_spskew']  = df['output_reach'] * df['sp_skew']
df['fanout_x_level']  = df['fan_out']      * df['level_norm']
df['cone_sq']         = df['cone_size']    ** 2
df['reach_sq']        = df['output_reach'] ** 2
df['sp_centered']     = (df['signal_prob'] - 0.5).abs()

NEW_FEATS = ['cone_x_reach','sp_x_fanout','level_x_reach',
             'cone_x_sp','reach_x_spskew','fanout_x_level',
             'cone_sq','reach_sq','sp_centered']

ALL_FEATS = BASE_FEATS + NEW_FEATS

X      = df[ALL_FEATS].values
y      = df['delta_sdc'].values
groups = df['circuit'].values

print(f"\n  Features: {len(BASE_FEATS)} base + {len(NEW_FEATS)} engineered = {len(ALL_FEATS)} total")
print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
print(f"  Target mean:  {y.mean():.4f}  std: {y.std():.4f}")

# ── LOCO-CV function ─────────────────────────────────────────
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def loco_cv(model, X, y, groups, scale=False):
    logo   = LeaveOneGroupOut()
    maes, r2s = [], []
    preds  = np.zeros_like(y)
    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[te] = p
        maes.append(mean_absolute_error(yte, p))
        r2s.append(r2_score(yte, p))
    return np.mean(maes), np.std(maes), np.mean(r2s), np.std(r2s), preds

# ── Models ───────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'LightGBM (baseline)': lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03,
        num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, min_child_samples=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1),

    'LightGBM (tuned)': lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.01,
        num_leaves=31, subsample=0.7,
        colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=42, verbose=-1),

    'XGBoost': xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03,
        max_depth=4, subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0),

    'Random Forest': RandomForestRegressor(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5, max_features=0.6,
        random_state=42, n_jobs=-1),

    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        random_state=42),

    'Huber Regression': HuberRegressor(epsilon=1.35, max_iter=500),
}

# ── Run all models ───────────────────────────────────────────
print("\n" + "="*65)
print("  LOCO-CV Results (train on 9 circuits, test on 1)")
print("="*65)
print(f"{'Model':<28} {'MAE':>10} {'R²':>10}")
print("-"*50)

results = {}
best_r2  = -999
best_name = None
best_preds = None

for name, model in models.items():
    scale = 'Regression' in name
    mae, mae_std, r2, r2_std, preds = loco_cv(model, X, y, groups, scale=scale)
    results[name] = {'mae': mae, 'r2': r2, 'preds': preds}
    marker = ' <-- BEST' if r2 > best_r2 else ''
    print(f"  {name:<26} {mae:>8.5f}   {r2:>+7.3f}{marker}")
    if r2 > best_r2:
        best_r2   = r2
        best_name = name
        best_preds = preds

print("-"*50)
print(f"\n  Best model: {best_name}  (R²={best_r2:.3f})")

# ── Per-circuit breakdown for best model ─────────────────────
print(f"\n  Per-circuit R² for {best_name}:")
print(f"  {'Circuit':<10} {'N gates':>8} {'R²':>10} {'MAE':>10}")
print("  " + "-"*40)
for circ in sorted(df['circuit'].unique()):
    mask = groups == circ
    yt   = y[mask]
    yp   = best_preds[mask]
    r2c  = r2_score(yt, yp)
    maec = mean_absolute_error(yt, yp)
    n    = mask.sum()
    print(f"  {circ:<10} {n:>8} {r2c:>+10.3f} {maec:>10.5f}")

# ── Train final best model on all data ───────────────────────
print(f"\n  Training final {best_name} on all data...")
best_model = models[best_name]
if 'Regression' in best_name:
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    best_model.fit(X_scaled, y)
else:
    best_model.fit(X, y)

# Save predictions
df['pred_delta_sdc'] = best_preds
df.to_csv(os.path.expanduser('~/results/gate_features_retrained.csv'), index=False)

# ── Feature importance for tree models ───────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'feature': ALL_FEATS,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv(os.path.expanduser('~/results/feature_importance_retrained.csv'), index=False)
    print(f"\n  Top-8 features ({best_name}):")
    print(fi.head(8).to_string(index=False))

# ── ML placement re-evaluation ───────────────────────────────
print("\n" + "="*65)
print("  ML Placement Re-evaluation with Best Model")
print("="*65)

import circuitgraph as cg
import random
import copy
import networkx as nx

def eval_circuit(G, input_vals, overrides=None):
    val = {}
    for node in nx.topological_sort(G.graph):
        if node in input_vals:
            val[node] = input_vals[node]; continue
        if overrides and node in overrides:
            val[node] = overrides[node]; continue
        t  = G.type(node)
        pv = [val.get(p,0) for p in G.graph.predecessors(node)]
        if   t=='and':   val[node]=int(all(pv))
        elif t=='nand':  val[node]=1-int(all(pv))
        elif t=='or':    val[node]=int(any(pv))
        elif t=='nor':   val[node]=1-int(any(pv))
        elif t=='xor':   val[node]=int(sum(pv)%2)
        elif t=='xnor':  val[node]=1-int(sum(pv)%2)
        elif t=='not':   val[node]=1-(pv[0] if pv else 0)
        elif t in ('buf','output'): val[node]=pv[0] if pv else 0
        elif t=='1':     val[node]=1
        elif t=='0':     val[node]=0
        else:            val[node]=pv[0] if pv else 0
    return val

LOCK_PREFIXES = ('KI_','LX_','KN_','LN_','KMN_','LMN_','KMX_','LMX_')

def get_internal_nodes(G):
    return [n for n in G.nodes()
            if G.type(n) not in ('input','output','1','0','buf')
            and not any(n.startswith(p) for p in LOCK_PREFIXES)]

def lock_xor(G, key_nodes):
    Gl = copy.deepcopy(G)
    for kg in key_nodes:
        if kg not in Gl.nodes(): continue
        ki = f"KI_{kg}"; lk = f"LX_{kg}"
        Gl.add(ki,"input")
        Gl.add(lk,"xor",fanin=[kg,ki])
        succs=[s for s in list(Gl.graph.successors(kg)) if s!=lk]
        for s in succs:
            Gl.graph.remove_edge(kg,s)
            Gl.graph.add_edge(lk,s)
    return Gl

def fault_campaign(G, n_trials=5000, seed=0):
    rng=random.Random(seed)
    inputs=list(G.inputs()); outputs=list(G.outputs())
    intnodes=get_internal_nodes(G)
    if not intnodes: return 0.0
    sdc=0
    for _ in range(n_trials):
        inv={i:rng.randint(0,1) for i in inputs}
        golden=eval_circuit(G,inv)
        fn=rng.choice(intnodes)
        ov={fn:1-golden.get(fn,0)}
        faulty=eval_circuit(G,inv,overrides=ov)
        if any(faulty.get(o,-1)!=golden.get(o,-1) for o in outputs):
            sdc+=1
    return sdc/n_trials

# Load baselines
baselines = pd.read_csv(os.path.expanduser('~/results/baselines.csv'))
baselines = dict(zip(baselines['circuit'], baselines['sdc']))

# Load locked results for random baseline
df_locked = pd.read_csv(os.path.expanduser('~/results/locked_results.csv'))

print(f"\n  {'Circuit':<10} {'Base':>8} {'Random':>10} {'SecGreedy':>11} {'ML-guided':>11} {'Reduc%':>8}")
print("  " + "-"*60)

K = 16
ml_rows = []
for circ in sorted(df['circuit'].unique()):
    G        = cg.from_lib(circ)
    sub      = df[df['circuit']==circ].copy()
    intnodes = sub['node'].tolist()
    feats    = sub[ALL_FEATS].values

    if 'Regression' in best_name:
        pred_d = best_model.predict(sc.transform(feats))
    else:
        pred_d = best_model.predict(feats)

    # ML: pick K gates with lowest predicted SDC delta
    ml_gates  = [intnodes[i] for i in np.argsort(pred_d)[:K]]
    sec_gates = sub.nlargest(K,'output_reach')['node'].tolist()

    base_sdc = baselines[circ]
    sdc_ml,  = fault_campaign(lock_xor(G,ml_gates),  5000, 999),
    sdc_sec, = fault_campaign(lock_xor(G,sec_gates), 5000, 998),
    sdc_ml   = sdc_ml[0] if isinstance(sdc_ml,tuple) else sdc_ml
    sdc_sec  = sdc_sec[0] if isinstance(sdc_sec,tuple) else sdc_sec

    mask = ((df_locked['circuit']==circ) &
            (df_locked['scheme']=='XOR') & (df_locked['K']==K))
    sdc_rand = df_locked[mask]['sdc'].mean()
    red = (sdc_rand - sdc_ml)/(sdc_rand+1e-9)*100

    ml_rows.append({'circuit':circ,'base_sdc':base_sdc,
                    'sdc_random':sdc_rand,'sdc_security':sdc_sec,
                    'sdc_ml':sdc_ml,'red_vs_random':red})
    print(f"  {circ:<10} {base_sdc:>7.1%} {sdc_rand:>10.1%} "
          f"{sdc_sec:>10.1%} {sdc_ml:>10.1%} {red:>+7.1f}%")

df_ml = pd.DataFrame(ml_rows)
df_ml.to_csv(os.path.expanduser('~/results/ml_placement_retrained.csv'), index=False)

avg_red = df_ml['red_vs_random'].mean()
max_red = df_ml['red_vs_random'].max()
best_c  = df_ml.loc[df_ml['red_vs_random'].idxmax(),'circuit']

print("\n" + "="*65)
print("  FINAL NUMBERS FOR PAPER")
print("="*65)
print(f"  Best model          : {best_name}")
print(f"  LOCO-CV R²          : {best_r2:.3f}")
print(f"  Avg SDC reduction   : {avg_red:.1f}%")
print(f"  Max SDC reduction   : {max_red:.1f}%  ({best_c})")
print(f"\n  Saved:")
print(f"  ~/results/gate_features_retrained.csv")
print(f"  ~/results/ml_placement_retrained.csv")
print(f"  ~/results/feature_importance_retrained.csv")
