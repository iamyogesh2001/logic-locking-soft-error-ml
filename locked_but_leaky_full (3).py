"""
=============================================================
  LOCKED BUT LEAKY — Full Experiment Script v2 (FIXED)
  Paper: "Locked but Leaky: Soft-Error Side-Effect of Logic
          Locking with ML-Driven Reliability-Aware Placement"
  Authors: Yogesh Rethinapandian, Arun Karthik Sundararajan
=============================================================

HOW TO RUN:
  source ~/paper_env/bin/activate
  python3 ~/Downloads/locked_but_leaky_full.py

Runtime on M4: ~25-35 minutes
Results: ./results/   Figures: ./figures/
=============================================================
"""

import os, copy, random, warnings
import numpy as np
import networkx as nx
import circuitgraph as cg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
N_TRIALS      = 10_000
N_PLACEMENTS  = 20
KEY_SIZES     = [8, 16, 32]
CIRCUITS      = ['c432','c499','c880','c1355','c1908',
                 'c2670','c3540','c5315','c6288','c7552']
N_JOBS        = -1
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 1. CIRCUIT EVALUATOR
# ══════════════════════════════════════════════════════════════
def eval_circuit(G, input_vals, overrides=None):
    val = {}
    for node in nx.topological_sort(G.graph):
        if node in input_vals:
            val[node] = input_vals[node]
            continue
        if overrides and node in overrides:
            val[node] = overrides[node]
            continue
        t  = G.type(node)
        pv = [val.get(p, 0) for p in G.graph.predecessors(node)]
        if   t == 'and':   val[node] = int(all(pv))
        elif t == 'nand':  val[node] = 1 - int(all(pv))
        elif t == 'or':    val[node] = int(any(pv))
        elif t == 'nor':   val[node] = 1 - int(any(pv))
        elif t == 'xor':   val[node] = int(sum(pv) % 2)
        elif t == 'xnor':  val[node] = 1 - int(sum(pv) % 2)
        elif t == 'not':   val[node] = 1 - (pv[0] if pv else 0)
        elif t in ('buf','output'): val[node] = pv[0] if pv else 0
        elif t == '1':     val[node] = 1
        elif t == '0':     val[node] = 0
        else:              val[node] = pv[0] if pv else 0
    return val


# ══════════════════════════════════════════════════════════════
# 2. LOCKING SCHEMES  — XOR, XNOR, Mixed (all gate-only, no MUX)
# ══════════════════════════════════════════════════════════════
def _xor_lock_single(Gl, kg, gate_type, key_prefix, lock_prefix):
    """Insert one XOR or XNOR key gate. Internal helper."""
    if kg not in Gl.nodes():
        return
    ki = f"{key_prefix}{kg}"
    lk = f"{lock_prefix}{kg}"
    Gl.add(ki, "input")
    Gl.add(lk, gate_type, fanin=[kg, ki])
    succs = [s for s in list(Gl.graph.successors(kg)) if s != lk]
    for s in succs:
        Gl.graph.remove_edge(kg, s)
        Gl.graph.add_edge(lk, s)


def lock_xor(G, key_nodes, key_bits):
    Gl = copy.deepcopy(G)
    for kg, kb in zip(key_nodes, key_bits):
        _xor_lock_single(Gl, kg, "xor", "KI_", "LX_")
    return Gl


def lock_xnor(G, key_nodes, key_bits):
    Gl = copy.deepcopy(G)
    for kg, kb in zip(key_nodes, key_bits):
        _xor_lock_single(Gl, kg, "xnor", "KN_", "LN_")
    return Gl


def lock_mixed(G, key_nodes, key_bits):
    """
    Mixed XOR/XNOR locking based on signal probability.
    Gates with SP > 0.5 get XNOR; others get XOR.
    Resists signal-probability-based structural attacks.
    """
    # estimate signal probabilities once
    rng_sp = random.Random(0)
    inputs = list(G.inputs())
    counts = defaultdict(int)
    for _ in range(200):
        inv = {i: rng_sp.randint(0, 1) for i in inputs}
        val = eval_circuit(G, inv)
        for n, v in val.items():
            counts[n] += v
    sp = {n: counts[n] / 200 for n in G.nodes()}

    Gl = copy.deepcopy(G)
    for kg, kb in zip(key_nodes, key_bits):
        if sp.get(kg, 0.5) > 0.5:
            _xor_lock_single(Gl, kg, "xnor", "KMN_", "LMN_")
        else:
            _xor_lock_single(Gl, kg, "xor",  "KMX_", "LMX_")
    return Gl


LOCKING_SCHEMES = {
    'XOR':   lock_xor,
    'XNOR':  lock_xnor,
    'Mixed': lock_mixed,
}

LOCK_PREFIXES = ('KI_','LX_','KN_','LN_','KMN_','LMN_','KMX_','LMX_')


# ══════════════════════════════════════════════════════════════
# 3. FAULT INJECTION
# ══════════════════════════════════════════════════════════════
def get_internal_nodes(G):
    return [n for n in G.nodes()
            if G.type(n) not in ('input','output','1','0','buf')
            and not any(n.startswith(p) for p in LOCK_PREFIXES)]


def fault_campaign(G, n_trials=10_000, seed=0):
    rng      = random.Random(seed)
    inputs   = list(G.inputs())
    outputs  = list(G.outputs())
    intnodes = get_internal_nodes(G)
    if not intnodes:
        return 0.0, 1.0
    sdc = masked = 0
    for _ in range(n_trials):
        inv    = {i: rng.randint(0, 1) for i in inputs}
        golden = eval_circuit(G, inv)
        fn     = rng.choice(intnodes)
        ov     = {fn: 1 - golden.get(fn, 0)}
        faulty = eval_circuit(G, inv, overrides=ov)
        if any(faulty.get(o, -1) != golden.get(o, -1) for o in outputs):
            sdc    += 1
        else:
            masked += 1
    return sdc / n_trials, masked / n_trials


# ══════════════════════════════════════════════════════════════
# 4. GATE FEATURES
# ══════════════════════════════════════════════════════════════
def extract_gate_features(G):
    topo   = list(nx.topological_sort(G.graph))
    levels = {}
    for node in topo:
        preds = list(G.graph.predecessors(node))
        levels[node] = max((levels.get(p, 0) for p in preds), default=0) + 1

    rng    = random.Random(0)
    inputs = list(G.inputs())
    counts = defaultdict(int)
    for _ in range(200):
        inv = {i: rng.randint(0, 1) for i in inputs}
        for n, v in eval_circuit(G, inv).items():
            counts[n] += v
    sp = {n: counts[n] / 200 for n in G.nodes()}

    cone_cache = {}
    def cone_size(node):
        if node in cone_cache: return cone_cache[node]
        preds = list(G.graph.predecessors(node))
        sz = 1 + sum(cone_size(p) for p in preds) if preds else 1
        cone_cache[node] = sz
        return sz

    out_set     = set(G.outputs())
    reach_cache = {}
    def output_reach(node):
        if node in reach_cache: return reach_cache[node]
        if node in out_set:
            reach_cache[node] = 1; return 1
        r = sum(output_reach(s) for s in G.graph.successors(node))
        reach_cache[node] = r; return r

    intnodes  = get_internal_nodes(G)
    max_level = max(levels.values()) if levels else 1
    features  = {}
    for node in intnodes:
        t = G.type(node)
        features[node] = {
            'fan_in':       len(list(G.graph.predecessors(node))),
            'fan_out':      len(list(G.graph.successors(node))),
            'level':        levels.get(node, 0),
            'level_norm':   levels.get(node, 0) / max_level,
            'cone_size':    cone_size(node),
            'output_reach': output_reach(node),
            'signal_prob':  sp.get(node, 0.5),
            'sp_skew':      abs(sp.get(node, 0.5) - 0.5),
            'is_xor':       int(t == 'xor'),
            'is_xnor':      int(t == 'xnor'),
            'is_and':       int(t == 'and'),
            'is_nand':      int(t == 'nand'),
            'is_or':        int(t == 'or'),
            'is_nor':       int(t == 'nor'),
            'is_not':       int(t == 'not'),
        }
    return features


# ══════════════════════════════════════════════════════════════
# 5. PARALLEL WORKER
# ══════════════════════════════════════════════════════════════
def run_one_placement(args):
    cname, scheme_name, K, run_id, seed = args
    G        = cg.from_lib(cname)
    intnodes = get_internal_nodes(G)
    rng      = random.Random(seed)
    kn       = rng.sample(intnodes, min(K, len(intnodes)))
    kb       = [rng.randint(0, 1) for _ in kn]
    Gl       = LOCKING_SCHEMES[scheme_name](G, kn, kb)
    sdc, mask = fault_campaign(Gl, n_trials=N_TRIALS, seed=seed + 1000)
    return {'circuit': cname, 'scheme': scheme_name,
            'K': K, 'run_id': run_id, 'sdc': sdc, 'mask': mask}


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  LOCKED BUT LEAKY — Full Experiment v2 (Fixed)")
print("=" * 65)

# Baselines
print("\n[1/4] Baseline SDC rates...")
baselines = {}
for cname in tqdm(CIRCUITS):
    G = cg.from_lib(cname)
    sdc, mask = fault_campaign(G, n_trials=N_TRIALS, seed=0)
    baselines[cname] = {'sdc': sdc, 'mask': mask}
    tqdm.write(f"  {cname:<10} SDC={sdc*100:.1f}%  Masked={mask*100:.1f}%")

pd.DataFrame([{'circuit':c,'sdc':v['sdc'],'mask':v['mask']}
              for c,v in baselines.items()]).to_csv('results/baselines.csv', index=False)

# Locked campaigns
print("\n[2/4] Locked fault campaigns...")
tasks = [(c, s, K, r, abs(hash((c,s,K,r))) % (2**31))
         for c in CIRCUITS for s in LOCKING_SCHEMES
         for K in KEY_SIZES for r in range(N_PLACEMENTS)]
print(f"  Total jobs: {len(tasks)}")

locked_results = Parallel(n_jobs=N_JOBS, prefer='threads')(
    delayed(run_one_placement)(t) for t in tqdm(tasks))
df_locked = pd.DataFrame(locked_results)
df_locked.to_csv('results/locked_results.csv', index=False)
print(f"  Saved {len(df_locked)} rows.")

# Gate features
print("\n[3/4] Gate features + per-gate SDC contribution...")
feature_rows = []
for cname in tqdm(CIRCUITS):
    G        = cg.from_lib(cname)
    feats    = extract_gate_features(G)
    base_sdc = baselines[cname]['sdc']
    rng2     = random.Random(99999)
    for node in list(feats.keys()):
        Gl    = lock_xor(G, [node], [rng2.randint(0,1)])
        sdc_g, _ = fault_campaign(Gl, n_trials=2000, seed=rng2.randint(0,99999))
        row   = dict(feats[node])
        row.update({'circuit': cname, 'node': node,
                    'delta_sdc': sdc_g - base_sdc})
        feature_rows.append(row)

df_features = pd.DataFrame(feature_rows)
df_features.to_csv('results/gate_features.csv', index=False)
print(f"  {len(df_features)} gate rows saved.")

# ML
print("\n[4/4] LightGBM LOCO-CV...")
FCOLS  = ['fan_in','fan_out','level','level_norm','cone_size',
          'output_reach','signal_prob','sp_skew',
          'is_xor','is_xnor','is_and','is_nand','is_or','is_nor','is_not']
X      = df_features[FCOLS].values
y      = df_features['delta_sdc'].values
groups = df_features['circuit'].values
logo   = LeaveOneGroupOut()
preds  = np.zeros_like(y)
mae_s, r2_s = [], []

for tr, te in logo.split(X, y, groups):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                           num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    preds[te] = p
    mae_s.append(mean_absolute_error(y[te], p))
    r2_s.append(r2_score(y[te], p))

print(f"  MAE: {np.mean(mae_s):.5f} ± {np.std(mae_s):.5f}")
print(f"  R² : {np.mean(r2_s):.3f} ± {np.std(r2_s):.3f}")

final_m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, verbose=-1)
final_m.fit(X, y)
feat_imp = pd.DataFrame({'feature': FCOLS,
                         'importance': final_m.feature_importances_}
                        ).sort_values('importance', ascending=False)
feat_imp.to_csv('results/feature_importance.csv', index=False)
print("\n  Top-5 features:")
print(feat_imp.head(5).to_string(index=False))

# ML placement evaluation
print("\n  ML placement vs baselines...")
ml_rows = []
for cname in CIRCUITS:
    G        = cg.from_lib(cname)
    feats    = extract_gate_features(G)
    intnodes = list(feats.keys())
    base_sdc = baselines[cname]['sdc']
    FM       = np.array([[feats[n][f] for f in FCOLS] for n in intnodes])
    pd_      = final_m.predict(FM)

    for K in KEY_SIZES:
        ml_kn  = [intnodes[i] for i in np.argsort(pd_)[:K]]
        sec_kn = [intnodes[i] for i in
                  np.argsort([-feats[n]['output_reach'] for n in intnodes])[:K]]
        sdc_ml,  _ = fault_campaign(lock_xor(G, ml_kn,  [0]*K), N_TRIALS, 555)
        sdc_sec, _ = fault_campaign(lock_xor(G, sec_kn, [0]*K), N_TRIALS, 556)
        mask = ((df_locked['circuit']==cname) &
                (df_locked['scheme']=='XOR') & (df_locked['K']==K))
        sr_avg = df_locked[mask]['sdc'].mean()
        sr_min = df_locked[mask]['sdc'].min()
        ml_rows.append({
            'circuit': cname, 'K': K,
            'base_sdc': base_sdc,
            'sdc_random': sr_avg, 'sdc_random_min': sr_min,
            'sdc_security': sdc_sec, 'sdc_ml': sdc_ml,
            'red_vs_random':  (sr_avg  - sdc_ml)/(sr_avg  +1e-9)*100,
            'red_vs_sec':     (sdc_sec - sdc_ml)/(sdc_sec +1e-9)*100,
        })

df_ml = pd.DataFrame(ml_rows)
df_ml.to_csv('results/ml_placement_comparison.csv', index=False)
print("\n  K=16 summary:")
print(df_ml[df_ml['K']==16][
    ['circuit','base_sdc','sdc_random','sdc_security','sdc_ml','red_vs_random']
].to_string(index=False))


# ══════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════
print("\n  Generating figures...")
CMAP = plt.cm.tab10.colors
plt.rcParams.update({'font.size': 11})
schemes = list(LOCKING_SCHEMES.keys())

# Fig 1 — Baseline + locked boxplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Fig 1 — SDC Rate: Unlocked vs Locked (ISCAS-85)',
             fontweight='bold')
ax = axes[0]
bv = [baselines[c]['sdc']*100 for c in CIRCUITS]
ax.barh(CIRCUITS, bv, color='steelblue', alpha=0.85, edgecolor='k', lw=0.6)
for i, v in enumerate(bv):
    ax.text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('SDC Rate (%)')
ax.set_title('Baseline (Unlocked)')
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.3)

ax2  = axes[1]
offs = [-0.28, 0, 0.28]
K16  = 16
for idx, scheme in enumerate(schemes):
    for ci, cname in enumerate(CIRCUITS):
        mask = ((df_locked['circuit']==cname) &
                (df_locked['scheme']==scheme) & (df_locked['K']==K16))
        vals = df_locked[mask]['sdc'].values * 100
        if len(vals) == 0: continue
        ax2.boxplot(vals, positions=[ci+offs[idx]], widths=0.22,
                    patch_artist=True,
                    boxprops=dict(facecolor=CMAP[idx], alpha=0.75),
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                    flierprops=dict(marker='.', ms=3), showfliers=True)
ax2.set_xticks(range(len(CIRCUITS)))
ax2.set_xticklabels(CIRCUITS, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('SDC Rate (%)')
ax2.set_title(f'Locked — Three Schemes, K={K16}, 20 placements')
ax2.legend(handles=[mpatches.Patch(color=CMAP[i], alpha=0.75, label=s)
                    for i,s in enumerate(schemes)], fontsize=9)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig1_sdc_baseline_locked.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig1 saved.")

# Fig 2 — SDC gap
fig, ax = plt.subplots(figsize=(12, 5))
x, w = np.arange(len(CIRCUITS)), 0.25
for idx, scheme in enumerate(schemes):
    gaps = []
    for cname in CIRCUITS:
        mask = ((df_locked['circuit']==cname) &
                (df_locked['scheme']==scheme) & (df_locked['K']==K16))
        vals = df_locked[mask]['sdc'].values * 100
        gaps.append(max(vals)-min(vals) if len(vals)>1 else 0)
    bars = ax.bar(x+(idx-1)*w, gaps, w, color=CMAP[idx],
                  alpha=0.85, edgecolor='k', lw=0.5, label=scheme)
    for bar, g in zip(bars, gaps):
        if g > 0.3:
            ax.text(bar.get_x()+bar.get_width()/2, g+0.05,
                    f'{g:.1f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(CIRCUITS, fontsize=10)
ax.set_ylabel('SDC Gap: max − min placement (%)')
ax.set_title(f'Fig 2 — Reliability Gap Across 20 Random Placements (K={K16})')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig2_sdc_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig2 saved.")

# Fig 3 — Four-way comparison
fig, ax = plt.subplots(figsize=(13, 5))
sub = df_ml[df_ml['K']==K16]
x, w = np.arange(len(CIRCUITS)), 0.2
ax.bar(x-1.5*w, [baselines[c]['sdc']*100 for c in CIRCUITS],
       w, color='steelblue', alpha=0.8, edgecolor='k', lw=0.5, label='Unlocked')
ax.bar(x-0.5*w, sub['sdc_random'].values*100,
       w, color='orange', alpha=0.8, edgecolor='k', lw=0.5, label='Random (avg)')
ax.bar(x+0.5*w, sub['sdc_security'].values*100,
       w, color='tomato', alpha=0.8, edgecolor='k', lw=0.5, label='Security-greedy')
ax.bar(x+1.5*w, sub['sdc_ml'].values*100,
       w, color='limegreen', alpha=0.9, edgecolor='k', lw=0.5, label='ML-guided (ours)')
ax.set_xticks(x); ax.set_xticklabels(CIRCUITS, fontsize=10)
ax.set_ylabel('SDC Rate (%)'); ax.legend(fontsize=9, ncol=2)
ax.set_title(f'Fig 3 — Unlocked / Random / Security-Greedy / ML-Guided (K={K16})')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig3_four_way_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig3 saved.")

# Fig 4 — Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
fi = feat_imp.head(10)
ax.barh(fi['feature'].values[::-1], fi['importance'].values[::-1],
        color=[CMAP[i%len(CMAP)] for i in range(len(fi))][::-1],
        alpha=0.85, edgecolor='k', lw=0.5)
ax.set_xlabel('LightGBM Feature Importance')
ax.set_title('Fig 4 — Gate Features Predicting SDC Contribution')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig4 saved.")


# ══════════════════════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════════════════════
avg_red = df_ml[df_ml['K']==16]['red_vs_random'].mean()
max_red = df_ml[df_ml['K']==16]['red_vs_random'].max()
best    = df_ml[df_ml['K']==16].sort_values('red_vs_random',ascending=False).iloc[0]
print("\n" + "="*65)
print("  DONE. Numbers for your abstract:")
print("="*65)
print(f"  Avg SDC reduction vs random  : {avg_red:.1f}%")
print(f"  Max SDC reduction            : {max_red:.1f}%  ({best['circuit']})")
print(f"  ML R²  (LOCO-CV)             : {np.mean(r2_s):.3f}")
print(f"  ML MAE (LOCO-CV)             : {np.mean(mae_s):.5f}")
print(f"\n  Figures → ./figures/")
print(f"  Data    → ./results/")
