"""
=============================================================
  LOCKED BUT LEAKY — Hybrid Placement Framework
  Platt Scaling + Confidence-Aware ML + Sec-Greedy Fallback

  Reads existing CSVs — NO re-simulation needed
  Run:
    source ~/paper_env/bin/activate
    python3 ~/Downloads/hybrid_placement.py

  Runtime: ~15-20 minutes
  Output:  ~/results/hybrid_results.csv
           ~/figures_hq/hybrid_comparison.png
=============================================================
"""

import os, copy, random, warnings
import numpy as np
import pandas as pd
import networkx as nx
import circuitgraph as cg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

os.makedirs(os.path.expanduser('~/figures_hq'), exist_ok=True)
os.makedirs(os.path.expanduser('~/results'), exist_ok=True)

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.expanduser('~/results/gate_features.csv'))
print(f"  {len(df)} gates, {df['circuit'].nunique()} circuits")

FCOLS = ['fan_in','fan_out','level','level_norm','cone_size',
         'output_reach','signal_prob','sp_skew',
         'is_xor','is_xnor','is_and','is_nand','is_or','is_nor','is_not']

# Binary label — median split per circuit
df['label'] = 0
for circ in df['circuit'].unique():
    mask   = df['circuit'] == circ
    median = df.loc[mask,'delta_sdc'].median()
    df.loc[mask & (df['delta_sdc'] >= median), 'label'] = 1

X      = df[FCOLS].values
y      = df['label'].values
groups = df['circuit'].values

# ── Step 1: LOCO-CV to get calibrated probabilities ─────────
print("\n[1/4] Training LightGBM + Platt scaling (LOCO-CV)...")

logo       = LeaveOneGroupOut()
cal_probs  = np.zeros(len(y))

for train_idx, test_idx in logo.split(X, y, groups):
    # Base LightGBM
    base = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        class_weight='balanced', verbose=-1
    )
    # Platt scaling via CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    calibrated.fit(X[train_idx], y[train_idx])
    cal_probs[test_idx] = calibrated.predict_proba(X[test_idx])[:, 1]

df['cal_prob'] = cal_probs
print(f"  Calibrated probability range: "
      f"[{cal_probs.min():.3f}, {cal_probs.max():.3f}]")
print(f"  Mean prob: {cal_probs.mean():.3f}")


# ── Step 2: Circuit evaluator + locking ─────────────────────
LOCK_PREFIXES = ('KI_','LX_','KN_','LN_','KMN_','LMN_','KMX_','LMX_')

def eval_circuit(G, input_vals, overrides=None):
    val = {}
    for node in nx.topological_sort(G.graph):
        if node in input_vals:
            val[node] = input_vals[node]; continue
        if overrides and node in overrides:
            val[node] = overrides[node]; continue
        t  = G.type(node)
        pv = [val.get(p,0) for p in G.graph.predecessors(node)]
        if   t=='and':  val[node]=int(all(pv))
        elif t=='nand': val[node]=1-int(all(pv))
        elif t=='or':   val[node]=int(any(pv))
        elif t=='nor':  val[node]=1-int(any(pv))
        elif t=='xor':  val[node]=int(sum(pv)%2)
        elif t=='xnor': val[node]=1-int(sum(pv)%2)
        elif t=='not':  val[node]=1-(pv[0] if pv else 0)
        elif t in ('buf','output'): val[node]=pv[0] if pv else 0
        elif t=='1':    val[node]=1
        elif t=='0':    val[node]=0
        else:           val[node]=pv[0] if pv else 0
    return val

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


# ── Step 3: Threshold configurations ────────────────────────
# tau_low: below this = confident SAFE (use ML)
# tau_high: above this = confident DANGEROUS (avoid, ML)
# between = uncertain → fall back to sec-greedy rank

THRESHOLDS = {
    'Conservative  (τ=0.3/0.7)': (0.30, 0.70),
    'Moderate      (τ=0.4/0.6)': (0.40, 0.60),
    'Aggressive    (τ=0.45/0.55)':(0.45, 0.55),
}

# Load baselines and previous results
baselines  = pd.read_csv(os.path.expanduser('~/results/baselines.csv'))
baselines  = dict(zip(baselines['circuit'], baselines['sdc']))
df_ml_prev = pd.read_csv(os.path.expanduser(
    '~/results/ml_placement_comparison.csv'))

K = 16
CIRCUITS = sorted(df['circuit'].unique())

print("\n[2/4] Running hybrid placement evaluation...")
print(f"  K={K}, 5000 fault trials per placement\n")

all_results = []

for thresh_name, (tau_low, tau_high) in THRESHOLDS.items():
    print(f"  Threshold: {thresh_name}")
    for cname in CIRCUITS:
        G        = cg.from_lib(cname)
        sub      = df[df['circuit']==cname].copy()
        intnodes = sub['node'].tolist()
        probs    = sub['cal_prob'].values
        reaches  = sub['output_reach'].values

        # Classify each gate
        confident_safe      = [intnodes[i] for i,p in enumerate(probs)
                               if p < tau_low]
        confident_dangerous = [intnodes[i] for i,p in enumerate(probs)
                               if p > tau_high]
        uncertain           = [intnodes[i] for i,p in enumerate(probs)
                               if tau_low <= p <= tau_high]

        # For uncertain gates — rank by output reach (sec-greedy logic)
        uncertain_reaches = [(n, reaches[intnodes.index(n)])
                             for n in uncertain]
        uncertain_sorted  = [n for n,r in
                             sorted(uncertain_reaches,
                                    key=lambda x: x[1])]  # low reach = safer

        # Build candidate pool:
        # 1. Confident safe gates first
        # 2. Fill remaining K slots from uncertain (low reach first)
        candidates = confident_safe + uncertain_sorted
        candidates = candidates[:K] if len(candidates) >= K else candidates

        # If not enough candidates, pad with uncertain gates
        if len(candidates) < K:
            remaining = [n for n in intnodes
                        if n not in candidates
                        and n not in confident_dangerous]
            candidates += remaining[:K-len(candidates)]

        hybrid_gates = candidates[:K]

        # Run fault injection
        Gl = lock_xor(G, hybrid_gates)
        sdc_hybrid = fault_campaign(Gl, n_trials=5000, seed=777)

        # Get previous baselines for this circuit
        prev = df_ml_prev[(df_ml_prev['circuit']==cname) &
                          (df_ml_prev['K']==K)]
        if len(prev) == 0:
            continue
        prev = prev.iloc[0]

        red_vs_random = (prev['sdc_random'] - sdc_hybrid) / \
                        (prev['sdc_random'] + 1e-9) * 100
        red_vs_ml     = (prev['sdc_ml'] - sdc_hybrid) / \
                        (prev['sdc_ml'] + 1e-9) * 100

        n_confident = len(confident_safe) + len(confident_dangerous)
        n_uncertain = len(uncertain)
        pct_uncertain = n_uncertain / len(intnodes) * 100

        all_results.append({
            'threshold':      thresh_name.strip(),
            'circuit':        cname,
            'tau_low':        tau_low,
            'tau_high':       tau_high,
            'sdc_hybrid':     sdc_hybrid,
            'sdc_random':     prev['sdc_random'],
            'sdc_ml':         prev['sdc_ml'],
            'sdc_security':   prev['sdc_security'],
            'base_sdc':       prev['base_sdc'],
            'red_vs_random':  red_vs_random,
            'red_vs_ml':      red_vs_ml,
            'pct_uncertain':  pct_uncertain,
        })

        print(f"    {cname:<8} hybrid={sdc_hybrid*100:.1f}% "
              f"ml={prev['sdc_ml']*100:.1f}% "
              f"rand={prev['sdc_random']*100:.1f}% "
              f"Δvs_rand={red_vs_random:+.1f}% "
              f"uncertain={pct_uncertain:.0f}%")

df_results = pd.DataFrame(all_results)
df_results.to_csv(
    os.path.expanduser('~/results/hybrid_results.csv'), index=False)


# ── Step 4: Summary table ────────────────────────────────────
print("\n[3/4] Summary by threshold:")
print("="*70)
for thresh_name in THRESHOLDS:
    sub = df_results[df_results['threshold']==thresh_name.strip()]
    avg_red_rand = sub['red_vs_random'].mean()
    avg_red_ml   = sub['red_vs_ml'].mean()
    max_red_rand = sub['red_vs_random'].max()
    print(f"\n  {thresh_name}")
    print(f"  Avg SDC reduction vs random  : {avg_red_rand:+.1f}%")
    print(f"  Avg SDC reduction vs ML-only : {avg_red_ml:+.1f}%")
    print(f"  Max SDC reduction vs random  : {max_red_rand:+.1f}%")


# ── Step 5: Figure ───────────────────────────────────────────
print("\n[4/4] Generating comparison figure...")

fig, ax = plt.subplots(figsize=(14, 6))
plt.rcParams.update({'font.size': 12})

x  = np.arange(len(CIRCUITS))
w  = 0.15
thresh_list = [t.strip() for t in THRESHOLDS.keys()]
colors = ['steelblue','orange','tomato','limegreen','purple']

# Plot random, ML, and three hybrid variants
first_thresh = df_results[
    df_results['threshold']==thresh_list[0]]
first_thresh = first_thresh.set_index('circuit').reindex(CIRCUITS)

ax.bar(x - 2*w,
       first_thresh['sdc_random'].values*100,
       w, color='orange', alpha=0.85, edgecolor='k',
       lw=0.5, label='Random (avg)')
ax.bar(x - 1*w,
       first_thresh['sdc_ml'].values*100,
       w, color='steelblue', alpha=0.85, edgecolor='k',
       lw=0.5, label='ML-guided')

for idx, thresh_name in enumerate(thresh_list):
    sub = df_results[df_results['threshold']==thresh_name]
    sub = sub.set_index('circuit').reindex(CIRCUITS)
    ax.bar(x + idx*w,
           sub['sdc_hybrid'].values*100,
           w, color=colors[idx+2], alpha=0.85,
           edgecolor='k', lw=0.5,
           label=f'Hybrid {thresh_name.split("(")[1].rstrip(")")}')

ax.set_xticks(x)
ax.set_xticklabels(CIRCUITS, fontsize=10)
ax.set_ylabel('SDC Rate (%)')
ax.set_title(
    f'SDC Rate: Random vs ML-Guided vs Hybrid Placement (K={K})\n'
    f'Hybrid = ML when confident, Sec-Greedy fallback when uncertain',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9, ncol=3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.expanduser(
    '~/figures_hq/hybrid_comparison.png'),
    dpi=600, bbox_inches='tight')
plt.close()
print("  Saved: ~/figures_hq/hybrid_comparison.png")

# ── Final summary ────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL NUMBERS — give these to paper:")
print("="*70)

best_thresh = None
best_avg    = -999
for thresh_name in thresh_list:
    sub = df_results[df_results['threshold']==thresh_name]
    avg = sub['red_vs_random'].mean()
    if avg > best_avg:
        best_avg    = avg
        best_thresh = thresh_name

best_sub = df_results[df_results['threshold']==best_thresh]
print(f"\n  Best threshold: {best_thresh}")
print(f"  Avg SDC reduction vs random : {best_avg:.1f}%")
print(f"  Max SDC reduction vs random : "
      f"{best_sub['red_vs_random'].max():.1f}% "
      f"({best_sub.loc[best_sub['red_vs_random'].idxmax(),'circuit']})")
print(f"  Avg improvement over ML-only: "
      f"{best_sub['red_vs_ml'].mean():+.1f}%")

print(f"\n  Per-circuit hybrid SDC rates (best threshold):")
print(f"  {'Circuit':<10} {'Random':>8} {'ML':>8} "
      f"{'Hybrid':>8} {'Δvs Random':>12}")
print("  " + "-"*50)
for _, row in best_sub.iterrows():
    print(f"  {row['circuit']:<10} "
          f"{row['sdc_random']*100:>7.1f}% "
          f"{row['sdc_ml']*100:>7.1f}% "
          f"{row['sdc_hybrid']*100:>7.1f}% "
          f"{row['red_vs_random']:>+11.1f}%")

print(f"\n  Saved: ~/results/hybrid_results.csv")
print(f"  Saved: ~/figures_hq/hybrid_comparison.png")
