# Logic Locking and Soft-Error Vulnerability
## An ML-Guided Framework for Reliability-Aware Key-Gate Placement in VLSI Circuits

> **Authors:** Yogesh Rethinapandian · Arun Karthik Sundararajan · Kaushik Kumar · Smrithi Prakash

---

## Overview

Logic locking is a widely used hardware IP protection technique that
inserts key-controlled gates into a circuit netlist. While existing
research focuses exclusively on the **security** properties of locking
schemes, this work asks a different question:

> *What happens to a circuit's soft-error vulnerability when you lock it?*

We present the **first systematic study** of the reliability
side-effects of logic locking. Using Monte Carlo fault injection across
all ten ISCAS-85 benchmark circuits under three locking schemes and
three key sizes, we show that **SDC rate varies by up to 23.2
percentage points** across placements of identical security strength.
We then propose an **ML-guided placement framework** that reduces SDC
rate by up to **26.5%** relative to random placement baselines, while
preserving equivalent SAT-attack resistance.

---

## Key Findings

| Finding | Result |
|---|---|
| Max SDC gap across identical security placements | **23.2 pp** (c499, XNOR locking) |
| Average SDC gap across all circuits and schemes | **6.8 pp** |
| Max SDC reduction by ML-guided placement | **26.5%** (c499) |
| Average SDC reduction vs. random placement | **11.1%** |
| Gate-level dataset size | **9,677 samples** |
| Total fault injection trials | **18,000,000** |

---

## Repository Structure

```
logic-locking-soft-error-ml/
│
├── simulation/
│   └── locked_but_leaky_full.py       # Full experiment script
│       ├── Phase 1: Baseline SDC characterization
│       ├── Phase 2: Logic locking (XOR, XNOR, Mixed) x K x placements
│       ├── Phase 3: Gate feature extraction + per-gate SDC contribution
│       └── Phase 4: LightGBM training + ML-guided placement evaluation
│
├── ml/
│   └── retrain_ml.py                  # ML retraining script
│       ├── Feature engineering (24 features)
│       ├── Multi-model comparison (LightGBM, XGBoost, Random Forest)
│       └── LOCO-CV evaluation
│
├── figures/
│   ├── fig1_sdc_baseline_locked.png   # Baseline SDC + locked boxplots
│   ├── fig2_sdc_gap.png               # Reliability gap across placements
│   ├── fig3_four_way_comparison.png   # ML vs random vs security-greedy
│   ├── fig4_feature_importance.png    # LightGBM feature importance
│   └── arch_diagram_simple.png        # System overview diagram
│
├── results/
│   ├── baselines.csv                  # Unlocked SDC rates per circuit
│   ├── locked_results.csv             # 1,800 locked variant results
│   ├── gate_features.csv              # 9,677 gate-level feature dataset
│   ├── feature_importance.csv         # ML feature importance ranking
│   └── ml_placement_comparison.csv    # ML vs baseline placement results
│
├── diagrams/
│   ├── arch_diagram_simple.drawio     # Editable system overview
│   └── circuit_level_diagram.drawio   # XOR locking + SEU propagation
│
├── paper/
│   └── locked_but_leaky_paper.tex     # Full IEEE-format LaTeX manuscript
│
└── README.md
```

---

## Experimental Setup

### Benchmarks
All ten **ISCAS-85** combinational benchmark circuits:
`c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552`

Gates range from **171** (c432) to **2,381** (c6288).
Loaded directly from the `circuitgraph` Python library — no external downloads needed.

### Locking Schemes
| Scheme | Description |
|---|---|
| **XOR** | XOR gate at selected node, key=0 passes signal |
| **XNOR** | XNOR gate at selected node, key=1 passes signal |
| **Mixed XOR/XNOR** | Gate type assigned by signal probability (SP>0.5 → XNOR) |

### Fault Model
Single-Event Upset (SEU) per **JEDEC JESD89A**: single bit-flip at a
randomly selected internal gate output for one evaluation cycle.

### Scale
- 3 schemes × 3 key sizes × 20 placements × 10 circuits = **1,800 locked variants**
- 10,000 Monte Carlo trials per variant = **18 million fault injection trials**
- Per-gate SDC labeling: 2,000 trials × 9,677 gates = **~19 million additional trials**

---

## ML Framework

### Features (15 structural + 9 engineered = 24 total)

**Structural:**
`fan_in, fan_out, level, level_norm, cone_size, output_reach,
signal_prob, sp_skew, is_xor, is_xnor, is_and, is_nand, is_or,
is_nor, is_not`

**Engineered interactions:**
`cone_x_reach, sp_x_fanout, level_x_reach, cone_x_sp,
reach_x_spskew, fanout_x_level, cone_sq, reach_sq, sp_centered`

### Model
**LightGBM** gradient-boosted regressor.
Target: per-gate delta SDC contribution when used as a lock site.
Validation: **Leave-One-Circuit-Out cross-validation** (LOCO-CV).

### Top Predictive Features
1. Signal Probability
2. Output Reach
3. Normalized Topological Level
4. Fanin Cone Size

---

## Reproducing Results

### Requirements

```bash
python3 -m venv env
source env/bin/activate
pip install circuitgraph networkx matplotlib numpy \
            scikit-learn lightgbm joblib tqdm pandas xgboost
```

### Run Full Experiment

```bash
python3 simulation/locked_but_leaky_full.py
```

Expected runtime: **25–40 minutes** on a modern multi-core CPU.
All results saved to `./results/` and figures to `./figures/`.

### Retrain ML Only (after full experiment)

```bash
python3 ml/retrain_ml.py
```

Reads existing `results/gate_features.csv` — no re-simulation needed.
Runtime: **under 5 minutes**.

### Regenerate Figures at 600 DPI

```bash
python3 simulation/regenerate_figures.py
```

---

## Results Summary

### Baseline SDC Rates (Unlocked Circuits)

| Circuit | Gates | SDC Rate |
|---|---|---|
| c432  | 171   | 27.9% |
| c499  | 174   | 54.6% |
| c880  | 323   | 51.5% |
| c1355 | 518   | 44.6% |
| c1908 | 479   | 53.4% |
| c2670 | 699   | 41.0% |
| c3540 | 1043  | 35.9% |
| c5315 | 1586  | 38.6% |
| c6288 | 2353  | 89.7% |
| c7552 | 2331  | 39.7% |

### ML-Guided Placement vs. Baselines (K=16, XOR Locking)

| Circuit | Random | Sec.-Greedy | ML-Guided | Reduction |
|---|---|---|---|---|
| c432  | 24.4% | 25.9% | **19.4%** | 20.6% |
| c499  | 62.7% | 54.4% | **46.0%** | 26.5% |
| c880  | 44.9% | 45.5% | **36.5%** | 18.7% |
| c1355 | 46.0% | 44.5% | **41.8%** |  9.2% |
| c1908 | 49.8% | 50.5% | **43.1%** | 13.4% |
| c2670 | 38.0% | 42.5% | **37.2%** |  2.3% |
| c3540 | 34.5% | 34.4% | **32.4%** |  6.0% |
| c5315 | 37.9% | 38.0% | **37.9%** |  0.0% |
| c6288 | 87.7% | 89.3% | **81.0%** |  7.7% |
| c7552 | 38.1% | 37.9% | **35.6%** |  6.4% |
| **Avg.** | -- | -- | -- | **11.1%** |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{rethinapandian2026logiclock,
  title     = {Logic Locking and Soft-Error Vulnerability: An ML-Guided
               Framework for Reliability-Aware Key-Gate Placement in
               VLSI Circuits},
  author    = {Rethinapandian, Yogesh and Sundararajan, Arun Karthik
               and Kumar, Kaushik and Prakash, Smrithi},
  year      = {2026}
}
```

---

## License

This repository is released for academic research purposes.
All simulation code is original work by the authors.
ISCAS-85 benchmark circuits are public domain.

---

## Contact

**Yogesh Rethinapandian** (Corresponding Author)
Department of Electrical and Computer Engineering
University of Illinois Chicago
yrethi2@uic.edu · ORCID: 0009-0000-6111-857X
