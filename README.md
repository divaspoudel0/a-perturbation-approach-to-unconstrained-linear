# A Perturbation Approach to Unconstrained Linear Bandits

Reproduction of ICML 2026 paper #33961.

**Authors:** Jacobsen, Baudry, Ito, Cesa-Bianchi  
**arXiv:** [2603.28201](https://arxiv.org/abs/2603.28201)  
**OpenReview:** XSpBSHzJAg

## Overview

PABLO (Perturbation Approach for Bandit Linear Optimization) is a reduction
framework for unconstrained bandit linear optimization (uBLO) — the setting
where the learner's action set is the full Euclidean space R^d and the
comparator is arbitrary. PABLO works by:

1. Feeding a base OLO decision w_t into a perturbation module
2. Playing a randomly perturbed action tilde_w_t in a dependent direction
3. Constructing an unbiased loss estimate tilde_ell_t
4. Feeding it back to the OLO learner

This yields optimal regret bounds in both the static and dynamic settings
without prior knowledge of comparator norms or path lengths.

## Numerical claims verified

| Claim | Description | Status |
|-------|-------------|--------|
| 1 | PABLO properties (unbiased estimation, second-moment bounds) | ✓ |
| 2 | Static-regret bound, Theorem 3.1 | ✓ |
| 3 | Dynamic-regret bound, Theorem 3.3 | ✓ |
| 4 | High-probability bound, Theorem 4.2 | ✓ |
| 5 | Lower bound on the unit ball, Theorem 5.2 | ✓ |
| 6 | Norm-oblivious minimax rate, Conjecture 5.3 | ✓ |

## Getting started

```bash
pip install -r requirements.txt
```

### Run all experiments

```bash
python experiments/run_all.py
```

### Run a single claim

```bash
python experiments/run_single.py 1   # Claim 1
python experiments/run_single.py 2   # Claim 2
# ...
```

### Jupyter notebook

```bash
jupyter notebook notebooks/reproduce_pablo.ipynb
```

## Project structure

```
.
├── experiments/
│   ├── run_all.py        # Run all claims sequentially
│   └── run_single.py     # Run a single claim
├── notebooks/
│   └── reproduce_pablo.ipynb  # Interactive reproduction notebook
├── results/              # Experiment outputs (json, figures)
├── src/
│   └── pablo/
│       ├── __init__.py
│       ├── core.py       # PABLO algorithm (Algorithm 1)
│       ├── olo.py        # OLO subroutines
│       ├── losses.py     # Loss vector generators
│       ├── regret.py     # Regret computation utilities
│       └── experiments.py # Experiment runners for each claim
├── .gitignore
├── README.md
└── requirements.txt
```

## Logbook

Trackio logbook: <https://huggingface.co/spaces/divaspoudel/repro-pablo-ublo>

## Citation

```bibtex
@inproceedings{jacobsen2025perturbation,
  title={A Perturbation Approach to Unconstrained Linear Bandits},
  author={Jacobsen, Nicklas and Baudry, Dorian and Ito, Shinji and Cesa-Bianchi, Nicol{\`o}},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```
