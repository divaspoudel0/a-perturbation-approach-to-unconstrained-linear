#!/usr/bin/env python3
"""
Run a single claim experiment.

Usage:
    python experiments/run_single.py <claim_number> [--quick]
    python experiments/run_single.py 1

Claims:
    1  - PABLO reduces uBLO to OLO
    2  - Static-regret bound (Theorem 3.1)
    3  - Dynamic-regret bound (Theorem 3.3)
    4  - High-probability bound (Theorem 4.2)
    5  - Lower bound on the unit ball (Theorem 5.2)
    6  - Norm-oblivious minimax rate (Conjecture 5.3)
"""
import sys
sys.path.insert(0, "src")

import argparse
from pathlib import Path
from pablo.experiments import run_claim1, run_claim2, run_claim3, run_claim4, run_claim5, run_claim6


CLAIMS = {
    "1": ("claim1", run_claim1),
    "2": ("claim2", run_claim2),
    "3": ("claim3", run_claim3),
    "4": ("claim4", run_claim4),
    "5": ("claim5", run_claim5),
    "6": ("claim6", run_claim6),
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run a single PABLO reproduction experiment")
    parser.add_argument("claim", type=str, choices=list(CLAIMS.keys()), help="Claim number to run")
    args = parser.parse_args()

    claim_id, runner = CLAIMS[args.claim]
    print(f"Running {claim_id}...")
    result = runner(save_path=str(RESULTS_DIR / f"{claim_id}_results.json"))
    print(f"Done. Results saved to {RESULTS_DIR / f'{claim_id}_results.json'}")


if __name__ == "__main__":
    main()
