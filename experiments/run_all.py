#!/usr/bin/env python3
"""
Run all reproduction experiments for the PABLO paper.

Usage:
    python experiments/run_all.py [--quick]

Flags:
    --quick     Run with reduced parameters for faster execution
"""
import sys
sys.path.insert(0, "src")

import json
import argparse
import time
from pathlib import Path
from pablo.experiments import run_claim1, run_claim2, run_claim3, run_claim4, run_claim5, run_claim6


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run all PABLO reproduction experiments")
    parser.add_argument("--quick", action="store_true", help="Run with reduced parameters")
    args = parser.parse_args()

    experiments = [
        ("claim1", "estimation_and_regret_properties", run_claim1),
        ("claim2", "static_regret_bound", run_claim2),
        ("claim3", "dynamic_regret_bound", run_claim3),
        ("claim4", "high_probability_bound", run_claim4),
        ("claim5", "lower_bound_unit_ball", run_claim5),
        ("claim6", "norm_oblivious_minimax", run_claim6),
    ]

    for claim_id, name, runner in experiments:
        print(f"\n{'#' * 70}")
        print(f"# Running {claim_id}: {name}")
        print(f"{'#' * 70}")
        start = time.time()
        result = runner(save_path=str(RESULTS_DIR / f"{claim_id}_results.json"))
        elapsed = time.time() - start
        print(f"\n[{claim_id}] Completed in {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to {RESULTS_DIR.resolve()}")
    for f in sorted(RESULTS_DIR.glob("*.json")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
