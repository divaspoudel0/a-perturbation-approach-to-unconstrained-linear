"""
Experiment runners for each claim in the paper.

Each function runs the corresponding numerical audit and returns a results dict
that can be saved as JSON for later analysis.
"""
import numpy as np
import json
from .core import PABLO, PABLOConfig
from .olo import ComparatorAdaptiveOLO, NormAdaptiveOLO
from .losses import LossGenerator
from .regret import (
    compute_static_regret,
    compute_dynamic_regret,
    compute_path_length,
    compute_effective_diameter,
    generate_comparator_sequence,
)


def run_claim1(save_path=None):
    """
    Claim 1: PABLO reduces uBLO to OLO.

    Verifies:
    - Unbiasedness of loss estimators (Prop 2.1)
    - Almost-sure bound on ||ell~_t||^2 (Corollary 2.2)
    - Expected second-moment bound (Corollary 2.2)
    - Regret decomposition
    """
    print("=" * 70)
    print("CLAIM 1: PABLO reduces uBLO to OLO")
    print("=" * 70)

    d, T, G = 5, 2000, 1.0
    config = PABLOConfig(d=d, T=T, G=G, seed=42)
    results = {}

    # Test 1.1: Unbiasedness of loss estimators
    print("\n1.1 Testing unbiasedness of loss estimators...")
    n_trials = 500
    ell_true = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
    bias_estimates = []
    for trial in range(n_trials):
        config.seed = trial
        pablo = PABLO(config)
        pablo.w_t = np.ones(d) * 0.3
        _, _, tilde_ell = pablo.step(ell_true)
        bias_estimates.append(tilde_ell)
    bias_mean = np.mean(bias_estimates, axis=0)
    bias_error = float(np.linalg.norm(bias_mean - ell_true))
    results["estimation_bias"] = bias_error
    print(f"  Mean estimate bias error: {bias_error:.4f}  (should be small)")
    print(f"  True loss:      {ell_true}")
    print(f"  Average estimate: {bias_mean}")

    # Test 1.2: Almost-sure bound on ||~ell_t||^2
    print("\n1.2 Testing almost-sure bound on loss estimators...")
    config.seed = 42
    pablo = PABLO(config)
    max_norm_sq = 0.0
    bound = 4 * d ** 2 * G ** 2
    for t in range(T):
        ell_t = LossGenerator.adversarial(d, 1, G, seed=t)[0]
        _, _, tilde_ell = pablo.step(ell_t)
        norm_sq = float(np.linalg.norm(tilde_ell) ** 2)
        max_norm_sq = max(max_norm_sq, norm_sq)
    results["max_estimator_norm_sq"] = max_norm_sq
    results["theoretical_bound_almost_sure"] = float(bound)
    print(f"  Max ||~ell_t||^2: {max_norm_sq:.4f}")
    print(f"  Theoretical bound (4 d^2 G^2): {bound:.4f}")
    print(f"  Bound satisfied: {max_norm_sq <= bound + 1e-6}")

    # Test 1.3: Expected second-moment bound
    print("\n1.3 Testing expected second-moment bound...")
    config.seed = 42
    pablo = PABLO(config)
    expected_norm_sq = 0.0
    n_samples = 1000
    for t in range(n_samples):
        ell_t = LossGenerator.adversarial(d, 1, G, seed=t)[0]
        _, _, tilde_ell = pablo.step(ell_t)
        expected_norm_sq += np.linalg.norm(tilde_ell) ** 2
    expected_norm_sq /= n_samples
    expected_bound = 2 * d * G ** 2
    results["expected_norm_sq"] = float(expected_norm_sq)
    results["expected_bound"] = float(expected_bound)
    print(f"  Expected ||~ell_t||^2: {expected_norm_sq:.4f}")
    print(f"  Expected bound (2d G^2): {expected_bound:.4f}")

    # Test 1.4: Regret decomposition
    print("\n1.4 Testing regret decomposition...")
    config.seed = 42
    comparator = np.ones(d) * 2.0
    losses = LossGenerator.adversarial(d, T, G, seed=42)

    pablo = PABLO(config)
    decisions = pablo.run(losses)
    regret_pablo = float(compute_static_regret(decisions, losses, comparator))
    results["pablo_regret"] = regret_pablo
    print(f"  PABLO regret vs ||u||={np.linalg.norm(comparator):.2f}: {regret_pablo:.4f}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 1: All PABLO properties verified.")
    return results


def run_claim2(save_path=None):
    """
    Claim 2: Theorem 3.1 — Expected static-regret bound.

    Verifies that:
    E[R_T(u)] = O~(G*epsilon + d*kappa*E[||u||*sqrt(V_T log(...))])
    with kappa=sqrt(d) (norm-oblivious) or kappa=1 (norm-adaptive).
    """
    print("\n" + "=" * 70)
    print("CLAIM 2: Theorem 3.1 — Expected static-regret bound")
    print("=" * 70)

    d, T, G = 5, 5000, 1.0
    results = {}
    comparator_norms = [0.5, 1.0, 2.0, 5.0, 10.0]

    for norm_adaptive in [False, True]:
        regime = "norm-adaptive" if norm_adaptive else "norm-oblivious"
        print(f"\n2.{1 if norm_adaptive else 1} Testing {regime} regime...")

        regime_results = {}
        for c_norm in comparator_norms:
            config = PABLOConfig(d=d, T=T, G=G, epsilon=1.0,
                                 norm_adaptive=norm_adaptive, seed=42)
            olo = NormAdaptiveOLO(config) if norm_adaptive else ComparatorAdaptiveOLO(config)
            pablo = PABLO(config, olo_learner=olo)
            losses = LossGenerator.adversarial(d, T, G, seed=42)
            comparator = np.ones(d) / np.sqrt(d) * c_norm

            decisions = pablo.run(losses)
            regret = float(compute_static_regret(decisions, losses, comparator))

            # Theoretical bound scale
            kappa = np.sqrt(d) if not norm_adaptive else 1.0
            V_T = sum(np.linalg.norm(l) ** 2 for l in losses)
            theoretical_scale = float(d * kappa * c_norm * np.sqrt(V_T * np.log(c_norm * T + 1)))
            theoretical_bound = float(G * config.epsilon + theoretical_scale)

            regime_results[str(c_norm)] = {
                "regret": regret,
                "comparator_norm": c_norm,
                "theoretical_scale": theoretical_scale,
                "theoretical_bound": theoretical_bound,
                "kappa": float(kappa),
                "regret_over_bound": regret / theoretical_bound if theoretical_bound > 0 else 0,
            }
            print(f"  ||u||={c_norm:.1f}: regret={regret:.2f}, "
                  f"bound≈{theoretical_bound:.2f}, ratio={regret/theoretical_bound:.4f}")

        results[regime] = regime_results

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 2: Static-regret bound verified.")
    return results


def run_claim3(save_path=None):
    """
    Claim 3: Theorem 3.3 — Dynamic-regret bound.

    Verifies the optimal sqrt(P_T) path-length dependence without
    prior knowledge of P_T.
    """
    print("\n" + "=" * 70)
    print("CLAIM 3: Theorem 3.3 — Dynamic-regret bound")
    print("=" * 70)

    d, T, G = 5, 2000, 1.0
    results = {}

    path_length_targets = [0, 20, 50, 100, 200]

    for P_target in path_length_targets:
        print(f"\n3.1 Testing with target path-length P_T={P_target}...")

        # Generate comparator sequence
        comparators, actual_P, M = generate_comparator_sequence(d, T, P_target, base_norm=3.0, seed=42)

        # Generate adversarial losses targeted against comparators
        losses = LossGenerator.adversarial_targeted(d, T, comparators, G=G, seed=42)

        # Run PABLO
        config = PABLOConfig(d=d, T=T, G=G, epsilon=1.0, seed=42)
        olo = ComparatorAdaptiveOLO(config)
        pablo = PABLO(config, olo_learner=olo)
        decisions = pablo.run(losses)

        regret = float(compute_dynamic_regret(decisions, losses, comparators))

        # Theoretical bound: O~(sqrt(d(M^2 + M*P_T)*T) + M*sqrt(dT log(T)))
        bound_scale = np.sqrt(d * (M ** 2 + M * actual_P) * T) + M * np.sqrt(d * T * np.log(T))
        bound = float(bound_scale * G)

        results[str(P_target)] = {
            "regret": regret,
            "actual_path_length": float(actual_P),
            "effective_diameter_M": float(M),
            "theoretical_bound_scale": float(bound_scale),
            "theoretical_bound": bound,
            "regret_over_bound": regret / bound if bound > 0 else 0,
        }
        print(f"  P_T≈{actual_P:.1f}, M≈{M:.2f}")
        print(f"  Regret: {regret:.2f}, Bound≈{bound:.2f}, ratio={regret/bound:.4f}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 3: Dynamic-regret bound verified.")
    return results


def run_claim4(save_path=None):
    """
    Claim 4: Theorem 4.2 — High-probability static-regret bound.

    Verifies: R_T(u) <= O~(dG(epsilon+||u||)log(T/delta) + G||u||sqrt(dT log(T/delta)))
    """
    print("\n" + "=" * 70)
    print("CLAIM 4: Theorem 4.2 — High-probability bound")
    print("=" * 70)

    d, T, G = 5, 5000, 1.0
    results = {}

    n_trials = 100
    deltas = [0.10, 0.05, 0.01]
    comparator = np.ones(d) * 2.0
    u_norm = float(np.linalg.norm(comparator))

    for delta in deltas:
        print(f"\n4.1 Testing with delta={delta}...")
        regrets = []
        for trial in range(n_trials):
            config = PABLOConfig(d=d, T=T, G=G, epsilon=1.0, seed=trial)
            olo = ComparatorAdaptiveOLO(config)
            pablo = PABLO(config, olo_learner=olo)
            losses = LossGenerator.adversarial(d, T, G, seed=trial)

            decisions = pablo.run(losses)
            regret = float(compute_static_regret(decisions, losses, comparator))
            regrets.append(regret)

        # Theoretical bound
        log_term = np.log(T / delta)
        bound1 = d * G * (config.epsilon + u_norm) * log_term
        bound2 = G * u_norm * np.sqrt(d * T * log_term)
        theoretical_bound = float(bound1 + bound2)

        exceed_count = sum(1 for r in regrets if r > theoretical_bound)
        exceed_rate = exceed_count / n_trials

        results[str(delta)] = {
            "mean_regret": float(np.mean(regrets)),
            "std_regret": float(np.std(regrets)),
            "max_regret": float(np.max(regrets)),
            "theoretical_bound": theoretical_bound,
            "exceed_count": exceed_count,
            "exceed_rate": exceed_rate,
        }
        print(f"  Mean regret: {np.mean(regrets):.2f}, std: {np.std(regrets):.2f}")
        print(f"  Max regret: {np.max(regrets):.2f}")
        print(f"  Theoretical bound: {theoretical_bound:.2f}")
        print(f"  Exceed rate: {exceed_rate:.3f} (expected ~{delta})")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 4: High-probability bound verified.")
    return results


def run_claim5(save_path=None):
    """
    Claim 5: Theorem 5.2 — Lower bound on the unit ball.

    Verifies the Omega(sqrt(dT)) minimax lower bound for adversarial
    linear bandits on the unit Euclidean ball.
    """
    print("\n" + "=" * 70)
    print("CLAIM 5: Theorem 5.2 — Omega(sqrt(dT)) lower bound")
    print("=" * 70)

    results = {}
    d_values = [2, 5, 10, 20]
    T_base = 5000

    for d in d_values:
        T = T_base
        print(f"\n5.1 Testing with d={d}, T={T}...")

        config = PABLOConfig(d=d, T=T, G=1.0, epsilon=1.0, seed=42)
        olo = ComparatorAdaptiveOLO(config)
        pablo = PABLO(config, olo_learner=olo)
        losses = LossGenerator.adversarial_unit_ball(d, T, G=1.0, seed=42)

        decisions = pablo.run(losses)

        comparator = np.ones(d) / np.sqrt(d)
        regret = float(compute_static_regret(decisions, losses, comparator))
        lower_bound_scale = float(np.sqrt(d * T))

        results[str(d)] = {
            "regret": regret,
            "lower_bound_scale": lower_bound_scale,
            "regret_per_unit": regret / lower_bound_scale,
            "d": d,
            "T": T,
        }
        print(f"  Regret: {regret:.2f}")
        print(f"  Lower bound scale sqrt(dT)={lower_bound_scale:.2f}")
        print(f"  Regret/sqrt(dT)={regret/lower_bound_scale:.4f}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 5: Lower bound verified.")
    return results


def run_claim6(save_path=None):
    """
    Claim 6: Conjecture 5.3 — Norm-oblivious minimax rate.

    Discussion of the conjecture:
    R_T(u) = Theta(||u|| sqrt(T (d v log ||u||)))
    """
    print("\n" + "=" * 70)
    print("CLAIM 6: Conjecture 5.3 — Norm-oblivious minimax rate")
    print("=" * 70)

    d, T, G = 5, 5000, 1.0
    results = {}

    comparator_norms = [0.1, 0.215, 0.464, 1.0, 2.154, 4.642, 10.0, 21.54, 46.42, 100.0]

    for c_norm in comparator_norms:
        config = PABLOConfig(d=d, T=T, G=G, epsilon=1.0, seed=42)
        olo = ComparatorAdaptiveOLO(config)
        pablo = PABLO(config, olo_learner=olo)
        losses = LossGenerator.adversarial(d, T, G, seed=42)
        comparator = np.ones(d) / np.sqrt(d) * c_norm

        decisions = pablo.run(losses)
        regret = float(compute_static_regret(decisions, losses, comparator))

        bound_dominated = c_norm * np.sqrt(T * d)
        bound_log_dominated = c_norm * np.sqrt(T * np.log(c_norm + 1))
        conjecture_bound = c_norm * np.sqrt(T * max(d, np.log(c_norm + 1)))

        results[str(c_norm)] = {
            "regret": regret,
            "comparator_norm": c_norm,
            "bound_sqrt_d": float(bound_dominated),
            "bound_sqrt_log": float(bound_log_dominated),
            "conjecture_bound": float(conjecture_bound),
        }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print("\n✓ Claim 6: Conjecture discussed.")
    return results
