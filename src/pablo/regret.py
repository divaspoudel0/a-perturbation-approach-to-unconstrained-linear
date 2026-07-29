"""
Regret computation utilities.
"""
import numpy as np


def compute_static_regret(decisions, losses, comparator):
    """
    Static regret: R_T(u) = sum_t <ell_t, w_t - u>

    Args:
        decisions: List of actions taken by the learner (length T)
        losses: List of loss vectors (length T)
        comparator: Fixed comparator vector u

    Returns:
        Static regret value
    """
    T = len(decisions)
    learner_loss = sum(np.dot(losses[t], decisions[t]) for t in range(T))
    comparator_loss = sum(np.dot(losses[t], comparator) for t in range(T))
    return learner_loss - comparator_loss


def compute_dynamic_regret(decisions, losses, comparators):
    """
    Dynamic regret: R_T(u_{1:T}) = sum_t <ell_t, w_t - u_t>

    Args:
        decisions: List of actions taken by the learner (length T)
        losses: List of loss vectors (length T)
        comparators: List of comparator vectors (length T)

    Returns:
        Dynamic regret value
    """
    T = len(decisions)
    learner_loss = sum(np.dot(losses[t], decisions[t]) for t in range(T))
    comparator_loss = sum(np.dot(losses[t], comparators[t]) for t in range(T))
    return learner_loss - comparator_loss


def compute_path_length(comparators):
    """
    Path length: P_T = sum_{t=2}^T ||u_t - u_{t-1}||

    Args:
        comparators: List of comparator vectors (length T)

    Returns:
        Path length value
    """
    P = 0.0
    for t in range(1, len(comparators)):
        P += np.linalg.norm(comparators[t] - comparators[t-1])
    return P


def compute_effective_diameter(comparators):
    """
    Effective diameter: M = max_t ||u_t||

    Args:
        comparators: List of comparator vectors (length T)

    Returns:
        Effective diameter
    """
    return max(np.linalg.norm(u) for u in comparators)


def generate_comparator_sequence(d, T, path_length_target=0, base_norm=3.0, seed=42):
    """
    Generate a sequence of comparators with a given path length.

    Args:
        d: Dimension
        T: Horizon
        path_length_target: Target path length P_T
        base_norm: Initial norm of comparator
        seed: Random seed

    Returns:
        comparators: List of T comparator vectors
        actual_P: Actual path length
        M: Effective diameter
    """
    rng = np.random.RandomState(seed)
    comparators = []
    current = np.ones(d) / np.sqrt(d) * base_norm
    comparators.append(current.copy())
    for t in range(1, T):
        if path_length_target > 0:
            step_dir = rng.randn(d)
            step_dir = step_dir / (np.linalg.norm(step_dir) + 1e-12)
            current = current + step_dir * (path_length_target / T)
        comparators.append(current.copy())

    actual_P = compute_path_length(comparators)
    M = compute_effective_diameter(comparators)
    return comparators, actual_P, M
