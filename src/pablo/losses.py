"""
Loss vector generators for bandit linear optimization experiments.
"""
import numpy as np


class LossGenerator:
    """Generate loss vectors for bandit linear optimization."""

    @staticmethod
    def adversarial(d, T, G=1.0, seed=42):
        """Generate adversarial losses (uniform on sphere of radius G)."""
        rng = np.random.RandomState(seed)
        losses = []
        for t in range(T):
            ell = rng.randn(d)
            ell_norm = np.linalg.norm(ell)
            if ell_norm > 1e-12:
                ell = ell / ell_norm * G
            else:
                ell = np.zeros(d)
                ell[0] = G
            losses.append(ell)
        return losses

    @staticmethod
    def stochastic(d, T, theta=None, sigma=0.1, G=1.0, seed=42):
        """Generate stochastic losses with mean theta."""
        rng = np.random.RandomState(seed)
        if theta is None:
            theta = rng.randn(d)
            theta = theta / (np.linalg.norm(theta) + 1e-12) * 0.5
        losses = []
        for t in range(T):
            noise = rng.randn(d) * sigma
            ell = theta + noise
            ell = np.clip(ell, -G, G)
            losses.append(ell)
        return losses

    @staticmethod
    def adversarial_unit_ball(d, T, G=1.0, seed=42):
        """
        Adversarial setting for Thm 5.2 lower bound on the unit ball.
        Uses Rademacher-like losses along coordinate directions.
        """
        rng = np.random.RandomState(seed)
        losses = []
        for t in range(T):
            i = rng.randint(0, d)
            sigma = rng.choice([-1, 1])
            ell = np.zeros(d)
            ell[i] = sigma * G
            losses.append(ell)
        return losses

    @staticmethod
    def correlated(d, T, theta, strength=0.3, G=1.0, seed=42):
        """Correlated losses (signal + noise) for dynamic regret experiments."""
        rng = np.random.RandomState(seed)
        losses = []
        for t in range(T):
            ell = strength * theta + (1 - strength) * rng.randn(d)
            ell = np.clip(ell, -G, G)
            losses.append(ell)
        return losses

    @staticmethod
    def adversarial_targeted(d, T, comparator_sequence, G=1.0, seed=42):
        """
        Adversarial losses that reward the comparator and punish the learner.
        Used for dynamic-regret experiments.
        """
        rng = np.random.RandomState(seed)
        losses = []
        for t in range(T):
            u = comparator_sequence[t]
            u_norm = np.linalg.norm(u)
            if u_norm > 1e-12:
                dir_vec = u / u_norm
            else:
                dir_vec = np.zeros(d)
                dir_vec[0] = 1.0
            ell = -dir_vec * G * 0.8 + 0.2 * rng.randn(d)
            ell_norm = np.linalg.norm(ell)
            if ell_norm > 1e-12:
                ell = ell / ell_norm * G
            else:
                ell = np.zeros(d)
                ell[0] = G
            losses.append(ell)
        return losses
