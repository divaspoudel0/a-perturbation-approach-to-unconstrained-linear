"""
PABLO Algorithm implementation (Algorithm 1 from the paper).
"""
import numpy as np
from dataclasses import dataclass
from .olo import DefaultOLO


@dataclass
class PABLOConfig:
    d: int
    T: int
    G: float = 1.0
    epsilon: float = 1.0
    norm_adaptive: bool = False
    seed: int = 42


class PABLO:
    """
    PABLO: Perturbation Approach for Bandit Linear Optimization.
    Algorithm 1 from Jacobsen et al. (ICML 2026).

    Args:
        config: Configuration parameters
        olo_learner: An OLO algorithm instance (must have get_decision() and update(gradient))
    """
    def __init__(self, config: PABLOConfig, olo_learner=None):
        self.config = config
        self.olo = olo_learner if olo_learner else DefaultOLO(config)
        self.rng = np.random.RandomState(config.seed)
        self.w_t = np.zeros(config.d)
        self.loss_estimates = []
        self.decisions = []

    def get_perturbation_matrix(self, w_t):
        """
        Choose H_t according to Corollary 2.2:
            H_t = I_d / (d * (||w_t||^2 v epsilon^2))
        """
        d = self.config.d
        eps = self.config.epsilon
        norm_sq = np.dot(w_t, w_t)
        scale = 1.0 / (d * max(norm_sq, eps ** 2))
        return np.eye(d) * scale

    def step(self, loss_vector):
        """
        One round of PABLO.

        Args:
            loss_vector: The true loss vector ell_t (not observed directly by learner)

        Returns:
            tilde_w_t: The perturbed action played
            loss_observed: The scalar bandit feedback <ell_t, tilde_w_t>
            tilde_ell_t: The unbiased loss estimate fed to the OLO learner
        """
        d = self.config.d

        # Step 1: Get OLO decision
        self.w_t = self.olo.get_decision()

        # Step 2: Choose perturbation matrix
        H_t = self.get_perturbation_matrix(self.w_t)
        eigvals, eigvecs = np.linalg.eigh(H_t)

        # Step 3: Sample s_t uniformly from {±v_i}
        idx = self.rng.randint(0, d)
        sign = self.rng.choice([-1, 1])
        s_t = sign * eigvecs[:, idx]
        lambda_t = eigvals[idx]

        # Step 4: Play perturbed action
        H_t_sqrt = np.linalg.cholesky(H_t + 1e-12 * np.eye(d))
        H_t_inv_sqrt = np.linalg.inv(H_t_sqrt)
        tilde_w_t = self.w_t + H_t_inv_sqrt @ s_t

        # Step 5: Observe bandit feedback
        loss_observed = np.dot(loss_vector, tilde_w_t)

        # Step 6: Construct unbiased loss estimate
        tilde_ell_t = d * H_t_sqrt @ s_t * loss_observed

        # Step 7: Pass estimate to OLO learner
        self.olo.update(tilde_ell_t)
        self.loss_estimates.append(tilde_ell_t)
        self.decisions.append(tilde_w_t)

        return tilde_w_t, loss_observed, tilde_ell_t

    def reset(self):
        """Reset the algorithm state."""
        self.w_t = np.zeros(self.config.d)
        self.loss_estimates = []
        self.decisions = []
        self.rng = np.random.RandomState(self.config.seed)
        self.olo = DefaultOLO(self.config)

    def run(self, losses):
        """
        Run PABLO over a sequence of loss vectors.

        Args:
            losses: List of loss vectors of length T

        Returns:
            decisions: List of actions taken (length T)
        """
        self.decisions = []
        for t, ell_t in enumerate(losses):
            tilde_w_t, _, _ = self.step(ell_t)
        return self.decisions
