"""
Online Linear Optimization (OLO) algorithms for use with PABLO.
"""
import numpy as np


class DefaultOLO:
    """Simple scale-free gradient descent OLO learner (for testing)."""
    def __init__(self, config):
        self.d = config.d
        self.w = np.zeros(config.d)
        self.t = 0

    def get_decision(self):
        return self.w.copy()

    def update(self, gradient):
        eta = 1.0 / (np.linalg.norm(gradient) + 1e-8)
        self.w = self.w - eta * gradient
        self.t += 1


class ComparatorAdaptiveOLO:
    """
    Comparator-adaptive OLO subroutine for PABLO, based on
    parameter-free online learning (Cutkosky & Orabona, 2018).
    This is the norm-oblivious variant (kappa = sqrt(d)).
    """
    def __init__(self, config):
        self.d = config.d
        self.G = config.G
        self.epsilon = config.epsilon
        self.w = np.zeros(config.d)
        self.V_t = 0.0

    def get_decision(self):
        return self.w.copy()

    def update(self, gradient):
        g_norm = np.linalg.norm(gradient)
        self.V_t += g_norm ** 2
        eta = min(1.0, self.epsilon / (np.sqrt(self.V_t) + 1e-8))
        self.w = self.w - eta * gradient


class NormAdaptiveOLO:
    """
    OLO that adapts to the comparator norm adaptively.
    Uses a scale-adaptive learning rate (kappa = 1).
    """
    def __init__(self, config):
        self.d = config.d
        self.G = config.G
        self.epsilon = config.epsilon
        self.w = np.zeros(config.d)
        self.V_t = 0.0

    def get_decision(self):
        return self.w.copy()

    def update(self, gradient):
        g_norm = np.linalg.norm(gradient)
        self.V_t += g_norm ** 2
        w_norm = np.linalg.norm(self.w) + 1e-8
        eta = min(1.0 / w_norm, self.epsilon / (np.sqrt(self.V_t) + 1e-8))
        self.w = self.w - eta * gradient
