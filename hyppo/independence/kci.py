import numpy as np
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from .base import IndependenceTest, IndependenceTestOutput


class KCI(IndependenceTest):
    def __init__(self, **kwargs):

        IndependenceTest.__init__(self, **kwargs)

    def statistic(self, x, y, width):
        T = len(y)

        x = np.array(x)
        y = np.array(y)
        x = x - np.mean(x)
        x = x / np.std(x)
        y = y - np.mean(y)
        y = y / np.std(y)

        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

        theta = 1 / (width ^ 2)

        H = np.eye(T) - np.ones(T, T) / T

        kernel = 1.0 * RBF(theta)

        Kx = GaussianProcessClassifier(kernel=kernel).fit(x, x)
        Kx = np.matmul(np.matmul(H, Kx), H)
        Ky = GaussianProcessClassifier(kernel=kernel).fit(y, y)
        Ky = np.matmul(np.matmul(H, Ky), H)
        Stat = np.trace(np.matmul(Kx * Ky))

        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.trace(Kx * Kx) * np.trace(Ky * Ky) / T ^ 2
        k_appr = mean_appr ^ 2 / var_appr
        theta_appr = var_appr / mean_appr
        p_val = 1 - gamma.cdf(Stat, k_appr, theta_appr)

        self.stat = Stat

        return IndependenceTestOutput(stat, pvalue)
