import numpy as np
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from .base import IndependenceTest, IndependenceTestOutput


class KCI(IndependenceTest):
    def __init__(self, **kwargs):

        IndependenceTest.__init__(self, **kwargs)

    def kernel(self, x, y):

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

        theta = 1 / (width**2)

        Kx = 1.0 * RBF(theta).__call__(x, x)
        Ky = 1.0 * RBF(theta).__call__(y, y)

        return Kx, Ky

    def statistic(self, x, y):

        T = len(y)

        H = np.eye(T) - np.ones((T, T)) / T

        Kx, Ky = self.kernel(x, y)

        Kx = np.matmul(np.matmul(H, Kx), H)
        Ky = np.matmul(np.matmul(H, Ky), H)

        stat = np.trace(np.matmul(Kx, Ky))

        return stat

    def test(self, x, y):

        T = len(y)

        Kx, Ky = self.kernel(x, y)
        stat = self.statistic(x, y)

        mean_appr = (np.trace(Kx) * np.trace(Ky)) / T
        var_appr = (
            2 * np.trace(np.matmul(Kx, Kx)) * np.trace(np.matmul(Ky, Ky)) / T**2
        )
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        pvalue = 1 - np.mean(gamma.cdf(stat, k_appr, theta_appr))

        self.stat = stat

        return IndependenceTestOutput(stat, pvalue)
