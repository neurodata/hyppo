import numpy as np
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class KCI(ConditionalIndependenceTest):
    r"""
    Kernel Conditional Independence Test Statistic and P-Value.

    This is a conditional indpendence test utilizing a radial basis 
    function to calculate the kernels of two datasets. The trace
    of the normalized matrix product is then calculated to extract the test 
    statistic. A Gaussian distribution is then utilized to calculate
    the p-value given the statistic and approximate mean and variance
    of the trace values of the independent kernel matrices.
    This test is consistent against similar tests.

    Notes
    -----
    Let :math:`x` be a combined sample of :math:`(n, p)` sample
    of random variables :math:`X` and let :math:`y` be a :math:`(n, 1)`
    labels of sample classes :math:`Y`. We can then generate
    :math:`Kx` and :math:`Ky` kernel matrices for each of the respective
    samples. Normalizing, multiplying, and taking the trace of these
    kernel matrices gives the resulting test statistic.
    The p-value and null distribution for the corrected statistic are calculated a
    gamma distribution approximation.
    """

    def __init__(self, **kwargs):

        ConditionalIndependenceTest.__init__(self, **kwargs)

    def compute_kern(self, x, y):

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

        Kx, Ky = self.compute_kern(x, y)

        Kx = (H @ Kx) @ H
        Ky = (H @ Ky) @ H

        stat = np.trace(Kx @ Ky)

        return stat

    def test(self, x, y):
        r"""
        Calculates the Kernel Conditional Independence test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            columns. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the dimension of samples and `p` is the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed Kernel Conditional Independence statistic.
        pvalue : float
            The computed Kernel Conditional Independence p-value.

        Example
        --------
        >>> from hyppo.conditional import KCI
        >>> from hyppo.tools.indep_sim import linear
        >>> np.random.seed(123456789)
        >>> x, y = linear(n, 1)
        >>> stat, pvalue = KCI().test(x, y)
        >>> print("Statistic: ", stat)
        >>> print("p-value: ", pvalue)
        """

        T = len(y)

        Kx, Ky = self.compute_kern(x, y)
        stat = self.statistic(x, y)

        mean_appr = (np.trace(Kx) * np.trace(Ky)) / T
        var_appr = (
            2 * np.trace(Kx @ Kx) * np.trace(Ky @ Ky) / T**2
        )
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        pvalue = 1 - np.mean(gamma.cdf(stat, k_appr, theta_appr))

        self.stat = stat

        return ConditionalIndependenceTestOutput(stat, pvalue)
