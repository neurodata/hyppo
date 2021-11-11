import numpy as np
from numba import jit
from warnings import warn
from numpy import concatenate, newaxis, exp, sin, cos
from scipy.stats import chi2
from numpy import mean, transpose, cov, shape
from numpy.linalg import linalg, LinAlgError, solve
from ._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest, KSampleTestOutput


class SmoothCFTest(KSampleTest):
    def __init__(self, compute_distance=False, bias=False, scale=2.0, num_random_features=5, **kwargs):

        self.scale = scale
        self.num_random_features = num_random_features
        KSampleTest.__init__(
            self, compute_distance=compute_distance, bias=bias, **kwargs
        )

    def statistic(self, x, y):
        """
        :return: test statistic for smoothCF
        """
        _, dim_x = np.shape(x)
        random_frequencies = _gen_random(dim_x, self.num_random_features)
        difference = smooth_difference(random_frequencies, x, y)
        return mahalanobis_distance(difference, 2 * self.num_random_features)

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        check_input = _CheckInputs(
            inputs=[x,y],
            indep_test=None
        )
        x, y = check_input()

        stat = self.statistic(x,y)
        pvalue = chi2.sf(stat, 2*self.num_random_features)
        self.stat = stat
        self.pvalue = pvalue

        return KSampleTestOutput(stat, pvalue)


#@jit(nopython=True, cache=True)
def _gen_random(dimension, num_random_features):
    '''
    :param dimension: number of
    :return: normally distributed array
    '''
    return np.random.randn(dimension, num_random_features)


#@jit(nopython=True, cache=True)
def smooth(data):
    '''
    :param data: X or Y
    :return:  normalized
    '''
    w = linalg.norm(data, axis=1)
    w = exp(-w ** 2 / 2)
    return w[:, newaxis]


#@jit(nopython=True, cache=True)
def smooth_cf(data, w, random_frequencies):
    """
    :param data: X or Y
    :param w:
    :param random_frequencies:
    :return: The smoothed CF
    """
    n, _ = data.shape
    _, d = random_frequencies.shape
    mat = data.dot(random_frequencies)
    arr = concatenate((sin(mat) * w, cos(mat) * w), 1)
    n1, d1 = arr.shape
    assert n1 == n and d1 == 2 * d and w.shape == (n, 1)
    return arr


#@jit(nopython=True, cache=True)
def smooth_difference(random_frequencies, X, Y):
    """
    :param random_frequencies: distributed normally
    :param X: X data
    :param Y: Y data
    :return: Distance between smooth characteristic functions
    """
    x_smooth = smooth(X)
    y_smooth = smooth(Y)
    return smooth_cf(X, x_smooth, random_frequencies) - smooth_cf(Y, y_smooth, random_frequencies)


#@jit(nopython=True, cache=True)
def mahalanobis_distance(difference, num_random_features):
    """

    :param difference: distance between two smooth characteristic functions
    :param num_random_features: random frequencies to be used
    :return: the test statistic, W * Sigma * W
    """
    num_samples, _ = shape(difference)
    sigma = cov(transpose(difference))

    try:
        linalg.inv(sigma)
    except LinAlgError:
        warn('covariance matrix is singular. Pvalue returned is 1.1')
        raise

    mu = mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(solve(sigma, transpose(mu)))

    return stat
    #return chi2.sf(stat, num_random_features)


