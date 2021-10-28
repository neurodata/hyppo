import numpy as np
from numba import jit

from warnings import warn
from numpy import mean, transpose, cov, shape
from numpy.linalg import linalg, LinAlgError, solve
from ..tools import check_perm_blocks_dim, chi2_approx, compute_dist
from ._utils import _CheckInputs
from .base import IndependenceTest, IndependenceTestOutput
from .base import IndependenceTest


class MeanEmbeddingTest(IndependenceTest):
    def __init__(self, scale=1, number_of_random_frequencies=5, **kwargs):
        self.number_of_frequencies = number_of_random_frequencies
        self.scale = scale
        IndependenceTest.__init__(self, **kwargs)

    def statistic(self, x, y):
        data_x = x
        data_y = y
        _, dimension = np.shape(data_x)
        obs = vector_of_differences(dimension, data_x, data_y, self.number_of_frequencies, self.scale)
        return _mahalanobis_distance(obs, self.number_of_frequencies)

    def test(
            self, x, y, reps=1000, workers=1, auto=True, perm_blocks=None, random_state=None
    ):

        check_input = _CheckInputs(
            x,
            y,
            reps=reps,
        )
        x, y = check_input()

        stat, pvalue = chi2_approx(self.statistic, x, y)
        self.stat = stat
        self.pvalue = pvalue

        return IndependenceTestOutput(stat, pvalue)


@jit(nopython=True, cache=True)
def _mahalanobis_distance(difference, num_random_features):
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


@jit(nopython=True, cache=True)
def get_estimate(data, point, scale):
    '''

    :param data:
    :param point:
    :return: mean embeddings of data
    '''
    z = data - scale * point
    z2 = np.linalg.norm(z, axis=1)**2
    return np.exp(-z2/2.0)


@jit(nopython=True, cache=True)
def get_difference(point, x, y, scale):
    '''

    :param point:
    :return: differences in ME
    '''
    return get_estimate(x, point, scale) - get_estimate(y, point, scale)


@jit(nopython=True, cache=True)
def vector_of_differences(dim, x, y, number_of_frequencies, scale):
    '''

    :param dim:
    :return: vector of difference b/w mean embeddings
    '''
    points = np.random.randn(number_of_frequencies, dim)
    a = [get_difference(point, x, y, scale) for point in points]
    return np.array(a).T

