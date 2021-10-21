import numpy
from warnings import warn
from numpy import mean, transpose, cov, shape, concatenate, newaxis, exp, sin, cos
from numpy.linalg import linalg, LinAlgError, solve
from scipy.stats import chi2


class SmoothCFTest:
    def _gen_random(self, dimension):
        '''
        :param dimension: number of
        :return: normally distributed array
        '''
        return numpy.random.randn(dimension, self.num_random_features)

    def smooth(self, data):
        '''
        :param data: X or Y
        :return:  normalized
        '''
        w = linalg.norm(data, axis=1)
        w = exp(-w ** 2 / 2)
        return w[:, newaxis]

    def smooth_cf(self, data, w, random_frequencies):
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

    def smooth_difference(self, random_frequencies, X, Y):
        """
        :param random_frequencies: distributed normally
        :param X: X data
        :param Y: Y data
        :return: Distance between smooth characteristic functions
        """
        x_smooth = self.smooth(X)
        y_smooth = self.smooth(Y)
        return self.smooth_cf(X, x_smooth, random_frequencies) - self.smooth_cf(Y, y_smooth, random_frequencies)

    def __init__(self, data_x, data_y, scale=2.0, num_random_features=5, frequency_generator=None):
        self.data_x = scale*data_x
        self.data_y = scale*data_y
        self.num_random_features = num_random_features
        _, dimension_x = numpy.shape(self.data_x)
        _, dimension_y = numpy.shape(self.data_y)
        assert dimension_x == dimension_y
        self.random_frequencies = self._gen_random(dimension_x)

    def mahalanobis_distance(self, difference, num_random_features):
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

        return chi2.sf(stat, num_random_features)

    def compute_pvalue(self):
        """
        :return: test statistic for smoothCF
        """
        difference = self.smooth_difference(self.random_frequencies, self.data_x, self.data_y)
        return self.mahalanobis_distance(difference, 2 * self.num_random_features)
