import numpy
from warnings import warn
from numpy import mean, transpose, cov, shape
from numpy.linalg import linalg, LinAlgError, solve
from scipy.stats import chi2

class SmoothCFTest:

    def _gen_random(self, dimension):
        return numpy.random.randn(dimension, self.num_random_features)

    def __init__(self, data_x, data_y, scale=2.0, num_random_features=5, frequency_generator=None):
        self.data_x = scale*data_x
        self.data_y = scale*data_y
        self.num_random_features = num_random_features
        _, dimension_x = numpy.shape(self.data_x)
        _, dimension_y = numpy.shape(self.data_y)
        assert dimension_x == dimension_y
        self.random_frequencies = self._gen_random(dimension_x)

    def mahalanobis_distance(self, difference, num_random_features):
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
        difference = smooth_difference(self.random_frequencies, self.data_x, self.data_y)
        return self.mahalanobis_distance(difference, 2 * self.num_random_features)
