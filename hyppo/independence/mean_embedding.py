import numpy
from warnings import warn
from numpy import mean, transpose, cov, shape
from numpy.linalg import linalg, LinAlgError, solve
from scipy.stats import chi2


class MeanEmbeddingTest:
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

    def __init__(self, data_x, data_y, scale=1, number_of_random_frequencies=5):
        self.data_x = scale*data_x
        self.data_y = scale*data_y
        self.number_of_frequencies = number_of_random_frequencies
        self.scale = scale

    def get_estimate(self, data, point):
        '''

        :param data:
        :param point:
        :return: mean embeddings of data
        '''
        z = data - self.scale * point
        z2 = numpy.linalg.norm(z, axis=1)**2
        return numpy.exp(-z2/2.0)

    def get_difference(self, point):
        '''

        :param point:
        :return: differences in ME
        '''
        return self.get_estimate(self.data_x, point) - self.get_estimate(self.data_y, point)

    def vector_of_differences(self, dim):
        '''

        :param dim:
        :return: vector of difference b/w mean embeddings
        '''
        points = numpy.random.randn(self.number_of_frequencies, dim)
        a = [self.get_difference(point) for point in points]
        return numpy.array(a).T

    def compute_pvalue(self):
        '''
        :return: W * Sigma * W statistic and p value
        '''
        _, dimension = numpy.shape(self.data_x)
        obs = self.vector_of_differences(dimension)

        return self.mahalanobis_distance(obs, self.number_of_frequencies)
