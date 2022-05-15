"""
Module containing datasources for representing distributions.
"""
from __future__ import print_function, division

from builtins import range, object
from past.utils import old_div

from abc import ABC, abstractmethod
import autograd.numpy as np
import scipy.stats as stats
from numpy.random import default_rng


class DataSource(ABC):
    """
    A source of data allowing resampling. Subclasses may prefix
    class names with DS.
    """

    @abstractmethod
    def sample(self, n, seed):
        """Return a Data. Returned result should be deterministic given
        the input (n, seed)."""
        raise NotImplementedError()


class DSIsotropicNormal(DataSource):
    """
    A DataSource providing samples from a mulivariate isotropic normal
    distribution.
    """

    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean
        variance: a positive floating-point number for the variance.
        """
        assert len(mean.shape) == 1
        self.mean = mean
        self.variance = variance

    def sample(self, n, seed=2):
        rng = default_rng(seed)
        d = len(self.mean)
        mean = self.mean
        variance = self.variance
        X = rng.standard_normal(size=(n, d)) * np.sqrt(variance) + mean
        return X


class DSNormal(DataSource):
    """
    A DataSource implementing a multivariate Gaussian.
    """

    def __init__(self, mean, cov):
        """
        mean: a numpy array of length d.
        cov: d x d numpy array for the covariance.
        """
        self.mean = mean
        self.cov = cov
        assert mean.shape[0] == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]

    def sample(self, n, seed=3):
        rng = default_rng(seed)
        mvn = stats.multivariate_normal(self.mean, self.cov)
        X = mvn.rvs(size=n)
        if len(X.shape) == 1:
            # This can happen if d=1
            X = X[:, np.newaxis]
        return X
