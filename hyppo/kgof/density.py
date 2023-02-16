"""
Module containing implementations of some unnormalized probability density 
functions.
"""
from __future__ import division

from builtins import range
from past.utils import old_div

from abc import ABC, abstractmethod
import autograd
import autograd.numpy as np
from .datasource import DSNormal, DSIsotropicNormal
import scipy.stats as stats


class UnnormalizedDensity(ABC):
    """
    An abstract class of an unnormalized probability density function.  This is
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """
    
    @abstractmethod
    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.
        X: n x d numpy array
        Return a one-dimensional numpy array of length n.
        """
        raise NotImplementedError()

    def log_normalized_den(self, X):
        """
        Evaluate the exact normalized log density. The difference to log_den()
        is that this method adds the normalizer. This method is not
        compulsory. Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this density.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function. Given an
        implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.
        X: n x d numpy array.
        Return an n x d numpy array of gradients.
        """
        g = autograd.elementwise_grad(self.log_den)
        G = g(X)
        return G

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()


class IsotropicNormal(UnnormalizedDensity):
    """
    Unnormalized density of an isotropic multivariate normal distribution.
    """

    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean
        variance: a positive floating-point number for the variance.
        """
        self.mean = mean
        self.variance = variance

    def log_den(self, X):
        mean = self.mean
        variance = self.variance
        unden = old_div(-np.sum((X - mean) ** 2, 1), (2.0 * variance))
        return unden

    def log_normalized_den(self, X):
        d = self.dim()
        return stats.multivariate_normal.logpdf(
            X, mean=self.mean, cov=self.variance * np.eye(d)
        )

    def get_datasource(self):
        return DSIsotropicNormal(self.mean, self.variance)

    def dim(self):
        return len(self.mean)


class Normal(UnnormalizedDensity):
    """
    A multivariate normal distribution.
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
        E, V = np.linalg.eigh(cov)
        if np.any(np.abs(E) <= 1e-7):
            raise ValueError("covariance matrix is not full rank.")
        # The precision matrix
        self.prec = np.dot(np.dot(V, np.diag(old_div(1.0, E))), V.T)

    def log_den(self, X):
        mean = self.mean
        X0 = X - mean
        X0prec = np.dot(X0, self.prec)
        unden = old_div(-np.sum(X0prec * X0, 1), 2.0)
        return unden

    def get_datasource(self):
        return DSNormal(self.mean, self.cov)

    def dim(self):
        return len(self.mean)
