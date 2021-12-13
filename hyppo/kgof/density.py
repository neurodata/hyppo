"""
Module containing implementations of some unnormalized probability density 
functions.
Contains overlapping functionality with sims that exist in numpy.
Module will be refactored to remove dependencies on this object.
"""
from __future__ import division

from builtins import range
from past.utils import old_div

from abc import ABC, abstractmethod
import autograd
import autograd.numpy as np
from .data import DSNormal, DSIsotropicNormal, DSGaussianMixture
import scipy.stats as stats
import logging


def warn_bounded_domain(self):
    logging.warning(
        "{} has a bounded domain. This may have an unintended effect to the test result of FSSD.".format(
            self.__class__
        )
    )


def from_log_den(d, f):
    """
    Construct an UnnormalizedDensity from the function f, implementing the log
    of an unnormalized density.
    f: X -> den where X: n x d and den is a numpy array of length n.
    """
    return UDFromCallable(d, flog_den=f)


def from_grad_log(d, g):
    """
    Construct an UnnormalizedDensity from the function g, implementing the
    gradient of the log of an unnormalized density.
    g: X -> grad where X: n x d and grad is n x d (2D numpy array)
    """
    return UDFromCallable(d, fgrad_log=g)


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


class UDFromCallable(UnnormalizedDensity):
    """
    UnnormalizedDensity constructed from the specified implementations of
    log_den() and grad_log() as callable objects.
    """

    def __init__(self, d, flog_den=None, fgrad_log=None):
        """
        Only one of log_den and grad_log are required.
        If log_den is specified, the gradient is automatically computed with
        autograd.
        d: the dimension of the domain of the density
        log_den: a callable object (function) implementing the log of an unnormalized density. See UnnormalizedDensity.log_den.
        grad_log: a callable object (function) implementing the gradient of the log of an unnormalized density.
        """
        if flog_den is None and fgrad_log is None:
            raise ValueError("At least one of {log_den, grad_log} must be specified.")
        self.d = d
        self.flog_den = flog_den
        self.fgrad_log = fgrad_log

    def log_den(self, X):
        flog_den = self.flog_den
        if flog_den is None:
            raise ValueError("log_den callable object is None.")
        return flog_den(X)

    def grad_log(self, X):
        fgrad_log = self.fgrad_log
        if fgrad_log is None:
            # autograd
            g = autograd.elementwise_grad(self.flog_den)
            G = g(X)
        else:
            G = fgrad_log(X)
        return G

    def dim(self):
        return self.d


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
        # print self.prec

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


class GaussianMixture(UnnormalizedDensity):
    """
    UnnormalizedDensity of a Gaussian mixture in R^d where each component
    can be arbitrary. This is the most general form of a Gaussian mixture.
    Let k be the number of mixture components.
    """

    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a k x d x d numpy array containing a stack of k covariance
            matrices, one for each mixture component.
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k, d = means.shape
        if k != variances.shape[0]:
            raise ValueError(
                "Number of components in means and variances do not match."
            )

        if pmix is None:
            pmix = old_div(np.ones(k), float(k))

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError("Mixture weights do not sum to 1.")

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def log_den(self, X):
        return self.log_normalized_den(X)

    def log_normalized_den(self, X):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        n = X.shape[0]

        den = np.zeros(n, dtype=float)
        for i in range(k):
            norm_den_i = GaussianMixture.multivariate_normal_density(
                means[i], variances[i], X
            )
            den = den + norm_den_i * pmix[i]
        return np.log(den)

    @staticmethod
    def multivariate_normal_density(mean, cov, X):
        """
        Exact density (not log density) of a multivariate Gaussian.
        mean: length-d array
        cov: a dxd covariance matrix
        X: n x d 2d-array
        """

        evals, evecs = np.linalg.eigh(cov)
        cov_half_inv = evecs.dot(np.diag(evals ** (-0.5))).dot(evecs.T)
        # print(evals)
        half_evals = np.dot(X - mean, cov_half_inv)
        full_evals = np.sum(half_evals ** 2, 1)
        unden = np.exp(-0.5 * full_evals)

        Z = np.sqrt(np.linalg.det(2.0 * np.pi * cov))
        den = unden / Z
        assert len(den) == X.shape[0]
        return den

    def get_datasource(self):
        return DSGaussianMixture(self.means, self.variances, self.pmix)

    def dim(self):
        k, d = self.means.shape
        return d
