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
from .data import (
    DSNormal,
    DSIsotropicNormal,
    DSIsoGaussianMixture,
    DSGaussianMixture,
    DSGaussBernRBM,
    DSGaussCosFreqs,
)
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


class IsoGaussianMixture(UnnormalizedDensity):
    """
    UnnormalizedDensity of a Gaussian mixture in R^d where each component
    is an isotropic multivariate normal distribution.
    Let k be the number of mixture components.
    """

    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a one-dimensional length-k array of variances
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k, d = means.shape
        if k != len(variances):
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
            norm_den_i = IsoGaussianMixture.normal_density(means[i], variances[i], X)
            den = den + norm_den_i * pmix[i]
        return np.log(den)

    @staticmethod
    def normal_density(mean, variance, X):
        """
        Exact density (not log density) of an isotropic Gaussian.
        mean: length-d array
        variance: scalar variances
        X: n x d 2d-array
        """
        Z = np.sqrt(2.0 * np.pi * variance)
        unden = np.exp(old_div(-np.sum((X - mean) ** 2.0, 1), (2.0 * variance)))
        den = old_div(unden, Z)
        assert len(den) == X.shape[0]
        return den

    def get_datasource(self):
        return DSIsoGaussianMixture(self.means, self.variances, self.pmix)

    def dim(self):
        k, d = self.means.shape
        return d


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


class GaussBernRBM(UnnormalizedDensity):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine.
    The joint density takes the form
        p(x, h) = Z^{-1} exp(0.5*x^T B h + b^T x + c^T h - 0.5||x||^2)
    where h is a vector of {-1, 1}.
    """

    def __init__(self, B, b, c):
        """
        B: a dx x dh matrix
        b: a numpy array of length dx
        c: a numpy array of length dh
        """
        dh = len(c)
        dx = len(b)
        assert B.shape[0] == dx
        assert B.shape[1] == dh
        assert dx > 0
        assert dh > 0
        self.B = B
        self.b = b
        self.c = c

    def log_den(self, X):
        B = self.B
        b = self.b
        c = self.c

        XBC = 0.5 * np.dot(X, B) + c
        unden = (
            np.dot(X, b)
            - 0.5 * np.sum(X ** 2, 1)
            + np.sum(np.log(np.exp(XBC) + np.exp(-XBC)), 1)
        )
        assert len(unden) == X.shape[0]
        return unden

    def grad_log(self, X):
        #    """
        #    Evaluate the gradients (with respect to the input) of the log density at
        #    each of the n points in X. This is the score function.

        #    X: n x d numpy array.
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function.
        X: n x d numpy array.
        Return an n x d numpy array of gradients.
        """
        XB = np.dot(X, self.B)
        Y = 0.5 * XB + self.c
        E2y = np.exp(2 * Y)
        # n x dh
        Phi = old_div((E2y - 1.0), (E2y + 1))
        # n x dx
        T = np.dot(Phi, 0.5 * self.B.T)
        S = self.b - X + T
        return S

    def get_datasource(self, burnin=2000):
        return DSGaussBernRBM(self.B, self.b, self.c, burnin=burnin)

    def dim(self):
        return len(self.b)


class ISIPoissonLinear(UnnormalizedDensity):
    """
    Unnormalized density of inter-arrival times from nonhomogeneous poisson process with linear intensity function.
    lambda = 1 + bt
    """

    def __init__(self, b):
        """
        b: slope of the linear function
        """
        warn_bounded_domain(self)
        self.b = b

    def log_den(self, X):
        b = self.b
        unden = -np.sum(0.5 * b * X ** 2 + X - np.log(1.0 + b * X), 1)
        return unden

    def dim(self):
        return 1


class ISIPoissonSine(UnnormalizedDensity):
    """
    Unnormalized density of inter-arrival times from nonhomogeneous poisson process with sine intensity function.
    lambda = b*(1+sin(w*X))
    """

    def __init__(self, w=10.0, b=1.0):
        """
        w: the frequency of sine function
        b: amplitude of intensity function
        """
        warn_bounded_domain(self)
        self.b = b
        self.w = w

    def log_den(self, X):
        b = self.b
        w = self.w
        unden = np.sum(
            b * (-X + old_div((np.cos(w * X) - 1), w))
            + np.log(b * (1 + np.sin(w * X))),
            1,
        )
        return unden

    def dim(self):
        return 1


class Gamma(UnnormalizedDensity):
    """
    A gamma distribution.
    """

    def __init__(self, alpha, beta=1.0):
        """
        alpha: shape of parameter
        beta: scale
        """
        warn_bounded_domain(self)
        self.alpha = alpha
        self.beta = beta

    def log_den(self, X):
        alpha = self.alpha
        beta = self.beta
        # unden = np.sum(stats.gamma.logpdf(X, alpha, scale = beta), 1)
        unden = np.sum(-beta * X + (alpha - 1) * np.log(X), 1)
        return unden

    def get_datasource(self):
        return DSNormal(self.mean, self.cov)

    def dim(self):
        return 1


class LogGamma(UnnormalizedDensity):
    """
    A gamma distribution with transformed domain.
    t = exp(x),  t \in R+  x \in R
    """

    def __init__(self, alpha, beta=1.0):
        """
        alpha: shape of parameter
        beta: scale
        """
        self.alpha = alpha
        self.beta = beta

    def log_den(self, X):
        alpha = self.alpha
        beta = self.beta
        # unden = np.sum(stats.gamma.logpdf(X, alpha, scale = beta), 1)
        unden = np.sum(-beta * np.exp(X) + (alpha - 1) * X + X, 1)
        return unden

    def get_datasource(self):
        return DSNormal(self.mean, self.cov)

    def dim(self):
        return 1


class ISILogPoissonLinear(UnnormalizedDensity):
    """
    Unnormalized density of inter-arrival times from nonhomogeneous poisson process with linear intensity function.
    lambda = 1 + bt
    """

    def __init__(self, b):
        """
        b: slope of the linear function
        """
        warn_bounded_domain(self)
        self.b = b

    def log_den(self, X):
        b = self.b
        unden = -np.sum(
            0.5 * b * np.exp(X) ** 2 + np.exp(X) - np.log(1.0 + b * np.exp(X)) - X, 1
        )
        return unden

    def dim(self):
        return 1


class ISIPoisson2D(UnnormalizedDensity):
    """
    Unnormalized density of nonhomogeneous spatial poisson process
    """

    def __init__(self):
        """
        lambda_(X,Y) = X^2 + Y^2
        """
        warn_bounded_domain(self)

    def quadratic_intensity(self, X, Y):
        int_intensity = -(X ** 2 + Y ** 2) * X * Y + 3 * np.log(X ** 2 + Y ** 2)
        return int_intensity

    def log_den(self, X):
        unden = self.quadratic_intensity(X[:, 0], X[:, 1])
        return unden

    def dim(self):
        return 1


class ISISigmoidPoisson2D(UnnormalizedDensity):
    """
    Unnormalized density of nonhomogeneous spatial poisson process with sigmoid transformation
    """

    def __init__(self, intensity="quadratic", w=1.0, a=1.0):
        """
        lambda_(X,Y) = a* X^2 + Y^2
        X = 1/(1+exp(s))
        Y = 1/(1+exp(t))
        X, Y \in [0,1], s,t \in R
        """
        warn_bounded_domain(self)
        self.a = a
        self.w = w
        if intensity == "quadratic":
            self.intensity = self.quadratic_intensity
        elif intensity == "sine":
            self.intensity = self.sine_intensity
        else:
            raise ValueError("Not intensity function found")

    def sigmoid(self, x):
        sig = old_div(1, (1 + np.exp(x)))
        return sig

    def quadratic_intensity(self, s, t):
        X = self.sigmoid(s)
        Y = self.sigmoid(t)
        int_intensity = -(self.a * X ** 2 + Y ** 2) * X * Y + 3 * (
            np.log(self.a * X ** 2 + Y ** 2) + np.log((X * (X - 1) * Y * (Y - 1)))
        )
        return int_intensity

    def log_den(self, S):
        unden = self.quadratic_intensity(S[:, 0], S[:, 1])
        return unden

    def dim(self):
        return 1


class Poisson2D(UnnormalizedDensity):
    """
    Unnormalized density of nonhomogeneous spatial poisson process
    """

    def __init__(self, w=1.0):
        """
        lambda_(X,Y) = sin(w*pi*X)+sin(w*pi*Y)
        """
        self.w = w

    def lamb_sin(self, X):
        return np.prod(np.sin(self.w * np.pi * X), 1)

    def log_den(self, X):
        unden = np.log(self.gmm_den(X))
        return unden

    def dim(self):
        return 1


class Resample(UnnormalizedDensity):
    """
    Unnormalized Density of real dataset with estimated intensity function
    fit takes the function to evaluate the density of resampled data
    """

    def __init__(self, fit):
        self.fit = fit

    def log_den(self, X):
        unden = np.log(self.fit(X))
        return unden

    def dim(self):
        return 1


class GaussCosFreqs(UnnormalizedDensity):
    """
    p(x) \propto exp(-||x||^2/2sigma^2)*(1+ prod_{i=1}^d cos(w_i*x_i))
    where w1,..wd are frequencies of each dimension.
    sigma^2 is the overall variance.
    """

    def __init__(self, sigma2, freqs):
        """
        sigma2: overall scale of the distribution. A positive scalar.
        freqs: a 1-d array of length d for the frequencies.
        """
        self.sigma2 = sigma2
        if sigma2 <= 0:
            raise ValueError("sigma2 must be > 0")
        self.freqs = freqs

    def log_den(self, X):
        sigma2 = self.sigma2
        freqs = self.freqs
        log_unden = (
            old_div(-np.sum(X ** 2, 1), (2.0 * sigma2))
            + 1
            + np.prod(np.cos(X * freqs), 1)
        )
        return log_unden

    def dim(self):
        return len(self.freqs)

    def get_datasource(self):
        return DSGaussCosFreqs(self.sigma2, self.freqs)
