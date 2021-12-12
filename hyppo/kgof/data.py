"""
Module containing data structures for representing datasets.
Contains overlapping functionality with sims that exist in hyppo.tools.
Module will be refactored to remove dependencies on this object.
"""
from __future__ import print_function, division

from builtins import range, object
from past.utils import old_div

from abc import ABC, abstractmethod
import autograd.numpy as np
from ._utils import tr_te_indices
import scipy.stats as stats
from numpy.random import default_rng


class Data(object):
    """
    Class representing a dataset i.e., en encapsulation of a data matrix
    whose rows are vectors drawn from a distribution.
    """

    def __init__(self, X):
        """
        :param X: n x d numpy array for dataset X
        """
        self.X = X

        if not np.all(np.isfinite(X)):
            raise ValueError("Not all elements in X are finite.")

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0)
        prec = 4
        desc = ""
        desc += "E[x] = %s \n" % (np.array_str(mean_x, precision=prec))
        desc += "Std[x] = %s \n" % (np.array_str(std_x, precision=prec))
        return desc

    def dim(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def sample_size(self):
        return self.X.shape[0]

    def n(self):
        return self.sample_size()

    def data(self):
        """Return the data matrix."""
        return self.X

    def split_tr_te(self, tr_proportion=0.5, seed=820, return_tr_ind=False):
        """Split the dataset into training and test sets.
        Return (Data for tr, Data for te)"""
        X = self.X
        nx, dx = X.shape
        Itr, Ite = tr_te_indices(nx, tr_proportion, seed)
        tr_data = Data(X[Itr, :])
        te_data = Data(X[Ite, :])
        if return_tr_ind:
            return (tr_data, te_data, Itr)
        else:
            return (tr_data, te_data)

    def subsample(self, n, seed=87, return_ind=False):
        """Subsample without replacement. Return a new Data."""
        if n > self.X.shape[0]:
            raise ValueError("n should not be larger than sizes of X")
        rng = default_rng(seed)
        ind_x = rng.choice(self.X.shape[0], n, replace=False)
        if return_ind:
            return Data(self.X[ind_x, :]), ind_x
        else:
            return Data(self.X[ind_x, :])

    def clone(self):
        """
        Return a new Data object with a separate copy of each internal
        variable, and with the same content.
        """
        nX = np.copy(self.X)
        return Data(nX)

    def __add__(self, data2):
        """
        Merge the current Data with another one.
        Create a new Data and create a new copy for all internal variables.
        """
        copy = self.clone()
        copy2 = data2.clone()
        nX = np.vstack((copy.X, copy2.X))
        return Data(nX)


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

    def dim(self):
        """
        Return the dimension of the data.  If possible, subclasses should
        override this. Determining the dimension by sampling may not be
        efficient, especially if the sampling relies on MCMC.
        """
        dat = self.sample(n=1, seed=3)
        return dat.dim()


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
        return Data(X)


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
        return Data(X)


class DSGaussianMixture(DataSource):
    """
    A DataSource implementing a Gaussian mixture in R^d where each component
    is an arbitrary Gaussian distribution.
    Let k be the number of mixture components.
    """

    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a k x d x d numpy array containing k covariance matrices,
            one for each component.
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

    def sample(self, n, seed=29):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        sam_list = []
        rng = default_rng(seed)
        # counts for each mixture component
        counts = rng.multinomial(n, pmix, size=1)

        # counts is a 2d array
        counts = counts[0]

        # For each component, draw from its corresponding mixture component.
        for i, nc in enumerate(counts):
            cov = variances[i]
            mnorm = stats.multivariate_normal(means[i], cov)
            # Sample from ith component
            sam_i = mnorm.rvs(size=nc)
            sam_list.append(sam_i)
        sample = np.vstack(sam_list)
        assert sample.shape[0] == n
        rng.shuffle(sample)
        return Data(sample)


class DSGaussBernRBM(DataSource):
    """
    A DataSource implementing a Gaussian-Bernoulli Restricted Boltzmann Machine.
    The probability of the latent vector h is controlled by the vector c.
    The parameterization of the Gaussian-Bernoulli RBM is given in
    density.GaussBernRBM.
    - It turns out that this is equivalent to drawing a vector of {-1, 1} for h
        according to h ~ Discrete(sigmoid(2c)).
    - Draw x | h ~ N(B*h+b, I)
    """

    def __init__(self, B, b, c, burnin=2000):
        """
        B: a dx x dh matrix
        b: a numpy array of length dx
        c: a numpy array of length dh
        burnin: burn-in iterations when doing Gibbs sampling
        """
        assert burnin >= 0
        dh = len(c)
        dx = len(b)
        assert B.shape[0] == dx
        assert B.shape[1] == dh
        assert dx > 0
        assert dh > 0
        self.B = B
        self.b = b
        self.c = c
        self.burnin = burnin

    @staticmethod
    def sigmoid(x):
        """
        x: a numpy array.
        """
        return old_div(1.0, (1 + np.exp(-x)))

    def _blocked_gibbs_next(self, X, H):
        """
        Sample from the mutual conditional distributions.
        """
        dh = H.shape[1]
        n, dx = X.shape
        B = self.B
        b = self.b

        # Draw H.
        XB2C = np.dot(X, self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = DSGaussBernRBM.sigmoid(XB2C)
        # H: n x dh
        rng = default_rng()
        H = (rng.random(size=(n, dh)) <= Ph) * 2 - 1.0
        assert np.all(np.abs(H) - 1 <= 1e-6)
        # Draw X.
        # mean: n x dx
        mean = old_div(np.dot(H, B.T), 2.0) + b
        X = rng.standard_normal(size=(n, dx)) + mean
        return X, H

    def sample(self, n, seed=3, return_latent=False):
        """
        Sample by blocked Gibbs sampling
        """
        B = self.B
        b = self.b
        c = self.c
        dh = len(c)
        dx = len(b)

        # Initialize the state of the Markov chain
        rng = default_rng(seed)
        X = rng.standard_normal(n, dx)
        H = rng.integers(1, 2, (n, dh)) * 2 - 1.0

        # burn-in
        for t in range(self.burnin):
            X, H = self._blocked_gibbs_next(X, H)
        # sampling
        X, H = self._blocked_gibbs_next(X, H)
        if return_latent:
            return Data(X), H
        else:
            return Data(X)

    def dim(self):
        return self.B.shape[0]


class DSGaussCosFreqs(DataSource):
    """
    A DataSource to sample from the density
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

    def sample(self, n, seed=872):
        """
        Rejection sampling.
        """
        d = len(self.freqs)
        sigma2 = self.sigma2
        freqs = self.freqs
        rng = default_rng(seed)
        # rejection sampling
        sam = np.zeros((n, d))
        # sample block_size*d at a time.
        block_size = 500
        from_ind = 0
        while from_ind < n:
            # The proposal q is N(0, sigma2*I)
            X = rng.standard_normal(size=(block_size, d)) * np.sqrt(sigma2)
            q_un = np.exp(old_div(-np.sum(X ** 2, 1), (2.0 * sigma2)))
            # unnormalized density p
            p_un = q_un * (1 + np.prod(np.cos(X * freqs), 1))
            c = 2.0
            I = stats.uniform.rvs(size=block_size) < old_div(p_un, (c * q_un))

            # accept
            accepted_count = np.sum(I)
            to_take = min(n - from_ind, accepted_count)
            end_ind = from_ind + to_take

            AX = X[I, :]
            X_take = AX[:to_take, :]
            sam[from_ind:end_ind, :] = X_take
            from_ind = end_ind
        return Data(sam)

    def dim(self):
        return len(self.freqs)
