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
