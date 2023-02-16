"""
Module containing kernel related classes

TODO: Replace kernel evaluation with compute_kern found in hyppo.common.tools by reducing
dependencies on autograd.numpy
"""
from __future__ import division

from builtins import str
from past.utils import old_div

from abc import ABC, abstractmethod
import autograd
import autograd.numpy as np

from ..tools import compute_kern


class Kernel(ABC):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        pass


class DifferentiableKernel(ABC):
    def gradX_y(self, X, y):
        """
        Compute the gradient with respect to X (the first argument of the
        kernel). Base class provides a default autograd implementation for convenience.
        Subclasses should override if this does not work.
        X: nx x d numpy array.
        y: numpy array of length d.
        Return a numpy array G of size nx x d, the derivative of k(X, y) with
        respect to X.
        """
        yrow = np.reshape(y, (1, -1))
        f = lambda X: self.eval(X, yrow)
        g = autograd.elementwise_grad(f)
        G = g(X)
        assert G.shape[0] == X.shape[0]
        assert G.shape[1] == X.shape[1]
        return G


class LinearKSTKernel(ABC):
    """
    Interface specifiying methods a kernel has to implement to be used with
    the linear-time version of Kernelized Stein discrepancy test of
    Liu et al., 2016 (ICML 2016).
    """

    @abstractmethod
    def pair_gradX_Y(self, X, Y):
        """
        Compute the gradient with respect to X in k(X, Y), evaluated at the
        specified X and Y.
        X: n x d
        Y: n x d
        Return a numpy array of size n x d
        """
        raise NotImplementedError()

    @abstractmethod
    def pair_gradY_X(self, X, Y):
        """
        Compute the gradient with respect to Y in k(X, Y), evaluated at the
        specified X and Y.
        X: n x d
        Y: n x d
        Return a numpy array of size n x d
        """
        raise NotImplementedError()

    @abstractmethod
    def pair_gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.
        X: n x d numpy array.
        Y: n x d numpy array.
        Return a one-dimensional length-n numpy array of the derivatives.
        """
        raise NotImplementedError()


class KSTKernel(ABC):
    """
    Interface specifiying methods a kernel has to implement to be used with
    the Kernelized Stein discrepancy test of Chwialkowski et al., 2016 and
    Liu et al., 2016 (ICML 2016 papers) See goftest.KernelSteinTest.
    """

    @abstractmethod
    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).
        X: nx x d
        Y: ny x d
        Return a numpy array of size nx x ny.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).
        X: nx x d
        Y: ny x d
        Return a numpy array of size nx x ny.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.
        X: nx x d numpy array.
        Y: ny x d numpy array.
        Return a nx x ny numpy array of the derivatives.
        """
        raise NotImplementedError()


class KGauss(DifferentiableKernel, KSTKernel, LinearKSTKernel):
    """
    The standard isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.
        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array
        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y**2, 1), (1, -1))
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2
        K = np.exp(old_div(-D2, (2.0 * self.sigma2)))
        return K

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).
        X: nx x d
        Y: ny x d
        Return a numpy array of size nx x ny.
        """
        sigma2 = self.sigma2
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        G = -K * Diff / sigma2
        return G

    def pair_gradX_Y(self, X, Y):
        """
        Compute the gradient with respect to X in k(X, Y), evaluated at the
        specified X and Y.
        X: n x d
        Y: n x d
        Return a numpy array of size n x d
        """
        sigma2 = self.sigma2
        Kvec = self.pair_eval(X, Y)
        # n x d
        Diff = X - Y
        G = -Kvec[:, np.newaxis] * Diff / sigma2
        return G

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).
        X: nx x d
        Y: ny x d
        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def pair_gradY_X(self, X, Y):
        """
        Compute the gradient with respect to Y in k(X, Y), evaluated at the
        specified X and Y.
        X: n x d
        Y: n x d
        Return a numpy array of size n x d
        """
        return -self.pair_gradX_Y(X, Y)

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.
        X: nx x d numpy array.
        Y: ny x d numpy array.
        Return a nx x ny numpy array of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        d = d1
        sigma2 = self.sigma2
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2 * np.dot(X, Y.T) + np.sum(Y**2, 1)
        K = np.exp(old_div(-D2, (2.0 * sigma2)))
        G = K / sigma2 * (d - old_div(D2, sigma2))
        return G

    def pair_gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.
        X: n x d numpy array.
        Y: n x d numpy array.
        Return a one-dimensional length-n numpy array of the derivatives.
        """
        d = X.shape[1]
        sigma2 = self.sigma2
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.exp(old_div(-D2, (2.0 * self.sigma2)))
        G = Kvec / sigma2 * (d - old_div(D2, sigma2))
        return G

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...
        Parameters
        ----------
        X, Y : n x d numpy array
        Return
        -------
        a numpy array with length n
        """
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.exp(old_div(-D2, (2.0 * self.sigma2)))
        return Kvec
