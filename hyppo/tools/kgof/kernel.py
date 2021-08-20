
"""Module containing kernel related classes"""
from __future__ import division

from builtins import str
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
#import numpy as np
#import kgof.config as config
#import kgof.util as util

from . import config
from . import util

class Kernel(with_metaclass(ABCMeta, object)):
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


class KSTKernel(with_metaclass(ABCMeta, Kernel)):
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

# end KSTKernel

class LinearKSTKernel(with_metaclass(ABCMeta, Kernel)):
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

class DifferentiableKernel(with_metaclass(ABCMeta, Kernel)):
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

# end class KSTKernel

class KDiagGauss(Kernel):
    """
    A Gaussian kernel with diagonal covariance structure i.e., one Gaussian
    width for each dimension.
    """
    def __init__(self, sigma2s):
        """
        sigma2s: a one-dimensional array of length d containing one width
            squared for each of the d dimensions.
        """
        self.sigma2s = sigma2s

    def eval(self, X, Y):
        """
        Equivalent to dividing each dimension with the corresponding width (not
        width^2) and using the standard Gaussian kernel.
        """
        sigma2s = self.sigma2s
        Xs = old_div(X,np.sqrt(sigma2s))
        Ys = old_div(Y,np.sqrt(sigma2s))
        k = KGauss(1.0)
        return k.eval(Xs, Ys)

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        sigma2s = self.sigma2s
        Xs = old_div(X,np.sqrt(sigma2s))
        Ys = old_div(Y,np.sqrt(sigma2s))
        k = KGauss(1.0)
        return k.pair_eval(Xs, Ys)

# end class KDiagGauss

class KIMQ(DifferentiableKernel, KSTKernel):
    """
    The inverse multiquadric (IMQ) kernel studied in

    Measure Sample Quality with Kernels
    Jackson Gorham, Lester Mackey

    k(x,y) = (c^2 + ||x-y||^2)^b
    where c > 0 and b < 0. Following a theorem in the paper, this kernel is
    convergence-determining only when -1 < b < 0. In the experiments,
    the paper sets b = -1/2 and c = 1.
    """

    def __init__(self, b=-0.5, c=1.0):
        if not b < 0:
            raise ValueError('b has to be negative. Was {}'.format(b))
        if not c > 0:
            raise ValueError('c has to be positive. Was {}'.format(c))
        self.b = b
        self.c = c

    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        b = self.b
        c = self.c
        D2 = util.dist2_matrix(X, Y)
        K = (c**2 + D2)**b
        return K

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        """
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c**2 + np.sum((X-Y)**2, 1))**b

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        D2 = util.dist2_matrix(X, Y)
        # 1d array of length nx
        Xi = X[:, dim]
        # 1d array of length ny
        Yi = Y[:, dim]
        # nx x ny
        dim_diff = Xi[:, np.newaxis] - Yi[np.newaxis, :]

        b = self.b
        c = self.c
        Gdim = ( 2.0*b*(c**2 + D2)**(b-1) )*dim_diff
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        b = self.b
        c = self.c
        D2 = util.dist2_matrix(X, Y)

        # d = input dimension
        d = X.shape[1]
        c2D2 = c**2 + D2
        T1 = -4.0*b*(b-1)*D2*(c2D2**(b-2) )
        T2 = -2.0*b*d*c2D2**(b-1)
        return T1 + T2

# end class KIMQ

class KGauss(DifferentiableKernel, KSTKernel, LinearKSTKernel):
    """
    The standard isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
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
        #(n1, d1) = X.shape
        #(n2, d2) = Y.shape
        #assert d1==d2, 'Dimensions of the two inputs must be the same'
        sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y**2, 1), (1, -1))
        D2 = sumx2 - 2*np.dot(X, Y.T) + sumy2
        K = np.exp(old_div(-D2,(2.0*self.sigma2)))
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
        #Diff = np.reshape(X[:, dim], (-1, 1)) - np.reshape(Y[:, dim], (1, -1))
        G = -K*Diff/sigma2
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
        G = -Kvec[:, np.newaxis]*Diff/sigma2
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
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        sigma2 = self.sigma2
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*np.dot(X, Y.T) + np.sum(Y**2, 1)
        K = np.exp(old_div(-D2,(2.0*sigma2)))
        G = K/sigma2*(d - old_div(D2,sigma2))
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
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,(2.0*self.sigma2)))
        G = Kvec/sigma2*(d - old_div(D2,sigma2))
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
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,(2.0*self.sigma2)))
        return Kvec

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2


class KMixGauss(DifferentiableKernel, KSTKernel):
    def __init__(self, sigma2s, wts=None):
        """
        Mixture of isotropic Gaussian kernels:
          sum wts[i] * exp(- ||x - y||^2 / (2 * sigma2s[i]))

        sigma2s: a list/array of squared bandwidths
        wts: a list/array of weights. Defaults to equal weights summing to 1.
        """
        self.sigma2s = sigma2s = np.asarray(sigma2s)
        assert len(sigma2s) > 0

        if wts is None:
            self.wts = wts = np.full(len(sigma2s), 1/len(sigma2s))
        else:
            self.wts = wts = np.asarray(wts)
            assert len(wts) == len(sigma2s)
            assert all(w >= 0 for w in wts)

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        sumx2 = np.sum(X**2, axis=1)[:, np.newaxis]
        sumy2 = np.sum(Y**2, axis=1)[np.newaxis, :]
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2
        return np.tensordot(
            self.wts,
            np.exp(
                D2[np.newaxis, :, :]
                / (-2 * self.sigma2s[:, np.newaxis, np.newaxis])),
            1)

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...

        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        n1, d1 = X.shape
        n2, d2 = Y.shape
        assert n1 == n2, 'Two inputs must have the same number of instances'
        assert d1 == d2, 'Two inputs must have the same dimension'
        D2 = np.sum((X - Y)**2, axis=1)
        return np.tensordot(
            self.wts,
            np.exp(D2[np.newaxis, :] / (-2 * self.sigma2s[:, np.newaxis])),
            1)

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        diffs = -X[:, [dim]] + Y[:, [dim]].T
        exps = np.exp(diffs[np.newaxis, :, :] ** 2
                      / (-2 * self.sigma2s[:, np.newaxis, np.newaxis]))
        return np.einsum('w,wij,ij->ij', self.wts / self.sigma2s, exps, diffs)

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        d = X.shape[1]
        sumx2 = np.sum(X**2, axis=1)[:, np.newaxis]
        sumy2 = np.sum(Y**2, axis=1)[np.newaxis, :]
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2
        s = (D2[np.newaxis, :, :] / self.sigma2s[:, np.newaxis, np.newaxis])
        return np.einsum('w,wij,wij->ij',
                         self.wts / self.sigma2s, np.exp(s / -2), d - s)


class KPoly(DifferentiableKernel, KSTKernel):
    def __init__(self, degree=3, gamma=None, coef0=1):
        """
        Polynomial kernel
          (gamma X^T Y + coef0)^degree

        degree: default 3
        gamma: default 1/dim
        coef0: float, default 1
        """
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        if degree <= 0:
            raise ValueError("KPoly needs positive degree")
        if not np.allclose(degree, int(degree)):
            raise ValueError("KPoly needs integral degree")

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        dot = np.dot(X, Y.T)
        gamma = 1/X.shape[1] if self.gamma is None else self.gamma
        return (gamma * dot + self.coef0) ** self.degree

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...

        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        n1, d1 = X.shape
        n2, d2 = Y.shape
        assert n1 == n2, 'Two inputs must have the same number of instances'
        assert d1 == d2, 'Two inputs must have the same dimension'

        dot = np.einsum('id,id->i', X, Y)
        gamma = 1/d1 if self.gamma is None else self.gamma
        return (gamma * dot + self.coef0) ** self.degree

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        gamma = 1/X.shape[1] if self.gamma is None else self.gamma

        if self.degree == 1:  # optimization, other expression is valid too
            out = gamma * Y[np.newaxis, :, dim]  # 1 x ny
            return np.repeat(out, X.shape[0], axis=0)

        dot = np.dot(X, Y.T)
        return (self.degree * (gamma * dot + self.coef0) ** (self.degree - 1)
                * gamma * Y[np.newaxis, :, dim])

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        gamma = 1/X.shape[1] if self.gamma is None else self.gamma

        if self.degree == 1:  # optimization, other expression is valid too
            out = gamma * X[:, dim, np.newaxis]  # nx x 1
            return np.repeat(out, Y.shape[0], axis=1)

        dot = np.dot(X, Y.T)
        return (self.degree * (gamma * dot + self.coef0) ** (self.degree - 1)
                * gamma * X[:, dim, np.newaxis])

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        gamma = 1/X.shape[1] if self.gamma is None else self.gamma

        if self.degree == 1:  # optimization, other expression is valid too
            return np.tile(gamma, (X.shape[0], X.shape[1]))

        dot = np.dot(X, Y.T)
        inside = gamma * dot + self.coef0
        to_dminus2 = inside ** (self.degree - 2)
        to_dminus1 = to_dminus2 * inside
        return (
            (self.degree * (self.degree-1) * gamma**2) * to_dminus2 * dot
            + (X.shape[1] * gamma * self.degree) * to_dminus1
        )


class KMixture(KSTKernel, LinearKSTKernel, DifferentiableKernel):
    def __init__(self, ks, wts=None):
        self.ks = ks
        if wts is None:
            self.wts = np.full(len(ks), 1/len(ks))
        else:
            self.wts = np.asarray(wts)

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        return sum(w * k.eval(X, Y) for w, k in zip(self.wts, self.ks))

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...

        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        return sum(w * k.pair_eval(X, Y) for w, k in zip(self.wts, self.ks))

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return sum(w * k.gradX_Y(X, Y, dim) for w, k in zip(self.wts, self.ks))

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return sum(w * k.gradY_X(X, Y, dim) for w, k in zip(self.wts, self.ks))

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        return sum(w * k.gradXY_sum(X, Y) for w, k in zip(self.wts, self.ks))

    def pair_gradX_Y(self, X, Y):
        """
        Compute the gradient with respect to X in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        return sum(w * k.pair_gradX_Y(X, Y) for w, k in zip(self.wts, self.ks))

    def pair_gradY_X(self, X, Y):
        """
        Compute the gradient with respect to Y in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        return sum(w * k.pair_gradY_X(X, Y) for w, k in zip(self.wts, self.ks))

    def pair_gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: n x d numpy array.
        Y: n x d numpy array.

        Return a one-dimensional length-n numpy array of the derivatives.
        """
        return sum(w * k.pair_gradXY_sum(X, Y) for w, k in zip(self.wts, self.ks))

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
        return sum(w * k.gradX_y(X, y) for w, k in zip(self.wts, self.ks))
