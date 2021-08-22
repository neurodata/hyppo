"""
Module containing data structures for representing datasets.
"""
from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import numpy as np
#import autograd.numpy as np
#import kgof.util as util
from . import util
import scipy.stats as stats

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
            print('X:')
            print(util.fullprint(X))
            raise ValueError('Not all elements in X are finite.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
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

    def split_tr_te(self, tr_proportion=0.5, seed=820, return_tr_ind = False):
        """Split the dataset into training and test sets.         

        Return (Data for tr, Data for te)"""
        X = self.X
        nx, dx = X.shape
        Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
        tr_data = Data(X[Itr, :])
        te_data = Data(X[Ite, :])
        if return_tr_ind:
            return (tr_data, te_data, Itr)
        else:
            return (tr_data, te_data)

    def subsample(self, n, seed=87, return_ind = False):
        """Subsample without replacement. Return a new Data. """
        if n > self.X.shape[0]:
            raise ValueError('n should not be larger than sizes of X')
        ind_x = util.subsample_ind( self.X.shape[0], n, seed )
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

### end Data class        


class DataSource(with_metaclass(ABCMeta, object)):
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

#  end DataSource

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
        with util.NumpySeedContext(seed=seed):
            d = len(self.mean)
            mean = self.mean
            variance = self.variance
            X = np.random.randn(n, d)*np.sqrt(variance) + mean
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
        with util.NumpySeedContext(seed=seed):
            mvn = stats.multivariate_normal(self.mean, self.cov)
            X = mvn.rvs(size=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

class DSIsoGaussianMixture(DataSource):
    """
    A DataSource implementing a Gaussian mixture in R^d where each component 
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
            raise ValueError('Number of components in means and variances do not match.')

        if pmix is None:
            pmix = old_div(np.ones(k),float(k))

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def sample(self, n, seed=29):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        sam_list = []
        with util.NumpySeedContext(seed=seed):
            # counts for each mixture component 
            counts = np.random.multinomial(n, pmix, size=1)

            # counts is a 2d array
            counts = counts[0]

            # For each component, draw from its corresponding mixture component.            
            for i, nc in enumerate(counts):
                # Sample from ith component
                sam_i = np.random.randn(nc, d)*np.sqrt(variances[i]) + means[i]
                sam_list.append(sam_i)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        return Data(sample)

# end of class DSIsoGaussianMixture

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
            raise ValueError('Number of components in means and variances do not match.')

        if pmix is None:
            pmix = old_div(np.ones(k),float(k))

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def sample(self, n, seed=29):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        sam_list = []
        with util.NumpySeedContext(seed=seed):
            # counts for each mixture component 
            counts = np.random.multinomial(n, pmix, size=1)

            # counts is a 2d array
            counts = counts[0]

            # For each component, draw from its corresponding mixture component.            
            for i, nc in enumerate(counts):
                # construct the component
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
                cov = variances[i]
                mnorm = stats.multivariate_normal(means[i], cov)
                # Sample from ith component
                sam_i = mnorm.rvs(size=nc)
                sam_list.append(sam_i)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        return Data(sample)

# end of DSGaussianMixture


class DSLaplace(DataSource):
    """
    A DataSource for a multivariate Laplace distribution.
    """
    def __init__(self, d, loc=0, scale=1):
        """
        loc: location 
        scale: scale parameter.
        Described in https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
        """
        assert d > 0
        self.d = d
        self.loc = loc
        self.scale = scale

    def sample(self, n, seed=4):
        with util.NumpySeedContext(seed=seed):
            X = np.random.laplace(loc=self.loc, scale=self.scale, size=(n, self.d))
            return Data(X)

class DSTDistribution(DataSource):
    """
    A DataSource for a univariate T-distribution.
    """
    def __init__(self, df):
        """
        df: degrees of freedom
        """
        assert df > 0
        self.df = df 

    def sample(self, n, seed=5):
        with util.NumpySeedContext(seed=seed):
            X = stats.t.rvs(df=self.df, size=n)
            X = X[:, np.newaxis]
            return Data(X)

# end class DSTDistribution


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
        return old_div(1.0,(1+np.exp(-x)))

    def _blocked_gibbs_next(self, X, H):
        """
        Sample from the mutual conditional distributions.
        """
        dh = H.shape[1]
        n, dx = X.shape
        B = self.B
        b = self.b

        # Draw H.
        XB2C = np.dot(X, self.B) + 2.0*self.c
        # Ph: n x dh matrix
        Ph = DSGaussBernRBM.sigmoid(XB2C)
        # H: n x dh
        H = (np.random.rand(n, dh) <= Ph)*2 - 1.0
        assert np.all(np.abs(H) - 1 <= 1e-6 )
        # Draw X.
        # mean: n x dx
        mean = old_div(np.dot(H, B.T),2.0) + b
        X = np.random.randn(n, dx) + mean
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
        with util.NumpySeedContext(seed=seed):
            X = np.random.randn(n, dx)
            H = np.random.randint(1, 2, (n, dh))*2 - 1.0

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

# end class DSGaussBernRBM

class DSISIPoissonLinear(DataSource):
    """
    A DataSource implementing non homogenous poisson process.
    """
    def __init__(self, b):
        """
        b: slope in of the linear function
        lambda_X = 1 + bX
        """
        if b < 0:
            raise ValueError('Parameter b must be non-negative.')
        self.b = b
    
    def nonhom_linear(self,size):
        b = self.b
        u = np.random.rand(size)
        if np.abs(b) <= 1e-8:
            F_l = -np.log(1-u)
        else:
            F_l = np.sqrt(-2.0/b*np.log(1-u)+old_div(1.0,(b**2)))-old_div(1.0,b)
        return F_l
        
    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = self.nonhom_linear(size=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSISIPoissonLinear

class DSISIPoissonSine(DataSource):
    """
    A DataSource implementing non homogenous poisson process with sine intensity.
    lambda = b*(1+sin(w*X))
    """
    def __init__(self, w, b, delta=0.001):
        """
        w: the frequency of sine function
        b: amplitude of intensity function
        """
        self.b = b
        self.w = w
        self.delta = delta
        
    def func(self,t):
        val = (t + old_div((1-np.cos(self.w*t)),self.w) )*self.b
        return val
    
    # slow step-by-step increment by assigned delta
    def find_time(self, x):
        t = 0.0
        while self.func(t) < x:
            t += self.delta
        return t

    # bisection search to find t value with accuracy delta
    def search_time(self, x):
        b = self.b
        w = self.w
        delta = self.delta
        t_old = 0.0
        t_new = b
        val_old = self.func(t_old)
        val_new = self.func(t_new)
        while np.abs(val_new-x) > delta:
            if val_new < x and t_old < t_new:
                t_old = t_new
                t_new = t_new * 2.0
                val_old = val_new
                val_new = self.func(t_new)
            elif val_new < x and t_old > t_new:
                temp = old_div((t_old + t_new), 2.0)
                t_old = t_new
                t_new = temp
                val_old = val_new
                val_new = self.func(t_new)
            elif val_new > x and t_old > t_new:
                t_old = t_new
                t_new = old_div(t_new, 2.0)
                val_old = val_new
                val_new = self.func(t_new)
            elif val_new > x and t_old < t_new:
                temp = old_div((t_old + t_new), 2.0)
                t_old = t_new
                t_new = temp
                val_old = val_new
                val_new = self.func(t_new)
        t = t_new
        return t
        
    def nonhom_sine(self,size=1000):
        u = np.random.rand(size)
        x = -np.log(1-u)
        t = np.zeros(size)
        for i in range(size):
            t[i] = self.search_time(x[i])
        return t

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = self.nonhom_sine(size=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSISIPoissonSine


class DSGamma(DataSource):
    """
    A DataSource implementing gamma distribution.
    """
    def __init__(self, alpha, beta=1.0):
        """
        alpha: shape of parameter
        beta: scale
        """
        self.alpha = alpha
        self.beta = beta

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = stats.gamma.rvs(self.alpha, size=n, scale = old_div(1.0,self.beta))
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSGamma


class DSLogGamma(DataSource):
    """
    A DataSource implementing the transformed gamma distribution.
    """
    def __init__(self, alpha, beta=1.0):
        """
        alpha: shape of parameter
        beta: scale
        """
        self.alpha = alpha
        self.beta = beta

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = np.log(stats.gamma.rvs(self.alpha, size=n, scale = old_div(1.0,self.beta)))
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSLogGamma

class DSISILogPoissonLinear(DataSource):
    """
    A DataSource implementing non homogenous poisson process.
    """
    def __init__(self, b):
        """
        b: slope in of the linear function
        lambda_X = 1 + bX
        """
        if b < 0:
            raise ValueError('Parameter b must be non-negative.')
        self.b = b
    
    def nonhom_linear(self,size):
        b = self.b
        u = np.random.rand(size)
        if np.abs(b) <= 1e-8:
            F_l = -np.log(1-u)
        else:
            F_l = np.sqrt(-2.0/b*np.log(1-u)+old_div(1.0,(b**2)))-old_div(1.0,b)
        return F_l
        
    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = np.log(self.nonhom_linear(size=n))
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSISILogPoissonLinear

class DSISIPoisson2D(DataSource):
    """
     A DataSource implementing non homogenous poisson process.
    """
    def __init__(self, intensity = 'quadratic', w=1.0):
        """
        lambda_(X,Y) = X^2 + Y^2
        lamb_bar: upper-bound used in rejection sampling
        """
        self.w = w
        if intensity == 'quadratic':
            self.intensity = self.quadratic_intensity
        elif intensity == 'sine':
            self.intensity = self.sine_intensity
        elif intensity == 'xsine':
            self.intensity = self.cross_sine_intensity
        else:
            raise ValueError('Not intensity function found')


    def quadratic_intensity(self, X):
        intensity = self.lamb_bar*np.sum(X**2,1)
        return intensity

    def sine_intensity(self, X):
        intensity = self.lamb_bar*np.sum(np.sin(self.w*X*np.pi),1)
        return intensity

    def cross_sine_intensity(self, X):
        intensity = self.lamb_bar*np.prod(np.sin(self.w*X*np.pi),1)
        return intensity

    def inh2d(self, lamb_bar = 100000):
        self.lamb_bar = lamb_bar
        N = np.random.poisson(2*self.lamb_bar)
        X = np.random.rand(N,2)
        intensity = self.intensity(X)
        u = np.random.rand(N)
        lamb_T = old_div(intensity,lamb_bar)
        X_acc = X[u < lamb_T]
        return X_acc

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = self.inh2d(lamb_bar=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSISIPoisson2D

class DSISISigmoidPoisson2D(DataSource):
    """
     A DataSource implementing non homogenous poisson process.
    """
    def __init__(self, intensity = 'quadratic', w=1.0, a=1.0):
        """
        lambda_(X,Y) = a*X^2 + Y^2
        X = 1/(1+exp(s))
        Y = 1/(1+exp(t))
        X, Y \in [0,1], s,t \in R
        """
        self.a = a
        self.w = w
        if intensity == 'quadratic':
            self.intensity = self.quadratic_intensity
        elif intensity == 'sine':
            self.intensity = self.sine_intensity
        elif intensity == 'xsine':
            self.intensity = self.cross_sine_intensity
        else:
            raise ValueError('Not intensity function found')

    def quadratic_intensity(self, X):
        intensity = self.lamb_bar*np.average(X**2, axis=1, weights=[self.a,1])
        return intensity

    def cross_sine_intensity(self, X):
        intensity = self.lamb_bar*np.prod(np.sin(self.w*X*np.pi),1)
        return intensity

    def sine_intensity(self, X):
        intensity = self.lamb_bar*np.sum(np.sin(self.w*X*np.pi),1)
        return intensity


    def inh2d(self, lamb_bar = 100000):
        self.lamb_bar = lamb_bar
        N = np.random.poisson(2*self.lamb_bar)
        X = np.random.rand(N,2)
        intensity = self.intensity(X)
        u = np.random.rand(N)
        lamb_T = old_div(intensity,lamb_bar)
        X_acc = X[u < lamb_T]
        return X_acc

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = np.log(old_div(1,self.inh2d(lamb_bar=n))-1)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)
# end class DSISISigmoidPoisson2D

class DSPoisson2D(DataSource):
    """
     A DataSource implementing non homogenous poisson process.
    """
    def __init__(self, w = 1.0):
        """
        2D spatial poission process with default lambda_(X,Y) = sin(w*pi*X)+sin(w*pi*Y)
        """
        self.w = w

    def gmm_sample(self, mean=None, w=None, N=10000,n=10,d=2,seed=10):
        np.random.seed(seed)
        self.d = d
        if mean is None:
            mean = np.random.randn(n,d)*10
        if w is None:
            w = np.random.rand(n)
        w = old_div(w,sum(w))
        multi = np.random.multinomial(N,w)
        X = np.zeros((N,d))
        base = 0
        for i in range(n):
            X[base:base+multi[i],:] = np.random.multivariate_normal(mean[i,:], np.eye(self.d), multi[i])
            base += multi[i]
        
        llh = np.zeros(N)
        for i in range(n):
            llh += w[i] * stats.multivariate_normal.pdf(X, mean[i,:], np.eye(self.d))
        #llh = llh/sum(llh)
        return X, llh

    def const(self, X):
        return np.ones(len(X))*8

    def lamb_sin(self, X):
        return np.prod(np.sin(self.w*np.pi*X),1)*15

    def rej_sample(self, X, llh, func = None):
        if func is None:
            self.func = self.lamb_sin
        rate = old_div(self.func(X),llh)
        u = np.random.rand(len(X))
        X_acc = X[u < rate]
        return X_acc

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X_gmm, llh = self.gmm_sample(N=n)
            X = X_gmm
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

# end class DSPoisson2D


class DSResample(DataSource):
    """
    A DataSource which subsamples without replacement from the specified 
    numpy array (n x d).
    """

    def __init__(self, X):
        """
        X: n x d numpy array. n = sample size. d = input dimension
        """
        self.X = X

    def sample(self, n, seed=900, return_ind = False):
        X = self.X
        if n > X.shape[0]:
            # Sample more than what we have
            raise ValueError('Cannot subsample n={0} from only {1} points.'.format(n, X.shape[0]))
        dat = Data(self.X)
        return dat.subsample(n, seed=seed, return_ind = return_ind)

    def dim(self):
        return self.X.shape[1]

# end class DSResample

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
        if sigma2 <= 0 :
            raise ValueError('sigma2 must be > 0')
        self.freqs = freqs

    def sample(self, n, seed=872):
        """
        Rejection sampling.
        """
        d = len(self.freqs)
        sigma2 = self.sigma2
        freqs = self.freqs
        with util.NumpySeedContext(seed=seed):
            # rejection sampling
            sam = np.zeros((n, d))
            # sample block_size*d at a time.
            block_size = 500
            from_ind = 0
            while from_ind < n:
                # The proposal q is N(0, sigma2*I)
                X = np.random.randn(block_size, d)*np.sqrt(sigma2)
                q_un = np.exp(old_div(-np.sum(X**2, 1),(2.0*sigma2)))
                # unnormalized density p
                p_un = q_un*(1+np.prod(np.cos(X*freqs), 1))
                c = 2.0
                I = stats.uniform.rvs(size=block_size) < old_div(p_un,(c*q_un))

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





