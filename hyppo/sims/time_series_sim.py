import numpy as np


def indep_ar(n, lag=1, phi=0.5, sigma=1):
    # X_t and Y_t are univarite AR(1) with phi = 0.5 for both series.
    # Noise follows N(0, sigma2).

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = epsilons
    y = etas

    # AR process.
    for t in range(lag, n):
        x[t] = phi * x[t - lag] + epsilons[t]
        y[t] = phi * y[t - lag] + etas[t]

    return x, y


def cross_corr_ar(n, lag=1, phi=0.5, sigma=1):
    # X_t and Y_t are together a bivarite AR(1) with Phi = [0 0.5; 0.5 0].
    # Noise follow N(0, sigma2).

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = epsilons
    y = etas

    for t in range(lag, n):
        x[t] = phi * y[t - lag] + epsilons[t]
        y[t] = phi * x[t - lag] + etas[t]

    return x, y


def nonlinear_process(n, lag=1, phi=1, sigma=1):
    # X_t and Y_t are together a bivarite nonlinear process.
    # Noise follow N(0, sigma2).

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = np.zeros(n)
    y = etas

    for t in range(lag, n):
        x[t] = phi * epsilons[t] * y[t - lag]

    return x, y
