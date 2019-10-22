import numpy as np


def _gen_coeffs(p):
    return np.array([1/(i+1) for i in range(p)]).reshape(-1, 1)


def _random_uniform(n, p, low=-1, high=1):
    return np.array(np.random.uniform(low, high, size=(n, p)))


def linear(n, p, noise=1, low=-1, high=1):
    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    kappa = int(p == 1)
    y = x.dot(coeffs) + kappa*noise

    return x, y


def spiral(n, p, noise=0.4, low=0, high=5):
    unif = _random_uniform(n, p=1, low=low, high=high)
    sinusoid = np.cos(np.pi * unif).reshape(-1, 1)
    y = unif * np.sin(np.pi * unif).reshape(-1, 1)

    x = np.zeros((n, p))
    if p > 1:
        for i in range(p-1):
            x[:, i] = np.squeeze(y * (sinusoid ** i))
    x[:, p-1] = np.squeeze(unif * sinusoid)

    guass_noise = np.random.normal(0, 1, size=(n, 1))
    y = y + noise*p*guass_noise

    return x, y
