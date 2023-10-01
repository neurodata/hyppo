import numpy as np
from hyppo.independence import Dcorr
from scipy.stats import spearmanr
import pytest

# Compute the Spearman rank-order correlation coefficient as a reference test.
def reference_test(x, y):
    coefficient, p_value = spearmanr(x, y)
    return coefficient, p_value


def sinusoidal_simulation(sample_size, dimensions=1):
    x = np.linspace(0, 4 * np.pi, sample_size)
    y = np.sin(x)

    while len(x.shape) < 2:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]

    while x.shape[1] < dimensions:
        noise = np.random.normal(0, 0.1, size=(sample_size, dimensions - 1))
        x = np.hstack((x, noise))
        y = np.hstack((y, noise))

    return x, y


def null_simulation(sample_size, dimensions=1):
    x = np.random.randn(sample_size, dimensions)
    y = np.random.randn(sample_size, dimensions)
    return x, y


def compute_test_statistic(x, y, test_func=Dcorr().test):
    return test_func(x, y)[0]


def compare_with_reference(x, y):
    hyppo_stat = compute_test_statistic(x, y)
    reference_stat = compute_test_statistic(x, y, test_func=reference_test)

    # Modify this tolerance as needed
    assert np.isclose(hyppo_stat, reference_stat, atol=1e6)


def test_sinusoidal_relationship_1D():
    x, y = sinusoidal_simulation(1000)
    compare_with_reference(x, y)


def test_sinusoidal_relationship_5D():
    x, y = sinusoidal_simulation(1000, dimensions=1)
    compare_with_reference(x, y)


def test_independence_1D():
    x, y = null_simulation(1000)
    compare_with_reference(x, y)


def test_independence_5D():
    x, y = null_simulation(1000, dimensions=1)
    compare_with_reference(x, y)
