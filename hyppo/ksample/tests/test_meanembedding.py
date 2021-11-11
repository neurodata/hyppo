import numpy as np
from hyppo.ksample import MeanEmbeddingTest
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_approx_equal
from hyppo.tools import linear, multimodal_independence, power, spiral


np.random.seed(120)
num_samples = 500
dimensions = 10
X = np.random.randn(num_samples, dimensions)
Y = np.random.randn(num_samples, dimensions)
print(X.shape, Y.shape)

test = MeanEmbeddingTest()
print(test.test(X,Y))


