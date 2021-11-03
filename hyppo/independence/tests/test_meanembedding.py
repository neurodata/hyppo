import numpy as np
from hyppo.independence import MeanEmbeddingTest

np.random.seed(120)
num_samples = 500
dimensions = 10
X = np.random.randn(num_samples, dimensions)
Y = np.random.randn(num_samples, dimensions)
print(X.shape, Y.shape)

test = MeanEmbeddingTest()
print(test.test(X,Y))