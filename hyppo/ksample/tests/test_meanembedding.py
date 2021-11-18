from hyppo.ksample import MeanEmbeddingTest
import numpy as np


num_samples = 500
dimensions = 10
X = np.random.randn(num_samples, dimensions)
Y = np.random.randn(num_samples, dimensions)
print(X.shape, Y.shape)

MEtest = MeanEmbeddingTest()
p_val = MEtest.test(X,Y)




