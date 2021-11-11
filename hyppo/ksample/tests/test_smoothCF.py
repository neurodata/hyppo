import numpy as np
from hyppo.ksample import SmoothCFTest


np.random.seed(120)

scale = 1
num_samples = 500
dimensions = 10
X = np.random.randn(num_samples, dimensions)
Y = np.random.randn(num_samples, dimensions)
print(X.shape, Y.shape)

test = SmoothCFTest()
print(test.test(X,Y))