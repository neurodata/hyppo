import numpy as np
from hyppo.ksample import SmoothCFTest


num_samples = 500
dimensions = 10
X = np.random.randn(num_samples, dimensions)
Y = np.random.randn(num_samples, dimensions)
print(X.shape, Y.shape)

smoothCF = SmoothCFTest()
p_val = smoothCF.test(X, Y)
