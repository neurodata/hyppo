import numpy as np
from hyppo.ksample import SmoothCFTest

np.random.seed(1234)
x = np.random.randn(500, 10)
y = np.random.randn(500, 10)
stat, pvalue = SmoothCFTest(random_state=1234).test(x, y)
print(stat, pvalue)
