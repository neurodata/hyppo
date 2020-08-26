from hyppo._utils import _PermTree, _PermNode
from hyppo.independence import Dcorr
import numpy as np

# within_y = np.asarray([
#     0,1,0,1,0,1
# ])
# within_blocks = np.vstack((
#     [1],
#     [1],
#     [2],
#     [2],
#     [3],
#     [3],
# ))
# perm_tree = _PermTree(within_blocks, within_y)
# print('-- Within --')
# print(perm_tree.original_indices())
# print('------------')
# for _ in range(5):
#     print(within_y[perm_tree.permute_indices()])


# across_y = np.asarray([
#     0,0,1,1,1,1
# ])
# across_blocks = np.vstack((
#     [1, 1],
#     [1, 1],
#     [2, 2],
#     [2, 2],
#     [2, 3],
#     [2, 3],
# ))
# perm_tree = _PermTree(across_blocks, across_y)
# print('-- Within --')
# print(perm_tree.original_indices())
# print('------------')
# for _ in range(5):
#     print(across_y[perm_tree.permute_indices()])

X = np.random.normal(0,1,(6,10))
y = np.asarray([
    0,1,0,1,0,1
])
perm_blocks = np.vstack((
    [1],
    [1],
    [2],
    [2],
    [3],
    [3],
))

dcorr = Dcorr()
print(dcorr.test(X, y, reps=1, perm_blocks=perm_blocks))
print(dcorr.null_dist)