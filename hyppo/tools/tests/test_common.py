import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_less,
    assert_equal,
    assert_raises,
)

from ..common import _PermTree


class TestPermTree:
    """
    Tests that permutations are restricted correctly
    """

    def test_within_permutatins(self):
        np.random.seed(1)
        # i.e. case: y = np.asarray([0,1,0,1,0,1])
        blocks = np.vstack(
            (
                [-1, 1],
                [-1, 2],
                [-2, 1],
                [-2, 2],
                [-3, 1],
                [-3, 2],
            )
        )
        perm_tree = _PermTree(blocks)
        original_indices = perm_tree.original_indices()
        perms = np.asarray([perm_tree.permute_indices() for _ in range(10)])
        assert_array_less(np.abs(original_indices - perms), 2)
        assert_allclose(
            np.mean(perms, axis=0), [0.5, 0.5, 2.5, 2.5, 4.5, 4.5], rtol=0, atol=0.2
        )

    def test_across_permutations(self):
        np.random.seed(0)
        # i.e. case: y = np.asarray([0,0,1,1,2,2])
        blocks = np.vstack(
            (
                [1, -1],
                [1, -2],
                [2, -1],
                [2, -2],
                [3, -1],
                [3, -2],
            )
        )
        perm_tree = _PermTree(blocks)
        _ = perm_tree.original_indices()
        perms = np.asarray([perm_tree.permute_indices() for _ in range(100)])
        assert_equal(perms[0][1::2] - perms[0][::2], [1, 1, 1])
        assert_allclose(np.mean(perms, axis=0), [2, 3, 2, 3, 2, 3], rtol=0, atol=0.2)

    def test_fixed_permutation(self):
        np.random.seed(0)
        blocks = [-1, -2, -3, -4]
        perm_tree = _PermTree(blocks)
        assert_equal(perm_tree.permute_indices(), perm_tree.original_indices())

    def test_semifixed(self):
        np.random.seed(1)
        blocks = [1, 2, -3, -4]
        perm_tree = _PermTree(blocks)
        perms = np.asarray([perm_tree.permute_indices() for _ in range(10)])
        assert_equal(perms[0][2:], perm_tree.original_indices()[2:])
        assert_allclose(np.mean(perms, axis=0)[:2], [0.5, 0.5], rtol=0, atol=0.2)

    def test_non_int_inputs(self):
        blocks = ["a", "b", "c"]
        _ = _PermTree(blocks)

    def test_block_sizes(self):
        blocks = [1, 1, 2]
        assert_raises(ValueError, _PermTree, blocks)
