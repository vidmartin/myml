
from typing import *
import unittest
import numpy as np
import permutation

class PermutationTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)
    
    def test_inverse(self):
        for i in range(10):
            array = np.arange(i)
            self._rng.shuffle(array)
            perm = permutation.Permutation(tuple(array))
            self.assertEqual(
                perm.inverse().compose(perm),
                permutation.Permutation.identity(i)
            )
            self.assertEqual(
                perm.compose(perm.inverse()),
                permutation.Permutation.identity(i)
            )

    def test_bringing(self):
        arr = np.arange(5 ** 5).reshape((5,) * 5)

        self.assertEqual(
            permutation.Permutation.bring_to_front((), 5),
            permutation.Permutation.identity(5),
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_front((3,), 5).permutation)[0,1,2,3,4],
            arr[1,2,3,0,4]
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_back((3,), 5).permutation)[0,1,2,3,4],
            arr[0,1,2,4,3]
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_front((-1,), 5).permutation)[0,1,2,3,4],
            arr[1,2,3,4,0]
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_back((-2,), 5).permutation)[0,1,2,3,4],
            arr[0,1,2,4,3]
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_front((1,2), 5).permutation)[0,1,2,3,4],
            arr[2,0,1,3,4]
        )
        self.assertEqual(
            arr.transpose(permutation.Permutation.bring_to_back((1,2), 5).permutation)[0,1,2,3,4],
            arr[0,3,4,1,2]
        )
