
import unittest
import numpy as np
import utils

class UtilsTestCase(unittest.TestCase):
    def test_one_hot_encode(self):
        self.assertTrue(
            np.array_equal(
                utils.one_hot_encode(
                    np.array([1,3,0,1,2]), 5
                ),
                np.array([
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ])
            )
        )
    
        self.assertTrue(
            np.array_equal(
                utils.one_hot_encode(
                    np.array([
                        [0, 1, 0, 2, 2],
                        [2, 1, 1, 0, 0]
                    ]), 3
                ),
                np.array([
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                ])
            )
        )
