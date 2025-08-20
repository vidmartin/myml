
import unittest
import numpy as np
import torch
import utils

class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)

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

    def test_logsumexp(self):
        X = self._rng.random((10, 10), dtype=np.float32)
        ref = torch.logsumexp(torch.tensor(X, requires_grad=False), -1)
        out = utils.log_sum_exp(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 10
        ref = torch.logsumexp(torch.tensor(X, requires_grad=False), -1)
        out = utils.log_sum_exp(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 100
        ref = torch.logsumexp(torch.tensor(X, requires_grad=False), -1)
        out = utils.log_sum_exp(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 1000
        ref = torch.logsumexp(torch.tensor(X, requires_grad=False), -1)
        out = utils.log_sum_exp(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

    def test_softmax(self):
        X = self._rng.random((10, 10), dtype=np.float32)
        ref = torch.softmax(torch.tensor(X, requires_grad=False), -1)
        out = utils.softmax(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 10
        ref = torch.softmax(torch.tensor(X, requires_grad=False), -1)
        out = utils.softmax(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 100
        ref = torch.softmax(torch.tensor(X, requires_grad=False), -1)
        out = utils.softmax(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

        X = self._rng.random((10, 10), dtype=np.float32) * 1000
        ref = torch.softmax(torch.tensor(X, requires_grad=False), -1)
        out = utils.softmax(X)
        self.assertTrue(np.allclose(ref.detach().numpy(), out, atol=0.001), f"got {out} but wanted {ref}, naive calculation gives {np.log(np.exp(X).sum(-1))}")

    def test_instance_memo(self):
        class Counter:
            def __init__(self):
                self._counter = 0
            @utils.instance_memo
            def inc_and_ret(self):
                self._counter += 1
                return self._counter
            @utils.instance_memo
            def add_and_ret(self, add: int):
                self._counter += add
                return self._counter
            
        c1, c2 = Counter(), Counter()

        self.assertEqual(c1.add_and_ret(3), 3)
        self.assertEqual(c2.add_and_ret(4), 4)

        self.assertEqual(c1.inc_and_ret(), 4)
        self.assertEqual(c2.add_and_ret(4), 4)

        self.assertEqual(c1.add_and_ret(4), 8)
        self.assertEqual(c2.inc_and_ret(), 5)

        self.assertEqual(c1.inc_and_ret(), 4)
        self.assertEqual(c2.add_and_ret(3), 8)

        self.assertEqual(c1.add_and_ret(3), 3)
        self.assertEqual(c2.inc_and_ret(), 5)

    def test_padding(self):
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        self.assertTrue(np.allclose(utils.pad_l(arr, (0, 0), 0.0), arr))
        self.assertTrue(np.allclose(utils.pad_r(arr, (0, 0), 0.0), arr))
        self.assertTrue(np.allclose(utils.pad_lr(arr, (0, 0), 0.0), arr))

        self.assertTrue(np.allclose(
            utils.pad_l(arr, (1, 0), 0.0),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_r(arr, (1, 0), 0.0),
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_lr(arr, (1, 0), 0.0),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_l(arr, (0, 1), 0.0),
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 4.0, 5.0, 6.0],
                    [0.0, 7.0, 8.0, 9.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_r(arr, (0, 1), 0.0),
            np.array(
                [
                    [1.0, 2.0, 3.0, 0.0],
                    [4.0, 5.0, 6.0, 0.0],
                    [7.0, 8.0, 9.0, 0.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_lr(arr, (0, 1), 0.0),
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 0.0],
                    [0.0, 4.0, 5.0, 6.0, 0.0],
                    [0.0, 7.0, 8.0, 9.0, 0.0],
                ]
            )
        ))

        self.assertTrue(np.allclose(
            utils.pad_lr(arr, (2, 2), 0.0),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ))


    def test_unpadding(self):
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        self.assertTrue(np.allclose(utils.unpad_l(utils.pad_l(arr, (0, 0), 0.0), (0, 0)), arr))
        self.assertTrue(np.allclose(utils.unpad_r(utils.pad_r(arr, (0, 0), 0.0), (0, 0)), arr))
        self.assertTrue(np.allclose(utils.unpad_lr(utils.pad_lr(arr, (0, 0), 0.0), (0, 0)), arr))

        self.assertTrue(np.allclose(utils.unpad_l(utils.pad_l(arr, (1, 0), 0.0), (1, 0)), arr))
        self.assertTrue(np.allclose(utils.unpad_r(utils.pad_r(arr, (1, 0), 0.0), (1, 0)), arr))
        self.assertTrue(np.allclose(utils.unpad_lr(utils.pad_lr(arr, (1, 0), 0.0), (1, 0)), arr))

        self.assertTrue(np.allclose(utils.unpad_l(utils.pad_l(arr, (0, 1), 0.0), (0, 1)), arr))
        self.assertTrue(np.allclose(utils.unpad_r(utils.pad_r(arr, (0, 1), 0.0), (0, 1)), arr))
        self.assertTrue(np.allclose(utils.unpad_lr(utils.pad_lr(arr, (0, 1), 0.0), (0, 1)), arr))

        self.assertTrue(np.allclose(utils.unpad_lr(utils.pad_lr(arr, (2, 2), 0.0), (2, 2)), arr))

    def test_padded_array_consistency(self):
        arr = self._rng.random((20, 20, 20))

        self.assertTrue(np.allclose(
            utils.pad_lr(arr, (5, 5), 0.0),
            utils.pad_r(utils.pad_l(arr, (5, 5), 0.0), (5, 5), 0.0)
        ))
        self.assertTrue(np.allclose(
            utils.pad_lr(arr, (5, 5), 0.0),
            utils.pad_l(utils.pad_r(arr, (5, 5), 0.0), (5, 5), 0.0)
        ))

    def test_roll_varied(self):
        arr = np.arange(25).reshape((5, 5))

        out = utils.roll_varied(arr, 1, 0, (3,0,1,3,2))
        ref = np.stack([
            np.roll(arr[0,:], 3),
            np.roll(arr[1,:], 0),
            np.roll(arr[2,:], 1),
            np.roll(arr[3,:], 3),
            np.roll(arr[4,:], 2),
        ])
        self.assertTrue(np.all(out == ref), f"got {out}, wanted {ref}")

        out = utils.roll_varied(arr, 0, 1, (3,0,1,3,2))
        ref = np.stack([
            np.roll(arr[:,0], 3),
            np.roll(arr[:,1], 0),
            np.roll(arr[:,2], 1),
            np.roll(arr[:,3], 3),
            np.roll(arr[:,4], 2),
        ]).T
        self.assertTrue(np.all(out == ref), f"got {out}, wanted {ref}")
