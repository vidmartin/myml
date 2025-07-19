
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

    