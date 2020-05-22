import unittest
import torch
import numpy as np

batch_size = 6
eta = 0.1


class TestExpAdvantage(unittest.TestCase):
    def test_exp_advantage(self):
        adv = torch.rand(batch_size, 2)
        adv_baseline = torch.max(adv, 0)[0]
        exp_adv = torch.exp((adv - adv_baseline) / eta)
        normalization = torch.mean(exp_adv, 0)
        psi = exp_adv / normalization
        adv_: np.ndarray = adv.numpy()
        psi_: np.ndarray = psi.numpy()

        self.assertEqual(adv_.shape, psi_.shape)
        for i in range(batch_size - 1):
            expected = adv_[i] > adv_[i + 1]
            actual = psi_[i] > psi_[i + 1]
            np.testing.assert_equal(expected, actual)
