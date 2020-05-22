import unittest
from scipy.special import logsumexp, softmax
import numpy as np


def logmeanexp(inputs, dim=1):
    input_max = inputs.max(dim, keepdims=True)
    return np.log(np.exp(inputs - input_max).mean(dim)) + input_max


class TestLogSumExp(unittest.TestCase):
    def test_lse(self):
        x = np.random.uniform(size=[2, 3], low=10_000, high=20_000)
        expected = logsumexp(x, 0, keepdims=True)
        max_x = np.max(x, 0, keepdims=True)
        actual = max_x + np.log(np.sum(np.exp(x - max_x), 0))
        print(expected)
        print(actual)
        np.testing.assert_almost_equal(expected, actual)


    def test_se(self):
        x = np.random.uniform(size=[2, 3], low=10, high=20)
        expected = logsumexp(x, 0, keepdims=True)
        max_x = np.max(x, 0, keepdims=True)
        actual = max_x + np.log(np.sum(np.exp(x - max_x), 0))
        print(expected)
        print(actual)
        np.testing.assert_almost_equal(expected, actual)
        expected = np.exp(expected)
        actual = np.exp(actual)
        print(expected)
        print(actual)
        np.testing.assert_almost_equal(expected, actual)


class TestLogMeanExp(unittest.TestCase):
    def test_lme(self):
        x = np.random.uniform(size=[3, 4], low=10_000, high=20_000)
        print(logmeanexp(x))
        max_x = np.max(x, 1, keepdims=True)
        actual = max_x + np.log(np.mean(np.exp(x - max_x), 1))
        print(actual)


if __name__ == '__main__':
    unittest.main()
