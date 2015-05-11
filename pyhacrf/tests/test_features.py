""" Tests for the feature extraction. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from pyhacrf import StringPairFeatureExtractor


class TestStringPairFeatureExtractor(unittest.TestCase):
    def test_transform_binary(self):
        s1 = "kat1"
        s2 = "cat2"
        # 1 . . . n
        # t . . m .
        # a . m . .
        # k . . . .
        #   c a t 2
        expected_x = np.zeros((4, 4, 4))
        expected_x[:, :, 0] = 2.0
        expected_x[:, 0, 1] = 1.0
        expected_x[0, :, 1] = 1.0
        expected_x[1, 1, 2] = 1.0
        expected_x[2, 2, 2] = 1.0
        expected_x[3, 3, 3] = 1.0

        test_extractor = StringPairFeatureExtractor(bias=2.0, start=True, match=True, numeric=True)
        actual_X = test_extractor.fit_transform([(s1, s2)])

        assert_array_almost_equal(expected_x, actual_X[0])

    def test_transform_transition(self):
        s1 = "ba"
        s2 = "ca"
        # a . .
        # b . .
        #   c a
        chars = StringPairFeatureExtractor.CHARACTERS
        nchars = len(chars)
        print nchars
        expected_x = np.zeros((2, 2, len(chars)**2 + 1))
        expected_x[:, :, 0] = 1.0
        expected_x[0, 0, 2 + nchars * 1 + 1] = 1.0  # b->c
        expected_x[0, 1, 0 + nchars * 1 + 1] = 1.0  # b->a
        expected_x[1, 0, 2 + nchars * 0 + 1] = 1.0  # a->c
        expected_x[1, 1, 0 + nchars * 0 + 1] = 1.0  # a->a

        test_extractor = StringPairFeatureExtractor(transition=True)
        actual_X = test_extractor.fit_transform([(s1, s2)])

        assert_array_almost_equal(expected_x, actual_X[0])

if __name__ == '__main__':
    unittest.main()
