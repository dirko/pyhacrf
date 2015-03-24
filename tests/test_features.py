""" Tests for the feature extraction. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from pyhacrf import StringPairFeatureExtractor


class TestStringPairFeatureExtractor(unittest.TestCase):
    def test_transform(self):
        s1 = "kat1"
        s2 = "cat2"
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789,./;\'\-=<>?:"|_+!@#$%^&*() '
        expected_x = np.zeros((4, 4, 5 + len(chars)**2))
        # TODO: activate the correct entries in expected_x

        test_extractor = StringPairFeatureExtractor(start=True, match=True, numeric=True, transition=True)
        actual_X = test_extractor.fit_transform([(s1, s2)])

        assert_array_almost_equal(expected_x, actual_X[0])


if __name__ == '__main__':
    unittest.main()
