""" Tests for the model. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from pyhacrf import Hacrf


class TestHacrf(unittest.TestCase):
    def test_initialize_parameters(self):
        state_machine = [(0, 0, (1, 1)),
                         (0, 1, (0, 1)),
                         (0, 0, (1, 0))]
        X = [np.zeros((6, 7, 3))]
        classes = {'y1': 0, 'y2': 1}

        actual_parameters = Hacrf._initialize_parameters(state_machine, X[0].shape[2], classes)
        expected_parameter_shape = (3, 5, 2)
        self.assertEqual(actual_parameters.shape, expected_parameter_shape)


if __name__ == '__main__':
    unittest.main()
