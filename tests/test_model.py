""" Tests for the model. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from pyhacrf import Hacrf
from pyhacrf import _Model


class TestHacrf(unittest.TestCase):
    def test_initialize_parameters(self):
        states = [(0,), 1]
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0))]
        state_machine = (states, transitions)
        X = [np.zeros((6, 7, 3))]

        actual_parameters = Hacrf._initialize_parameters(state_machine, X[0].shape[2])
        expected_parameter_shape = (3, 5)
        self.assertEqual(actual_parameters.shape, expected_parameter_shape)

    def test_default_state_machine(self):
        classes = ['a', 'b']
        expected_state_machine = ([(0, 1)],
                                  [(0, 0, (1, 1)),
                                   (1, 1, (1, 1)),
                                   (0, 0, (0, 1)),
                                   (1, 1, (0, 1)),
                                   (0, 0, (1, 0)),
                                   (1, 1, (1, 0))])
        expected_states_to_classes = {0: 'a', 1: 'b'}
        actual_state_machine, actual_states_to_classes = Hacrf._default_state_machine(classes)
        self.assertEqual(actual_state_machine, expected_state_machine)
        self.assertEqual(actual_states_to_classes, expected_states_to_classes)


class TestModel(unittest.TestCase):
    def test_build_lattice(self):
        states = [(0, 1), 3]
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0)),
                       (0, 3, lambda i, j, k: (0, 2))]
        state_machine = (states, transitions)
        x = np.zeros((2, 3, 9))
        #               #     ________
        # 1.  .  .      # 1  0 - 10 - 31
        #               #    | /_______
        # 0.  .  .      # 0  0 -- 1    3
        #  0  1  2      #    0    1    2
        actual_lattice = _Model._build_lattice(x, state_machine)
        expected_lattice = [(0, 0, 0),
                            (0, 0, 0, 1, 0, 1, 1),
                            (0, 0, 0, 2, 0, 3, 3),
                            (0, 0, 1),
                            (0, 0, 1, 0, 0, 0, 2),
                            (0, 0, 1, 1, 0, 0, 0),
                            (0, 1, 1),
                            (0, 2, 3),
                            (1, 0, 0),
                            (1, 0, 1, 1, 0, 1, 1),
                            (1, 0, 1, 2, 0, 3, 3),
                            (1, 1, 0),
                            (1, 1, 1),
                            (1, 1, 1, 2, 0, 1, 1),
                            (1, 2, 1),
                            (1, 2, 3)]
        self.assertEqual(actual_lattice, expected_lattice)

if __name__ == '__main__':
    unittest.main()
