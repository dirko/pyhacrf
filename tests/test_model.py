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
        expected_parameter_shape = (5, 3)
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
        n_states = 4  # Because 3 is the max
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0)),
                       (0, 3, lambda i, j, k: (0, 2))]
        state_machine = (states, transitions)
        x = np.zeros((2, 3, 9))
        #               #     ________
        # 1.  .  .      # 1  0 - 10 - 31
        #               #    | /_______
        # 0.  .  .      # 0 10 -- 1    3
        #  0  1  2      #    0    1    2
        #
        # 1(0, 1), 3(0, 2), 1(1, 1), 1(0, 0) should be pruned because they represent partial alignments.
        # Only nodes that are reachable by stepping back from (1, 2) must be included in the lattice.
        actual_lattice = _Model._build_lattice(x, state_machine)
        expected_lattice = [(0, 0, 0),
                            (0, 0, 1, 0, 0, 0, 2 + n_states),
                            (0, 0, 1, 1, 0, 0, 0 + n_states),
                            (1, 0, 0),
                            (1, 0, 1, 2, 0, 3, 3 + n_states),
                            (1, 1, 0),
                            (1, 1, 1, 2, 0, 1, 1 + n_states),
                            (1, 2, 1),
                            (1, 2, 3)]
        self.assertEqual(actual_lattice, expected_lattice)

    def test_forward(self):
        states = [(0, 1), 2]
        n_states = 3
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0)),
                       (0, 2, lambda i, j, k: (0, 2))]
        state_machine = (states, transitions)
        states_to_classes = {0: 'a', 1: 'a', 2: 'b'}  # Dummy
        parameters = np.array(range(-7, 7)).reshape((7, 2))
        # parameters =
        # 0([[-7, -6],
        # 1  [-5, -4],
        # 2  [-3, -2],
        # 3  [-1,  0],
        # 4  [ 1,  2],
        # 5  [ 3,  4],
        # 6  [ 5,  6]])
        x = np.array([[[0, 1],
                       [1, 0],
                       [2, 1]],
                      [[0, 1],
                       [1, 0],
                       [1, 0]]])
        y = 'a'
        # Expected lattice:
        #               #     ________
        # 1.  .  .      # 1  0  __0 - 21
        #               #    | /
        # 0.  .  .      # 0  0
        #  0  1  2      #    0    1    2
        expected_alpha = {
            (0, 0, 0): np.exp(-6),
            (0, 0, 1, 0, 0, 0, 5): np.exp(-6) * np.exp(4),
            (0, 0, 1, 1, 0, 0, 3): np.exp(-6) * np.exp(-1),
            (1, 0, 0): np.exp(-6) * np.exp(4) * np.exp(-6),
            (1, 0, 1, 2, 0, 2, 6): np.exp(-6) * np.exp(4) * np.exp(-6) * np.exp(5),
            (1, 1, 0): np.exp(-6) * np.exp(-1) * np.exp(-7),
            (1, 1, 1, 2, 0, 1, 4): np.exp(-6) * np.exp(-1) * np.exp(-7) * np.exp(1),
            (1, 2, 1): np.exp(-6) * np.exp(-1) * np.exp(-7) * np.exp(1) * np.exp(-5),
            (1, 2, 2): np.exp(-6) * np.exp(4) * np.exp(-6) * np.exp(5) * np.exp(-3)
        }
        test_model = _Model(state_machine, states_to_classes, x, y)
        actual_alpha = test_model._forward(parameters)

        self.assertEqual(len(actual_alpha), len(expected_alpha))
        print
        for key in sorted(expected_alpha.keys()):
            print key, np.emath.log(expected_alpha[key]), np.emath.log(actual_alpha[key])
            self.assertEqual(actual_alpha[key], expected_alpha[key])


if __name__ == '__main__':
    unittest.main()
