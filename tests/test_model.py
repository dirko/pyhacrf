""" Tests for the model. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from pyhacrf import Hacrf
from pyhacrf import _Model


class TestHacrf(unittest.TestCase):
    def test_initialize_parameters(self):
        start_states = [0]
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0))]
        state_machine = (start_states, transitions)
        X = [np.zeros((6, 7, 3))]

        actual_parameters = Hacrf._initialize_parameters(state_machine, X[0].shape[2])
        expected_parameter_shape = (5, 3)
        self.assertEqual(actual_parameters.shape, expected_parameter_shape)

    def test_default_state_machine(self):
        classes = ['a', 'b']
        expected_state_machine = ([0, 1],
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
        start_states = [0, 1]
        n_states = 4  # Because 3 is the max
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0)),
                       (0, 3, lambda i, j, k: (0, 2))]
        state_machine = (start_states, transitions)
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
                            (0, 0, 0, 1, 0, 0, 2 + n_states),
                            (0, 0, 0, 1, 1, 0, 0 + n_states),
                            (1, 0, 0),
                            (1, 0, 0, 1, 2, 3, 3 + n_states),
                            (1, 1, 0),
                            (1, 1, 0, 1, 2, 1, 1 + n_states),
                            (1, 2, 1),
                            (1, 2, 3)]
        self.assertEqual(actual_lattice, expected_lattice)

    def test_forward_single(self):
        start_states = [0, 1]
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0)),
                       (0, 2, lambda i, j, k: (0, 2))]
        state_machine = (start_states, transitions)
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
            (0, 0, 0, 1, 0, 0, 5): np.exp(-6) * np.exp(4),
            (0, 0, 0, 1, 1, 0, 3): np.exp(-6) * np.exp(-1),
            (1, 0, 0): np.exp(-6) * np.exp(4) * np.exp(-6),
            (1, 0, 0, 1, 2, 2, 6): np.exp(-6) * np.exp(4) * np.exp(-6) * np.exp(5),
            (1, 1, 0): np.exp(-6) * np.exp(-1) * np.exp(-7),
            (1, 1, 0, 1, 2, 1, 4): np.exp(-6) * np.exp(-1) * np.exp(-7) * np.exp(1),
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

    def test_forward_connected(self):
        classes = ['a', 'b']
        parameters = np.array(range(-8, 8)).reshape((8, 2))
        # parameters =
        #0([[-8, -7],
        #1  [-6, -5],
        #2  [-4, -3],
        #3  [-2, -1],
        #4  [ 0,  1],
        #5  [ 2,  3],
        #6  [ 4,  5],
        #7  [ 6,  7]])
        x = np.array([[[0, 1],
                       [2, 1]],
                      [[0, 1],
                       [1, 0]]])
        y = 'a'
        expected_alpha = {
            (0, 0, 0): np.exp(-7),
            (0, 0, 0, 0, 1, 0, 4): np.exp(-7) * np.exp(1),
            (0, 0, 0, 1, 0, 0, 6): np.exp(-7) * np.exp(5),
            (0, 0, 0, 1, 1, 0, 2): np.exp(-7) * np.exp(-4),
            (0, 0, 1): np.exp(-5),
            (0, 0, 1, 0, 1, 1, 5): np.exp(-5) * np.exp(7),
            (0, 0, 1, 1, 0, 1, 7): np.exp(-5) * np.exp(7),
            (0, 0, 1, 1, 1, 1, 3): np.exp(-5) * np.exp(-2),
            (0, 1, 0): np.exp(-7) * np.exp(1) * np.exp(-23),
            (0, 1, 0, 1, 1, 0, 6): np.exp(-7) * np.exp(1) * np.exp(-23) * np.exp(4),
            (0, 1, 1): np.exp(-5) * np.exp(7) * np.exp(-17),
            (0, 1, 1, 1, 1, 1, 7): np.exp(-5) * np.exp(7) * np.exp(-17) * np.exp(6),
            (1, 0, 0): np.exp(-7) * np.exp(5) * np.exp(-7),
            (1, 0, 0, 1, 1, 0, 4): np.exp(-7) * np.exp(5) * np.exp(-7) * np.exp(0),
            (1, 0, 1): np.exp(-5) * np.exp(7) * np.exp(-5),
            (1, 0, 1, 1, 1, 1, 5): np.exp(-5) * np.exp(7) * np.exp(-5) * np.exp(2),
            (1, 1, 0): (np.exp(-11) + np.exp(-25) + np.exp(-9)) * np.exp(-8),
            (1, 1, 1): (np.exp(-1) + np.exp(-9) + np.exp(-7)) * np.exp(-6)
        }
        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        print
        test_model = _Model(state_machine, states_to_classes, x, y)
        for s in test_model._lattice:
            print s
        actual_alpha = test_model._forward(parameters)

        self.assertEqual(len(actual_alpha), len(expected_alpha))
        for key in sorted(expected_alpha.keys()):
            print key, expected_alpha[key], actual_alpha[key]
            self.assertAlmostEqual(actual_alpha[key], expected_alpha[key])

    def test_backward_connected(self):
        classes = ['a', 'b']
        parameters = np.array(range(-3, 3)).reshape((3, 2))
        # parameters =
        #0 ([[-3, -2],
        #1   [-1,  0],
        #2   [ 1,  2]])
        x = np.array([[[0, 1],
                       [2, 1]],
                      [[0, 1],
                       [1, 0]]])
        y = 'a'
        expected_beta = {
            (0, 0, 0): (np.exp(-4) + np.exp(-12)),  # * np.exp(-2),
            (0, 0, 0, 0, 1, 0, 1): np.exp(-3) * np.exp(1) * np.exp(-8),  # * np.exp(-2),
            (0, 0, 0, 1, 0, 0, 2): np.exp(-3) * np.exp(-1) * np.exp(-2),  # * np.exp(2),
            (0, 1, 0): np.exp(-3) * np.exp(1),  # * np.exp(-8),
            (0, 1, 0, 1, 1, 0, 2): np.exp(-3),  # * np.exp(1),
            (1, 0, 0): np.exp(-3) * np.exp(-1),  # * np.exp(-2),
            (1, 0, 0, 1, 1, 0, 1): np.exp(-3),  # * np.exp(-1),
            (1, 1, 0): 1.0  # np.exp(-3)
        }
        state_machine = ([0], [(0, 0, (0, 1)), (0, 0, (1, 0))])
        states_to_classes = {0: 'a'}

        print state_machine, states_to_classes
        print
        test_model = _Model(state_machine, states_to_classes, x, y)
        for s in test_model._lattice:
            print s
        actual_beta = test_model._backward(parameters)

        print
        self.assertEqual(len(actual_beta), len(expected_beta))
        for key in sorted(expected_beta.keys(), reverse=True):
            print key, expected_beta[key], actual_beta[key]
            self.assertAlmostEqual(actual_beta[key], expected_beta[key])

    def test_forward_backward_same_partition_value(self):
        classes = ['a', 'b']
        parameters = np.array(range(-8, 8)).reshape((8, 2))
        x = np.array([[[0, 1],
                       [2, 1]],
                      [[0, 1],
                       [1, 0]]])
        y = 'a'
        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        test_model = _Model(state_machine, states_to_classes, x, y)
        actual_alpha = test_model._forward(parameters)
        actual_beta = test_model._backward(parameters)

        print actual_alpha[(1, 1, 0)], actual_beta[(0, 0, 0)]
        print actual_alpha[(1, 1, 1)], actual_beta[(0, 0, 1)]
        self.assertAlmostEqual(actual_alpha[(1, 1, 0)], actual_beta[(0, 0, 0)] * np.exp(np.dot(x[0, 0, :],
                                                                                               parameters[0, :])))
        self.assertAlmostEqual(actual_alpha[(1, 1, 1)], actual_beta[(0, 0, 1)] * np.exp(np.dot(x[0, 0, :],
                                                                                               parameters[1, :])))

    def test_derivate_chain(self):
        classes = ['a', 'b']
        parameters = np.array(range(-8, 8)).reshape((8, 2))
        # parameters =
        #0([[-8, -7],
        #1  [-6, -5],
        #2  [-4, -3],
        #3  [-2, -1],
        #4  [ 0,  1],
        #5  [ 2,  3],
        #6  [ 4,  5],
        #7  [ 6,  7]])
        x = np.array([[[0, 1],
                       [1, 2]]])
        y = 'a'
        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        test_model = _Model(state_machine, states_to_classes, x, y)
        print test_model._lattice
        print states_to_classes
        #
        # 0   01 --- 01
        #     0      1
        # states_to_classes = {0: 'a', 1: 'b'}
        # (0, 0, 0) :               exp(-7)
        # (0, 0, 0, 0, 1, 0, 4) :   exp(-7) * exp(2)
        # (0, 0, 1) :               exp(-5)
        # (0, 0, 1, 0, 1, 1, 5) :   exp(-5) * exp(8)
        # (0, 1, 0) :               exp(-7) * exp(2) * exp(-8 - 14)     = exp(-27)
        # (0, 1, 1) :               exp(-5) * exp(8) * exp(-6 - 10)     = exp(-13)
        # p(y|G,X) = f0(g00,g01,x00,x01,y) f1(g40,g41,x10,x11,y) f2(g00,g01,x00,x01,y)  +
        #            f0(g10,g11,x00,x01,y) f1(g50,g51,x10,x11,y) f2(g10,g11,x00,x01,y)
        # = exp(-27) / (exp(-27) + exp(-13))
        expected_ll = np.emath.log(np.exp(-27) / (np.exp(-27) + np.exp(-13)))
        expected_dll = np.zeros(parameters.shape)

        # Finite element approximation
        delta = 10.0**-14
        S, D = expected_dll.shape
        for s in xrange(S):
            for d in xrange(D):
                dg = np.zeros(parameters.shape)
                dg[s, d] = delta
                y0, _ = test_model.forward_backward(parameters)
                y1, _ = test_model.forward_backward(parameters + dg)
                print s, d, y0, y1
                expected_dll[s, d] = (np.exp(y1) - np.exp(y0)) / delta

        actual_ll, actual_dll = test_model.forward_backward(parameters)

        print expected_ll, actual_ll
        print expected_dll
        print actual_dll
        kaas
        self.assertAlmostEqual(actual_alpha[(1, 1, 0)], actual_beta[(0, 0, 0)])


if __name__ == '__main__':
    unittest.main()
