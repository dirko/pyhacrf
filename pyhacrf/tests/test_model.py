""" Tests for the model. """

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from numpy import random
from pyhacrf import Hacrf
from pyhacrf.pyhacrf import _Model
from pyhacrf import StringPairFeatureExtractor


class TestHacrf(unittest.TestCase):
    def test_initialize_parameters(self):
        start_states = [0]
        transitions = [(0, 0, (1, 1)),
                       (0, 1, (0, 1)),
                       (0, 0, (1, 0))]
        state_machine = (start_states, transitions)
        n_features = 3

        actual_parameters = Hacrf._initialize_parameters(state_machine, n_features)
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

    def test_fit_predict(self):
        incorrect = ['helloooo', 'freshh', 'ffb', 'h0me', 'wonderin', 'relaionship', 'hubby', 'krazii', 'mite', 'tropic']
        correct = ['hello', 'fresh', 'facebook', 'home', 'wondering', 'relationship', 'husband', 'crazy', 'might', 'topic']
        training = zip(incorrect, correct)

        fe = StringPairFeatureExtractor(match=True, numeric=True)
        xf = fe.fit_transform(training)

        model = Hacrf()
        model.fit(xf, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        expected_parameters = np.array([[-10.76945326, 144.03414923, 0.],
                                        [31.84369748, -106.41885651, 0.],
                                        [-52.08919467, 4.56943665, 0.],
                                        [31.01495044, -13.0593297, 0.],
                                        [49.77302218, -6.42566204, 0.],
                                        [-28.69877796, 24.47127009, 0.],
                                        [-85.34524911, 21.87370646, 0.],
                                        [106.41949333, 6.18587125, 0.]])
        print model.parameters
        assert_array_almost_equal(model.parameters, expected_parameters)

        expected_probas = np.array([[1.00000000e+000, 3.51235685e-039],
                                    [1.00000000e+000, 4.79716208e-039],
                                    [1.00000000e+000, 2.82744641e-139],
                                    [1.00000000e+000, 6.49580729e-012],
                                    [9.99933798e-001, 6.62022561e-005],
                                    [8.78935957e-005, 9.99912106e-001],
                                    [4.84538335e-009, 9.99999995e-001],
                                    [1.25170233e-250, 1.00000000e+000],
                                    [2.46673086e-010, 1.00000000e+000],
                                    [1.03521293e-033, 1.00000000e+000]])
        actual_predict_probas = model.predict_proba(xf)
        print actual_predict_probas
        assert_array_almost_equal(actual_predict_probas, expected_probas)

        expected_predictions = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        actual_predictions = model.predict(xf)
        assert_array_almost_equal(actual_predictions, expected_predictions)

    def test_fit_predict_regularized(self):
        incorrect = ['helloooo', 'freshh', 'ffb', 'h0me', 'wonderin', 'relaionship', 'hubby', 'krazii', 'mite', 'tropic']
        correct = ['hello', 'fresh', 'facebook', 'home', 'wondering', 'relationship', 'husband', 'crazy', 'might', 'topic']
        training = zip(incorrect, correct)

        fe = StringPairFeatureExtractor(match=True, numeric=True)
        xf = fe.fit_transform(training)

        model = Hacrf(l2_regularization=10.0)
        model.fit(xf, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        print model.parameters

        expected_parameters = np.array([[-0.0569188, 0.07413339, 0.],
                                        [0.00187709, -0.06377866, 0.],
                                        [-0.01908823, 0.00586189, 0.],
                                        [0.01721114, -0.00636556, 0.],
                                        [0.01578279, 0.0078614, 0.],
                                        [-0.0139057, -0.00862948, 0.],
                                        [-0.00623241, 0.02937325, 0.],
                                        [0.00810951, -0.01774676, 0.]])
        assert_array_almost_equal(model.parameters, expected_parameters)

        expected_probas = np.array([[0.5227226, 0.4772774],
                                    [0.52568993, 0.47431007],
                                    [0.4547091, 0.5452909],
                                    [0.51179222, 0.48820778],
                                    [0.46347576, 0.53652424],
                                    [0.45710098, 0.54289902],
                                    [0.46159657, 0.53840343],
                                    [0.42997978, 0.57002022],
                                    [0.47419724, 0.52580276],
                                    [0.50797852, 0.49202148]])
        actual_predict_probas = model.predict_proba(xf)
        print actual_predict_probas
        assert_array_almost_equal(actual_predict_probas, expected_probas)

        expected_predictions = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        actual_predictions = model.predict(xf)
        assert_array_almost_equal(actual_predictions, expected_predictions)


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
        expected_alpha = {k: np.emath.log(v) for k, v in expected_alpha.items()}
        test_model = _Model(state_machine, states_to_classes, x, y)
        x_dot_parameters = np.dot(x, parameters.T)  # Pre-compute the dot product
        actual_alpha = test_model._forward(x_dot_parameters)

        self.assertEqual(len(actual_alpha), len(expected_alpha))
        print
        for key in sorted(expected_alpha.keys()):
            print key, (expected_alpha[key]), (actual_alpha[key])
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
        expected_alpha = {k: np.emath.log(v) for k, v in expected_alpha.items()}

        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        print
        test_model = _Model(state_machine, states_to_classes, x, y)
        for s in test_model._lattice:
            print s
        x_dot_parameters = np.dot(x, parameters.T)  # Pre-compute the dot product
        actual_alpha = test_model._forward(x_dot_parameters)

        self.assertEqual(len(actual_alpha), len(expected_alpha))
        for key in sorted(expected_alpha.keys()):
            print key, expected_alpha[key], actual_alpha[key]
            self.assertAlmostEqual(actual_alpha[key], expected_alpha[key])

    def test_backward_connected(self):
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
        expected_beta = {k: np.emath.log(v) for k, v in expected_beta.items()}

        state_machine = ([0], [(0, 0, (0, 1)), (0, 0, (1, 0))])
        states_to_classes = {0: 'a'}

        print state_machine, states_to_classes
        print
        test_model = _Model(state_machine, states_to_classes, x, y)
        for s in test_model._lattice:
            print s
        x_dot_parameters = np.dot(x, parameters.T)  # Pre-compute the dot product
        actual_beta = test_model._backward(x_dot_parameters)

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
        x_dot_parameters = np.dot(x, parameters.T)  # Pre-compute the dot product
        actual_alpha = test_model._forward(x_dot_parameters)
        actual_beta = test_model._backward(x_dot_parameters)

        print actual_alpha[(1, 1, 0)], actual_beta[(0, 0, 0)]
        print actual_alpha[(1, 1, 1)], actual_beta[(0, 0, 1)]
        self.assertAlmostEqual(actual_alpha[(1, 1, 0)], actual_beta[(0, 0, 0)] + (np.dot(x[0, 0, :], parameters[0, :])))
        self.assertAlmostEqual(actual_alpha[(1, 1, 1)], actual_beta[(0, 0, 1)] + (np.dot(x[0, 0, :], parameters[1, :])))

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

        # Finite difference gradient approximation
        delta = 10.0**-7
        S, D = expected_dll.shape
        for s in xrange(S):
            for d in xrange(D):
                dg = np.zeros(parameters.shape)
                dg[s, d] = delta
                y0, _ = test_model.forward_backward(parameters)
                y1, _ = test_model.forward_backward(parameters + dg)
                print s, d, y0, y1
                expected_dll[s, d] = (y1 - y0) / delta

        actual_ll, actual_dll = test_model.forward_backward(parameters)

        print expected_ll, actual_ll
        print expected_dll
        print actual_dll
        self.assertAlmostEqual(actual_ll, expected_ll)
        assert_array_almost_equal(actual_dll, expected_dll, decimal=5)

    def test_derivate_medium(self):
        classes = ['a', 'b']
        parameters = np.array(range(-8, 8)).reshape((8, 2))
        x = np.array([[[0, 1],
                       [2, 1]],
                      [[0, 1],
                       [1, 0]]])
        y = 'a'
        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        test_model = _Model(state_machine, states_to_classes, x, y)
        print test_model._lattice
        print states_to_classes

        expected_dll = np.zeros(parameters.shape)

        # Finite difference gradient approximation
        delta = 10.0**-7
        S, D = expected_dll.shape
        for s in xrange(S):
            for d in xrange(D):
                dg = np.zeros(parameters.shape)
                dg[s, d] = delta
                y0, _ = test_model.forward_backward(parameters)
                y1, _ = test_model.forward_backward(parameters + dg)
                print s, d, y0, y1
                expected_dll[s, d] = (y1 - y0) / delta

        actual_ll, actual_dll = test_model.forward_backward(parameters)

        print expected_dll
        print actual_dll
        assert_array_almost_equal(actual_dll, expected_dll, decimal=5)

    def test_derivate_large(self):
        classes = ['a', 'b', 'c']
        y = 'b'
        x = random.randn(8, 3, 10) * 5 + 3
        state_machine, states_to_classes = Hacrf._default_state_machine(classes)
        parameters = Hacrf._initialize_parameters(state_machine, x.shape[2])
        parameters = random.randn(*parameters.shape) * 10 - 2

        test_model = _Model(state_machine, states_to_classes, x, y)
        print test_model._lattice
        print states_to_classes

        expected_dll = np.zeros(parameters.shape)

        # Finite difference gradient approximation
        delta = 10.0**-7
        S, D = expected_dll.shape
        for s in xrange(S):
            for d in xrange(D):
                dg = np.zeros(parameters.shape)
                dg[s, d] = delta
                y0, _ = test_model.forward_backward(parameters)
                y1, _ = test_model.forward_backward(parameters + dg)
                print s, d, y0, y1
                expected_dll[s, d] = (y1 - y0) / delta

        actual_ll, actual_dll = test_model.forward_backward(parameters)

        print expected_dll
        print actual_dll
        self.assertEqual((np.isnan(actual_dll)).any(), False)
        assert_array_almost_equal(actual_dll, expected_dll, decimal=4)

if __name__ == '__main__':
    unittest.main()
