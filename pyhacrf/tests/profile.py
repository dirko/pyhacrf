""" A slow test for profiling """

from numpy.testing import assert_array_almost_equal
import numpy as np
from numpy import random
from pyhacrf import Hacrf
from pyhacrf.pyhacrf import _Model


def test_derivate_large():
    classes = ['a', 'b', 'c']
    y = 'b'
    x = random.randn(20, 3, 10) * 5 + 3
    state_machine, states_to_classes = Hacrf._default_state_machine(classes)
    parameters = Hacrf._initialize_parameters(state_machine, x.shape[2])
    parameters = random.randn(*parameters.shape) * 10 - 2

    test_model = _Model(state_machine, states_to_classes, x, y)
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
            expected_dll[s, d] = (y1 - y0) / delta

    actual_ll, actual_dll = test_model.forward_backward(parameters)

    print (abs(actual_dll) - abs(expected_dll)).sum()
    assert_array_almost_equal(actual_dll, expected_dll, decimal=4)

if __name__ == '__main__':
    test_derivate_large()
