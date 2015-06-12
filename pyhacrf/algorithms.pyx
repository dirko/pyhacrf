#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from numpy.math cimport logaddexp, INFINITY as inf


cpdef dict forward(np.ndarray[long, ndim=2] lattice, np.ndarray[double, ndim=3] x_dot_parameters, long S):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef unsigned int r
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef unsigned int I, J, s

    cdef unsigned int old_i0, old_j0, old_s0 
    cdef double edge_potential

    old_i0, old_j0, old_s0 = -1, -1, -1

    for r in range(lattice.shape[0]):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2] 
        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]
        
        if i0 != old_i0 or j0 != old_j0 or s0 != old_s0:
            if i0 == 0 and j0 == 0:
                alpha[(i0, j0, s0)] = x_dot_parameters[i0, j0, s0]
            else:
                alpha[(i0, j0, s0)] += x_dot_parameters[i0, j0, s0]

            old_i0, old_j0, old_s0 = i0, j0, s0

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                          + <double> alpha[(i0, j0, s0)])
        alpha[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] = edge_potential
        alpha[(i1, j1, s1)] = logaddexp(<double> alpha.get((i1, j1, s1), -inf), 
                                        edge_potential)

    I = x_dot_parameters.shape[0] - 1
    J = x_dot_parameters.shape[1] - 1

    for s in range(S):
        if I == J == 0:
            alpha[(I, J, s)] = x_dot_parameters[I, J, s]
        else:
            alpha[(I, J, s)] = <double> alpha.get((I, J, s), -inf) + x_dot_parameters[I, J, s]

    return alpha

cpdef ndarray[double, ndim=3] forward_predict(ndarray[long, ndim=2] lattice, ndarray[double, ndim=3] x_dot_parameters, long S):
    """ Helper to calculate the forward weights for prediction.  """

    cdef ndarray[double, ndim=3] alpha = np.full_like(x_dot_parameters, -inf)

    cdef unsigned int r 
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef unsigned int I, J, s

    cdef unsigned int old_i0, old_j0, old_s0 
    cdef double edge_potential

    old_i0, old_j0, old_s0 = -1, -1, -1

    for r in range(lattice.shape[0]):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2], 
        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]
        if i0 != old_i0 or j0 != old_j0 or s0 != old_s0:
            if i0 == 0 and j0 == 0:
                alpha[(i0, j0, s0)] = x_dot_parameters[i0, j0, s0]
            else:
                alpha[(i0, j0, s0)] += x_dot_parameters[i0, j0, s0]

            old_i0, old_j0, old_s0 = i0, j0, s0

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                              + alpha[(i0, j0, s0)])
        alpha[(i1, j1, s1)] = logaddexp(alpha[(i1, j1, s1)], edge_potential)

    I = x_dot_parameters.shape[0] - 1
    J = x_dot_parameters.shape[1] - 1

    for s in range(S):
        if I == J == 0:
            alpha[(I, J, s)] = x_dot_parameters[I, J, s]
        else:
            alpha[(I, J, s)] += x_dot_parameters[I, J, s]
        
    return alpha


cpdef dict backward(ndarray[long, ndim=2] lattice,
                    ndarray[double, ndim=3] x_dot_parameters,
                    long I, long J, long S):
    """ Helper to calculate the backward weights.  """
    cdef dict beta = {}

    cdef unsigned int r
    cdef unsigned int s
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index

    cdef double edge_potential

    for s in range(S):
        beta[(I-1, J-1, s)] = 0.0

    for r in range((lattice.shape[0] - 1), -1, -1):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2], 
        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]

        edge_potential = <double> beta[(i1, j1, s1)] + x_dot_parameters[i1, j1, s1]
        beta[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] = edge_potential
        beta[(i0, j0, s0)] = logaddexp(<double> beta.get((i0, j0, s0), -inf),
                                       (edge_potential 
                                        + x_dot_parameters[i1, 
                                                           j1, 
                                                           edge_parameter_index]))
    return beta


def gradient(dict alpha,
                dict beta,
                ndarray[double, ndim=2] parameters,
                dict states_to_classes,
                ndarray[double, ndim=3] x,
                object y,
                long I, long J):
    """ Helper to calculate the marginals and from that the gradient given the forward and backward weights. """
    class_Z = {}
    Z = -inf

    for state, predicted_class in states_to_classes.items():
        weight = alpha[(I - 1, J - 1, state)]
        class_Z[states_to_classes[state]] = weight
        Z = logaddexp(Z, weight)

    cdef ndarray[double, ndim=2] derivative = np.full_like(parameters, 0.0)

    for node in alpha.viewkeys() | beta.viewkeys():
        alphabeta = alpha[node] + beta[node]
        if len(node) == 3:
            i, j, s = node
            E_f = (np.exp(alphabeta - class_Z[y]) * x[i, j, :]) if states_to_classes[s] == y else 0.0
            E_Z = np.exp(alphabeta - Z) * x[i, j, :]
            derivative[s, :] += E_f - E_Z

        else:
            i0, j0, s0, i1, j1, s1, edge_parameter_index = node
            E_f = (np.exp(alphabeta - class_Z[y]) * x[i1, j1, :]) if states_to_classes[s1] == y else 0.0
            E_Z = np.exp(alphabeta - Z) * x[i1, j1, :]
            derivative[edge_parameter_index, :] += E_f - E_Z

    return (class_Z[y]) - (Z), derivative
