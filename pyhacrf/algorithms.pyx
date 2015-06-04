#!python
#cython: boundscheck=True

from collections import defaultdict
import numpy as np
cimport numpy as np
from numpy.math cimport logaddexp


cpdef dict forward(np.ndarray[long, ndim=2] lattice, np.ndarray[double, ndim=3] x_dot_parameters):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef int i, j, s, i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef int I, J, S
    cdef int old_i, old_j, old_s 
    cdef float edge_potential
    cdef tuple node

    old_i, old_j, old_s = -1, -1, -1

    for i in range(lattice.shape[0]) :
        i0, j0, s0 = lattice[i, 0], lattice[i, 1], lattice[i, 2] 
        i1, j1, s1 = lattice[i, 3], lattice[i, 4], lattice[i, 5]
        edge_parameter_index = lattice[i, 6]
        if i0 != old_i or j0 != old_j or s0 != old_s :
            if i0 == 0 and j0 == 0:
                alpha[(i0, j0, s0)] = x_dot_parameters[i0, j0, s0]
            else:
                alpha[(i0, j0, s0)] += x_dot_parameters[i0, j0, s0]

            old_i, old_j, old_s = i0, j0, s0

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                              + alpha[(i0, j0, s0)])
        alpha[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] = edge_potential
        alpha[(i1, j1, s1)] = logaddexp(alpha.get((i1, j1, s1), -np.inf), edge_potential)

    I = x_dot_parameters.shape[0] - 1
    J = x_dot_parameters.shape[1] - 1

    for s in range(2) :
        alpha[(I, J, s)] += x_dot_parameters[I, J, s]

    return alpha

cpdef np.ndarray[double, ndim=3] forward_predict(np.ndarray[long, ndim=2] lattice, np.ndarray[double, ndim=3] x_dot_parameters):
    """ Helper to calculate the forward weights.  """
    cdef np.ndarray[double, ndim=3] alpha = np.empty_like(x_dot_parameters)
    alpha.fill(-np.inf)

    cdef int i, j, s, i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef int old_i, old_j, old_s 
    cdef float edge_potential
    cdef tuple node

    old_i, old_j, old_s = -1, -1, -1

    for i in range(lattice.shape[0]) :
        i0, j0, s0 = lattice[i, 0], lattice[i, 1], lattice[i, 2], 
        i1, j1, s1 = lattice[i, 3], lattice[i, 4], lattice[i, 5]
        edge_parameter_index = lattice[i, 6]
        if i0 != old_i or j0 != old_j or s0 != old_s :
            if i0 == 0 and j0 == 0:
                alpha[(i0, j0, s0)] = x_dot_parameters[i0, j0, s0]
            else:
                alpha[(i0, j0, s0)] += x_dot_parameters[i0, j0, s0]

            old_i, old_j, old_s = i0, j0, s0

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                              + alpha[(i0, j0, s0)])
        alpha[(i1, j1, s1)] = logaddexp(alpha[(i1, j1, s1)], edge_potential)
    return alpha





def backward(lattice, x_dot_parameters, I, J):
    """ Helper to calculate the backward weights.  """
    beta = defaultdict(lambda: -np.inf)
    for node, neighbors in reversed(lattice):
        i0, j0, s0 = node
        if i0 == I - 1 and j0 == J - 1:
            beta[node] = 0.0
        for (i1, j1, s1), edge_parameter_index in neighbors :
            edge = node + (i1, j1, s1, edge_parameter_index) 
            edge_potential = beta[(i1, j1, s1)] + (x_dot_parameters[i1, j1, s1])
            beta[edge] = edge_potential
            beta[(i0, j0, s0)] = np.logaddexp(beta[(i0, j0, s0)],
                                              edge_potential + x_dot_parameters[i1, j1, edge_parameter_index])
    return beta
