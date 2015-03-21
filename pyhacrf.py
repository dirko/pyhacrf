# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from collections import defaultdict


class Hacrf(object):
    """ Hidden Alignment Conditional Random Field with L2 regularizer.

    See *A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance* by McCallum, Bellare, and Pereira,
        and the report *Conditional Random Fields for Noisy text normalisation* by Dirko Coetsee.
    """

    def __init__(self):
        self._optimizer_result = None
        self.parameters = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len, string2_len, n_features, where
            string1_len and string2_len are the length of the two training strings and n_features the
            number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        classes = list(set(y))
        n_points = len(y)
        if len(X) != n_points:
            raise Exception('Number of training points should be the same as training labels.')

        # Default state machine. Tuple (list_of_states, list_of_transitions)
        state_machine, states_to_classes = self._default_state_machine(classes)

        # Initialize the parameters given the state machine, features, and target classes.
        self.parameters = self._initialize_parameters(state_machine, X[0].shape[2])

        # Create a new model object for each training example
        models = [_Model(state_machine, states_to_classes, x, ty) for x, ty in zip(X, y)]

        def _objective(parameters):
            derivative = np.zeros(self.parameters.shape)
            ll = 0.0  # Log likelihood
            # TODO: Embarrassingly parallel
            for model in models:
                dll, dderivative = model.forward_backward(parameters)
                ll += dll
                derivative += dderivative
            return -ll, -derivative

        self._optimizer_result = fmin_l_bfgs_b(_objective, self.parameters)
        return self

    def predict_proba(self):
        pass

    def predict(self):
        pass

    @staticmethod
    def _initialize_parameters(state_machine, n_features):
        """ Helper to create initial parameter vector with the correct shape. """
        n_states, n_transitions = _n_states(state_machine)
        return np.zeros((n_states + n_transitions, n_features))

    @staticmethod
    def _default_state_machine(classes):
        """ Helper to construct a state machine that includes insertions, matches, and deletions for each class. """
        n_classes = len(classes)
        return (([i for i in xrange(n_classes)],  # A state for each class.
                 [(i, i, (1, 1)) for i in xrange(n_classes)] +  # Match
                 [(i, i, (0, 1)) for i in xrange(n_classes)] +  # Insertion
                 [(i, i, (1, 0)) for i in xrange(n_classes)]),  # Deletion
                dict((i, c) for i, c in enumerate(classes)))


class _Model(object):
    """ The actual model that implements the inference routines. """
    def __init__(self, state_machine, states_to_classes, x, y):
        self.state_machine = state_machine
        self.states_to_classes = states_to_classes
        self.x = x
        self.y = y
        self._lattice = self._build_lattice(self.x, self.state_machine)

    def forward_backward(self, parameters):
        """ Run the forward backward algorithm with the given parameters. """
        alpha, class_Z, Z = self._forward_probabilities(parameters)
        beta = self._backward(parameters)

        derivative = np.zeros(parameters.shape)
        for node in self._lattice:
            if len(node) == 3:
                i, j, s = node
                in_class = 1.0 if self.states_to_classes[s] == self.y else 0.0
                E_f = alpha[node] * beta[node] / class_Z[self.y] * self.x[i, j, :] * in_class
                E_Z = (alpha[node] * beta[node] * self.x[i, j, :]) / Z
                derivative[s, :] += E_f - E_Z

            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node
                in_class = 1.0 if self.states_to_classes[s1] == self.y else 0.0
                E_f = (alpha[node] * beta[node] / class_Z[self.y]) * self.x[i1, j1, :] * in_class
                E_Z = (alpha[node] * beta[node] / Z) * self.x[i1, j1, :]
                derivative[edge_parameter_index, :] += E_f - E_Z

        return np.emath.log(class_Z[self.y]) - np.emath.log(Z), derivative

    def _forward_probabilities(self, parameters):
        """ Helper to calculate the predicted probability distribution over classes given some parameters. """
        alpha = self._forward(parameters)
        I, J, _ = self.x.shape

        class_Z = {}
        Z = 0.0

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[(I - 1, J - 1, state)]
            class_Z[self.states_to_classes[state]] = weight
            Z += weight
        return alpha, class_Z, Z

    def _forward(self, parameters):
        """ Helper to calculate the forward weights.  """
        alpha = defaultdict(float)
        for node in self._lattice:
            if len(node) == 3:
                i, j, s = node
                if i == 0 and j == 0:
                    alpha[node] = np.exp(np.dot(self.x[i, j, :], parameters[s, :]))
                else:
                    alpha[node] *= np.exp(np.dot(self.x[i, j, :], parameters[s, :]))
            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
                # Use the features at the destination of the edge.
                edge_potential = (np.exp(np.dot(self.x[i1, j1, :], parameters[edge_parameter_index, :]))
                                  * alpha[(i0, j0, s0)])
                alpha[node] = edge_potential
                alpha[(i1, j1, s1)] += edge_potential
        return alpha

    def _backward(self, parameters):
        """ Helper to calculate the backward weights.  """
        beta = defaultdict(float)
        I, J, _ = self.x.shape
        for node in self._lattice[::-1]:
            if len(node) == 3:
                i, j, s = node
                if i == I - 1 and j == J - 1:
                    beta[node] = 1.0  # np.exp(np.dot(self.x[i, j, :], parameters[s, :]))
                else:
                    beta[node] *= 1.0  # np.exp(np.dot(self.x[i, j, :], parameters[s, :]))
            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
                # Use the features at the destination of the edge.
                edge_potential = beta[(i1, j1, s1)] * np.exp(np.dot(self.x[i1, j1, :], parameters[s1, :]))
                beta[node] = edge_potential
                beta[(i0, j0, s0)] += edge_potential * (np.exp(np.dot(self.x[i1, j1, :],
                                                                      parameters[edge_parameter_index, :])))
        return beta



    @staticmethod
    def _build_lattice(x, state_machine):
        """ Helper to construct the list of nodes and edges. """
        I, J, _ = x.shape
        lattice = []
        start_states, transitions = state_machine
        # Add start states
        unvisited_nodes = [(0, 0, s) for s in start_states]
        visited_nodes = set()
        n_states, _ = _n_states(state_machine)

        while unvisited_nodes:
            i, j, s = unvisited_nodes.pop(0)
            if (i, j, s) not in visited_nodes:
                lattice.append((i, j, s))
                visited_nodes.add((i, j, s))
            for transition_index, (s0, s1, delta) in enumerate(transitions):
                if s == s0:
                    if callable(delta):
                        di, dj = delta(i, j, x)
                    else:
                        di, dj = delta
                    if i + di < I and j + dj < J:
                        edge = (i, j, s0, i + di, j + dj, s1, transition_index + n_states)
                        if edge not in visited_nodes:
                            lattice.append(edge)
                            unvisited_nodes.append((i + di, j + dj, s1))
                            visited_nodes.add(edge)
        lattice.sort()

        # Step backwards through lattice and add visitable nodes to the set of nodes to keep. The rest are discarded.
        final_lattice = []
        visited_nodes = set()
        for node in lattice[::-1]:
            if len(node) <= 3:
                i, j, s = node
                if i == I - 1 and j == J - 1:
                    visited_nodes.add(node)
            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node
                if (i1, j1, s1) in visited_nodes:
                    visited_nodes.add(node)
                    visited_nodes.add((i0, j0, s0))
            if node in visited_nodes:
                final_lattice.insert(0, node)

        return final_lattice


def _n_states(state_machine):
    """ Helper to calculate the number of states.  """
    start_states, edges = state_machine
    max_state = max(max(s for s, _, _ in edges), max(s for _, s, _ in edges)) + 1
    n_transitions = len(state_machine[1])
    return max_state, n_transitions
