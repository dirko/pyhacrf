# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
import lbfgs
from collections import defaultdict, deque
from .algorithms import forward, backward
from .algorithms import forward_predict


class Hacrf(object):
    """ Hidden Alignment Conditional Random Field with L2 regularizer.

    Parameters
    ----------
    l2_regularization : float, optional (default=0.0)
        The regularization parameter.

    optimizer : function, optional (default=None)
        The optimizing function that should be used minimize the negative log posterior.
        The function should have the signature:
            min_objective, argmin_objective, ... = fmin(obj, x0, **optimizer_kwargs),
        where obj is a function that returns
        the objective function and its gradient given a parameter vector; and x0 is the initial parameter vector.

    optimizer_kwargs : dictionary, optional (default=None)
        The keyword arguments to pass to the optimizing function. Only used when `optimizer` is also specified.

    References
    ----------
    See *A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance*
    by McCallum, Bellare, and Pereira, and the report *Conditional Random Fields for Noisy text normalisation*
    by Dirko Coetsee.
    """

    def __init__(self, l2_regularization=0.0, optimizer=None, optimizer_kwargs=None):
        self.parameters = None
        self.classes = None
        self.l2_regularization = l2_regularization
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        # TODO: make it possible to add own state machine / provide alternative state machines.

        self.optimizer_result = None
        self._state_machine = None
        self._states_to_classes = None
        self._evaluation_count = None

    def fit(self, X, y, verbosity=0):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len, string2_len, n_features), where
            string1_len and string2_len are the length of the two training strings and n_features the
            number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes = list(set(y))
        n_points = len(y)
        if len(X) != n_points:
            raise Exception('Number of training points should be the same as training labels.')

        # Default state machine.
        self._state_machine, self._states_to_classes = self._default_state_machine(self.classes)

        # Initialize the parameters given the state machine, features, and target classes.
        self.parameters = self._initialize_parameters(self._state_machine, X[0].shape[2])

        # Create a new model object for each training example
        models = [_Model(self._state_machine, self._states_to_classes, x, ty) for x, ty in zip(X, y)]

        self._evaluation_count = 0

        def _objective(parameters):
            gradient = np.zeros(self.parameters.shape)
            ll = 0.0  # Log likelihood
            # TODO: Embarrassingly parallel
            for model in models:
                dll, dgradient = model.forward_backward(parameters.reshape(self.parameters.shape))
                ll += dll
                gradient += dgradient

            parameters_without_bias = np.array(parameters)  # exclude the bias parameters from being regularized
            parameters_without_bias[0] = 0
            ll -= self.l2_regularization * np.dot(parameters_without_bias.T, parameters_without_bias)
            gradient = gradient.flatten() - 2.0 * self.l2_regularization * parameters_without_bias

            if verbosity > 0:
                if self._evaluation_count == 0:
                    print('{:10} {:10} {:10}'.format('Iteration', 'Log-likelihood', '|gradient|'))
                if self._evaluation_count % verbosity == 0:
                    print('{:10} {:10.4} {:10.4}'.format(self._evaluation_count, ll, (abs(gradient).sum())))
            self._evaluation_count += 1

            return -ll, -gradient

        def _objective_copy_gradient(paramers, g):
            nll, ngradient = _objective(paramers)
            g[:] = ngradient
            return nll

        if self._optimizer:
            self.optimizer_result = self._optimizer(_objective, self.parameters.flatten(), **self._optimizer_kwargs)
            self.parameters = self.optimizer_result[0].reshape(self.parameters.shape)
        else:
            optimizer = lbfgs.LBFGS()
            final_betas = optimizer.minimize(_objective_copy_gradient,
                                             x0=self.parameters.flatten(),
                                             progress=None)
            self.optimizer_result = final_betas
            self.parameters = final_betas.reshape(self.parameters.shape)
        return self

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len, string2_len, n_features, where
            string1_len and string2_len are the length of the two training strings and n_features the
            number of features.

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        class_to_index = {class_name: index for index, class_name in enumerate(self.classes)}
        return np.array(
            [zip(*sorted(_Model(self._state_machine, self._states_to_classes, x).predict(self.parameters).items(),
                         key=lambda item: class_to_index[item[0]]))[1] for x in X])

    def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len, string2_len, n_features, where
            string1_len and string2_len are the length of the two training strings and n_features the
            number of features.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.
        """
        return [self.classes[prediction.argmax()] for prediction in self.predict_proba(X)]

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

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return {'l2_regularization': self.l2_regularization,
                'optimizer': self._optimizer,
                'optimizer_kwargs': self._optimizer_kwargs}

    def set_params(self, l2_regularization=0.0, optimizer=None, optimizer_kwargs=None):
        """Set the parameters of this estimator.

        Returns
        -------
        self
        """
        self.l2_regularization = l2_regularization
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        return self


class _Model(object):
    """ The actual model that implements the inference routines. """
    def __init__(self, state_machine, states_to_classes, x, y=None):
        self.state_machine = state_machine
        self.states_to_classes = states_to_classes
        self.x = x
        self.y = y
        self._lattice = self._build_lattice(self.x, self.state_machine)

    def forward_backward(self, parameters):
        """ Run the forward backward algorithm with the given parameters. """
        x_dot_parameters = np.dot(self.x, parameters.T)  # Pre-compute the dot product
        alpha, class_Z, Z = self._forward_probabilities(x_dot_parameters)
        beta = self._backward(x_dot_parameters)

        derivative = np.zeros(parameters.shape)
        for node in self._lattice:
            alphabeta = alpha[node] + beta[node]
            if len(node) == 3:
                i, j, s = node
                E_f = (np.exp(alphabeta - class_Z[self.y]) * self.x[i, j, :]) if self.states_to_classes[s] == self.y else 0.0
                E_Z = np.exp(alphabeta - Z) * self.x[i, j, :]
                derivative[s, :] += E_f - E_Z

            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node
                E_f = (np.exp(alphabeta - class_Z[self.y]) * self.x[i1, j1, :]) if self.states_to_classes[s1] == self.y else 0.0
                E_Z = np.exp(alphabeta - Z) * self.x[i1, j1, :]
                derivative[edge_parameter_index, :] += E_f - E_Z

        return (class_Z[self.y]) - (Z), derivative

    def predict(self, parameters):
        """ Run forward algorithm to find the predicted distribution over classes. """
        x_dot_parameters = np.dot(self.x, parameters.T)  # Pre-compute the dot product
        alpha = forward_predict(self._lattice, x_dot_parameters)
        I, J, _ = self.x.shape

        class_Z = {}
        Z = -np.inf

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[I - 1, J - 1, state]
            class_Z[self.states_to_classes[state]] = weight
            Z = np.logaddexp(Z, weight)

        return {label: np.exp(class_z - Z) for label, class_z in class_Z.iteritems()}

    def _forward_probabilities(self, x_dot_parameters):
        """ Helper to calculate the forward probabilities and the predicted probability
        distribution over classes given some parameters. """
        alpha = self._forward(x_dot_parameters)
        I, J, _ = self.x.shape

        class_Z = {}
        Z = -np.inf

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[(I - 1, J - 1, state)]
            class_Z[self.states_to_classes[state]] = weight
            Z = np.logaddexp(Z, weight)
        return alpha, class_Z, Z

    def _forward(self, x_dot_parameters):
        """ Helper to calculate the forward weights.  """
        return forward(self._lattice, x_dot_parameters)

    def _backward(self, x_dot_parameters):
        """ Helper to calculate the backward weights.  """
        I, J, _ = self.x.shape
        return backward(self._lattice, x_dot_parameters, I, J)

    @staticmethod
    def _build_lattice(x, state_machine):
        """ Helper to construct the list of nodes and edges. """
        I, J, _ = x.shape
        lattice = []
        start_states, transitions = state_machine
        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(transitions) :
            transitions_d[s0].append((s1, delta, transition_index))
        # Add start states
        unvisited_nodes = deque([(0, 0, s) for s in start_states])
        visited_nodes = set()
        n_states, _ = _n_states(state_machine)

        while unvisited_nodes:
            node = unvisited_nodes.popleft()
            lattice.append(node)
            i, j, s0 = node
            for s1, delta, transition_index in transitions_d[s0] :
                try :
                    di, dj = delta
                except TypeError :
                    di, dj = delta(i, j, x)

                if i + di < I and j + dj < J:
                    edge = (i, j, s0, i + di, j + dj, s1, transition_index + n_states)
                    lattice.append(edge)
                    dest_node = (i + di, j + dj, s1)
                    if dest_node not in visited_nodes :
                        unvisited_nodes.append(dest_node)
                        visited_nodes.add(dest_node)

        lattice.sort()

        # Step backwards through lattice and add visitable nodes to the set of nodes to keep. The rest are discarded.
        final_lattice = []
        visited_nodes = set((I-1, J-1, s) for s in xrange(n_states))

        for node in lattice[::-1]:
            if node in visited_nodes:
                final_lattice.append(node)
            elif len(node) > 3:
                source_node, dest_node = node[0:3], node[3:6]
                if dest_node in visited_nodes:
                    visited_nodes.add(source_node)
                    final_lattice.append(node)

        return list(reversed(final_lattice))


def _n_states(state_machine):
    """ Helper to calculate the number of states.  """
    start_states, edges = state_machine
    max_state = max(max(s for s, _, _ in edges), max(s for _, s, _ in edges)) + 1
    n_transitions = len(state_machine[1])
    return max_state, n_transitions

