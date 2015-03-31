# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from collections import defaultdict


class Hacrf(object):
    """ Hidden Alignment Conditional Random Field with L2 regularizer.

    Parameters
    ----------
    l2_regularization : float, optional (default=0.0)
        The regularization parameter.

    References
    ----------
    See *A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance*
    by McCallum, Bellare, and Pereira, and the report *Conditional Random Fields for Noisy text normalisation*
    by Dirko Coetsee.
    """

    def __init__(self, l2_regularization=0.0):
        self.parameters = None
        self.classes = None
        self.l2_regularization = l2_regularization
        # TODO: make it possible to add own state machine / provide alternative state machines.

        self._optimizer_result = None
        self._state_machine = None
        self._states_to_classes = None
        self._evaluation_count = None

    def fit(self, X, y, verbosity=0):
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

            ll -= self.l2_regularization * np.dot(parameters.T, parameters)
            gradient = gradient.flatten() - 2.0 * self.l2_regularization * parameters

            if verbosity > 0:
                if self._evaluation_count % verbosity == 0:
                    print('{:10} {:10.4} {:10.4}'.format(self._evaluation_count, ll, (abs(gradient).sum())))
            self._evaluation_count += 1

            return -ll, -gradient

        self._optimizer_result = fmin_l_bfgs_b(_objective, self.parameters)
        self.parameters = self._optimizer_result[0].reshape(self.parameters.shape)
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


class StringPairFeatureExtractor(object):
    """ Extract features from sequence pairs.

    A grid is constructed for each sequence pair, for example for ("kaas", "cheese"):

     s * . . . @ .
     a * . . . . .
     a * . . . . .
     k * * * * * *
       c h e e s e

    For each element in the grid, a feature vector is constructed. The elements in the feature
    vector are determined by which features are active at that position in the grid. So for the
    example above, the 'match' feature will be 0 in every vector in every position except the
    position indicated with '@', where it will be 1. The 'start' feature will be 1 in all the
    positions with '*' and 0 everywhere else.


    Parameters
    ----------
    start: boolean: optional
        Binary feature that activates at the start of either sequence.

    end: boolean: optional
        Binary feature that activates at the end of either sequence.

    match: boolean: optional
        Binary feature that activates when elements at a position are equal.

    numeric: boolean, optional
        Binary feature that activates when all elements at a position are numerical.

    transition: boolean, optional
        Adds binary features for pairs of (lower case) input characters.
    """

    # Constants
    CHARACTERS = 'abcdefghijklmnopqrstuvwxyz0123456789,./;\'\-=<>?:"|_+!@#$%^&*() '

    def __init__(self, start=False, end=False, match=False, numeric=False, transition=False):
        # TODO: For longer strings, tokenize and use Levenshtein distance up until a lattice position.
        #       Other (possibly) useful features might be whether characters are consonant or vowel,
        #       punctuation, case.
        binary_features_active = [start, end, match, numeric]
        binary_features = [lambda i, j, s1, s2: 1.0 if i == 0 or j == 0 else 0.0,
                           lambda i, j, s1, s2: 1.0 if i == len(s1) - 1 or j == len(s2) - 1 else 0.0,
                           lambda i, j, s1, s2: 1.0 if s1[i] == s2[j] else 0.0,
                           lambda i, j, s1, s2: 1.0 if s1[i].isdigit() and s2[j].isdigit() else 0.0]
        self._binary_features = [feature for feature, active in zip(binary_features, binary_features_active) if active]
        self._sparse_features = []
        if transition:
            characters_to_index = {character: index for index, character in enumerate(self.CHARACTERS)}
            self._sparse_features.append(((lambda i, j, s1, s2, chars_to_index=characters_to_index:
                                           chars_to_index[s2[j].lower()] +
                                           chars_to_index[s1[i].lower()] * len(chars_to_index)),
                                          len(characters_to_index) ** 2))

    def fit_transform(self, raw_X, y=None):
        """Transform sequence pairs to feature arrays that can be used as input to `Hacrf` models.

        Parameters
        ----------
        raw_X : List of (sequence1_n, sequence2_n) pairs, one for each training example n.
        y : (ignored)

        Returns
        -------
         X : List of numpy ndarrays, each with shape = (I_n, J_n, K), where I_n is the length of sequence1_n, J_n is the
            length of sequence2_n, and K is the number of features.
            Feature matrix list, for use with estimators or further transformers.
        """
        return [self._extract_features(sequence1, sequence2) for sequence1, sequence2 in raw_X]

    def _extract_features(self, sequence1, sequence2):
        """ Helper to extract features for one data point. """
        I = len(sequence1)
        J = len(sequence2)
        K = len(self._binary_features) + sum(num_feats for _, num_feats in self._sparse_features)
        feature_array = np.zeros((I, J, K))
        for i in xrange(I):
            for j in xrange(J):
                for k, feature_function in enumerate(self._binary_features):
                    feature_array[i, j, k] = feature_function(i, j, sequence1, sequence2)
                k = len(self._binary_features)
                for feature_function, num_features in self._sparse_features:
                    feature_array[i, j, k + feature_function(i, j, sequence1, sequence2)] = 1.0
                    k += num_features
        return feature_array


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
        alpha, class_Z, Z = self._forward_probabilities(parameters)
        beta = self._backward(parameters)

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
        _, class_Z, Z = self._forward_probabilities(parameters)
        return {label: np.exp(class_z - Z) for label, class_z in class_Z.iteritems()}

    def _forward_probabilities(self, parameters):
        """ Helper to calculate the predicted probability distribution over classes given some parameters. """
        alpha = self._forward(parameters)
        I, J, _ = self.x.shape

        class_Z = {}
        Z = -np.inf

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[(I - 1, J - 1, state)]
            class_Z[self.states_to_classes[state]] = weight
            Z = np.logaddexp(Z, weight)
        return alpha, class_Z, Z

    def _forward(self, parameters):
        """ Helper to calculate the forward weights.  """
        alpha = defaultdict(lambda: -np.inf)
        for node in self._lattice:
            if len(node) == 3:
                i, j, s = node
                if i == 0 and j == 0:
                    alpha[node] = (np.dot(self.x[i, j, :], parameters[s, :]))
                else:
                    alpha[node] += (np.dot(self.x[i, j, :], parameters[s, :]))
            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
                # Use the features at the destination of the edge.
                edge_potential = ((np.dot(self.x[i1, j1, :], parameters[edge_parameter_index, :]))
                                  + alpha[(i0, j0, s0)])
                alpha[node] = edge_potential
                alpha[(i1, j1, s1)] = np.logaddexp(alpha[(i1, j1, s1)], edge_potential)
        return alpha

    def _backward(self, parameters):
        """ Helper to calculate the backward weights.  """
        beta = defaultdict(lambda: -np.inf)
        I, J, _ = self.x.shape
        for node in reversed(self._lattice):
            if len(node) == 3:
                i, j, s = node
                if i == I - 1 and j == J - 1:
                    beta[node] = 0.0
            else:
                i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
                # Use the features at the destination of the edge.
                edge_potential = beta[(i1, j1, s1)] + (np.dot(self.x[i1, j1, :], parameters[s1, :]))
                beta[node] = edge_potential
                beta[(i0, j0, s0)] = np.logaddexp(beta[(i0, j0, s0)],
                                                  edge_potential + (
                                                  (np.dot(self.x[i1, j1, :], parameters[edge_parameter_index, :]))))
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
