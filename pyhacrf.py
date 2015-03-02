# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


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
        classes = dict((target_name, target_number) for target_name, target_number in enumerate(set(y)))
        n_points = len(y)
        if len(X) != n_points:
            raise Exception('Number of training points should be the same as training labels.')

        # Default state machine. Tuple (list_of_states, list_of_transitions)
        state_machine = ([0],
                         [(0, 0, (1, 1)),  # Match
                          (0, 0, (0, 1)),  # Insertion
                          (0, 0, (1, 0))])  # Deletion

        # Initialize the parameters given the state machine, features, and target classes.
        self.parameters = self._initialize_parameters(state_machine, X[0].shape[2], classes)

        # Create a new model object for each training example
        models = [_Model(state_machine, classes, x, ty) for x, ty in zip(X, y)]

        derivative = np.zeros(self.parameters.shape)

        def _objective(parameters):
            derivative.fill(0.0)
            ll = 0.0  # Log likelihood
            # TODO: Embarrassingly parallel
            for model in models:
                model.forward_backward(parameters)
                model.add_derivative(derivative)
                ll += model.ll
            return -ll, -derivative

        self._optimizer_result = fmin_l_bfgs_b(_objective, self.parameters)
        return self

    def predict_proba(self):
        pass

    def predict(self):
        pass

    @staticmethod
    def _initialize_parameters(state_machine, n_features, classes):
        """ Helper to create initial parameter vector with the correct shape. """
        n_states = len(state_machine[0])
        n_transitions = len(state_machine[1])
        n_classes = len(classes.keys())
        return np.zeros((n_features, n_states + n_transitions, n_classes))


class _Model(object):
    """ The actual model that implements the inference routines. """
    def __init__(self, state_machine, classes, x, y):
        self.state_machine = state_machine
        self.classes = classes
        self.x = x
        self.y = y

    def forward_backward(self, parameters):
        """ Run the forward backward algorithm with the given parameters. """

    @staticmethod
    def _build_lattice(x, state_machine):
        """ Helper to construct the list of nodes and edges. """
        I, J, _ = x.shape
        lattice = []
        states, transitions = state_machine
        for i in xrange(I):
            for j in xrange(J):
                for state in states:
                    lattice.append((i, j, state))
                for state1, state2, delta in transitions:
                    if callable(delta):
                        di, dj = delta(i, j, x)
                    else:
                        di, dj = delta
                    if i + di < I and j + dj < J:
                        lattice.append((i, j, i + di, j + dj, state1, state2))
        return lattice
