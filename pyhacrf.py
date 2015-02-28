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
        n_classes = len(set(y))
        n_points = len(y)
        if len(X) != n_points:
            raise Exception('Number of training points should be the same as training labels.')

        # Create a new model object for each training example
        models = [_Model(state_machine, n_classes, x, ty) for x, ty in zip(X, y)]

        derivative = np.zeros(self.parameters.shape)

        def _objective(parameters):
            derivative.fill(0.0)
            ll = 0.0  # Log likelihood
            # TODO: Embarrassingly parallel
            for model in models:
                model.set_parameters(parameters)
                model.forward_backward()
                model.add_derivative(derivative)
                ll += model.ll
            return -ll, -derivative

        self._optimizer_result = fmin_l_bfgs_b(_objective, self.parameters)
        return self

    def predict_proba(self):
        pass

    def predict(self):
        pass


class _Model(object):
    """ The actual model that implements the inference routines. """
    def __init__(self, state_machine,