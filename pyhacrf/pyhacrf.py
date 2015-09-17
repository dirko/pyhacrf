# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
import lbfgs
from .algorithms import forward, backward
from .algorithms import forward_predict, forward_max_predict
from .algorithms import gradient, gradient_sparse, populate_sparse_features, sparse_multiply
from .state_machine import DefaultStateMachine


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

    state_machine : Instance of `GeneralStateMachine` or `DefaultStateMachine`, optional (default=`DefaultStateMachine`)
        The state machine to use to generate the lattice.

    viterbi : Boolean, optional (default=False).
        Whether to use Viterbi (max-sum) decoding for predictions (not training)
        instead of the default sum-product algorithm.

    References
    ----------
    See *A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance*
    by McCallum, Bellare, and Pereira, and the report *Conditional Random Fields for Noisy text normalisation*
    by Dirko Coetsee.
    """

    def __init__(self,
                 l2_regularization=0.0,
                 optimizer=None,
                 optimizer_kwargs=None,
                 state_machine=None,
                 viterbi=False):
        self.parameters = None
        self.classes = None
        self.l2_regularization = l2_regularization
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self.viterbi = viterbi

        self._optimizer_result = None
        self._state_machine = state_machine
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

        if not self._state_machine:
            self._state_machine = DefaultStateMachine(self.classes)

        # Initialize the parameters given the state machine, features, and target classes.
        self.parameters = self._initialize_parameters(self._state_machine, X[0].shape[2])

        # Create a new model object for each training example
        models = [_Model(self._state_machine, x, ty) for x, ty in zip(X, y)]

        self._evaluation_count = 0

        def _objective(parameters):
            gradient = np.zeros(self.parameters.shape)
            ll = 0.0  # Log likelihood
            # TODO: Embarrassingly parallel
            for model in models:
                dll, dgradient = model.forward_backward(parameters.reshape(self.parameters.shape))
                ll += dll
                gradient += dgradient

            parameters_without_bias = np.array(parameters, dtype='float64')  # exclude the bias parameters from being regularized
            parameters_without_bias[0] = 0
            ll -= self.l2_regularization * np.dot(parameters_without_bias.T, parameters_without_bias)
            gradient = gradient.flatten() - 2.0 * self.l2_regularization * parameters_without_bias

            if verbosity > 0:
                if self._evaluation_count == 0:
                    print('{:10} {:10} {:10}'.format('Iteration', 'Log-likelihood', '|gradient|'))
                if self._evaluation_count % verbosity == 0:
                    print('{:10} {:10.4} {:10.4}'.format(self._evaluation_count, ll, (abs(gradient).sum())))
            self._evaluation_count += 1

            # TODO: Allow some of the parameters to be frozen. ie. not trained. Can later also completely remove
            # TODO:     the computation associated with these parameters.
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
        
        parameters = np.ascontiguousarray(self.parameters.T)

        predictions = [_Model(self._state_machine, x).predict(parameters, self.viterbi)
                       for x in X]
        predictions = np.array([[probability
                                 for _, probability
                                 in sorted(prediction.items())]
                                for prediction in predictions])
        return predictions

    def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len,
            string2_len, n_features), where string1_len and
            string2_len are the length of the two training strings and
            n_features the number of features.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.

        """
        return [self.classes[prediction.argmax()] for prediction in self.predict_proba(X)]

    @staticmethod
    def _initialize_parameters(state_machine, n_features):
        """ Helper to create initial parameter vector with the correct shape. """
        return np.zeros((state_machine.n_states 
                         + state_machine.n_transitions,
                         n_features))

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
    def __init__(self, state_machine, x, y=None):
        self.state_machine = state_machine
        self.states_to_classes = state_machine.states_to_classes
        self.x = x
        self.sparse_x = 'uninitialized'
        self.y = y
        self._lattice = self.state_machine.build_lattice(self.x)

    def forward_backward(self, parameters):
        """ Run the forward backward algorithm with the given parameters. """
        # If the features are sparse, we can use an optimization.
        # I'm not using scipy.sparse here because we want to avoid a scipy dependency and also scipy.sparse doesn't seem
        # to handle arrays of shape higher than 2.
        if isinstance(self.sparse_x, str) and self.sparse_x == 'uninitialized':
            if (self.x == 0).sum() * 1.0 / self.x.size > 0.6:
                self.sparse_x = self._construct_sparse_features(self.x)
            else:
                self.sparse_x = 'not sparse'

        I, J, K = self.x.shape
        if not isinstance(self.sparse_x, str):
            C = self.sparse_x[0].shape[2]
            S, _ = parameters.shape
            x_dot_parameters = np.zeros((I, J, S))
            sparse_multiply(x_dot_parameters, self.sparse_x[0], self.sparse_x[1], parameters.T, I, J, K, C, S)
        else:
            x_dot_parameters = np.dot(self.x, parameters.T)  # Pre-compute the dot product
        alpha = self._forward(x_dot_parameters)
        beta = self._backward(x_dot_parameters)
        classes_to_ints = {k: i for i, k in enumerate(set(self.states_to_classes.values()))}
        states_to_classes = np.array([classes_to_ints[self.states_to_classes[state]]
                                      for state in range(max(self.states_to_classes.keys()) + 1)], dtype='int64')
        if not isinstance(self.sparse_x, str):
            ll, deriv = gradient_sparse(alpha, beta, parameters, states_to_classes,
                                        self.sparse_x[0], self.sparse_x[1], classes_to_ints[self.y],
                                        I, J, self.sparse_x[0].shape[2])
        else:
            ll, deriv = gradient(alpha, beta, parameters, states_to_classes,
                                 self.x, classes_to_ints[self.y], I, J, K)
        return ll, deriv

    def predict(self, parameters, viterbi):
        """ Run forward algorithm to find the predicted distribution over classes. """
        x_dot_parameters = np.einsum('ijk,kl->ijl', self.x, parameters)

        if not viterbi:
            alpha = forward_predict(self._lattice, x_dot_parameters,
                                    self.state_machine.n_states)
        else:
            alpha = forward_max_predict(self._lattice, x_dot_parameters,
                                        self.state_machine.n_states)

        I, J, _ = self.x.shape

        class_Z = {}
        Z = -np.inf

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[I - 1, J - 1, state]
            class_Z[self.states_to_classes[state]] = weight
            Z = np.logaddexp(Z, weight)

        return {label: np.exp(class_z - Z) for label, class_z in class_Z.items()}

    def _forward(self, x_dot_parameters):
        """ Helper to calculate the forward weights.  """
        return forward(self._lattice, x_dot_parameters, 
                       self.state_machine.n_states)

    def _backward(self, x_dot_parameters):
        """ Helper to calculate the backward weights.  """
        I, J, _ = self.x.shape
        return backward(self._lattice, x_dot_parameters, I, J,
                        self.state_machine.n_states)

    def _construct_sparse_features(self, x):
        """ Helper to construct a sparse representation of the features. """
        I, J, K = x.shape
        new_array_height = (x != 0).sum(axis=2).max()
        index_array = -np.ones((I, J, new_array_height), dtype='int64')
        value_array = -np.ones((I, J, new_array_height), dtype='float64')
        populate_sparse_features(x, index_array, value_array, I, J, K)
        return index_array, value_array
