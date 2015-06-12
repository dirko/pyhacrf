# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements a Hidden Alignment Conditional Random Field (HACRF). """

import numpy as np
import lbfgs
from algorithms import forward, backward
from algorithms import forward_predict, gradient
from state_machine import DefaultStateMachine
from feature_extraction import StringPairFeatureExtractor


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
                 state_machine=None):
        self.parameters = None
        self.classes = None
        self.l2_regularization = l2_regularization
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs

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
            [zip(*sorted(_Model(self._state_machine, x).predict(self.parameters).items(),
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
        self.y = y
        self._lattice = self.state_machine.build_lattice(self.x)

    @profile
    def forward_backward(self, parameters):
        """ Run the forward backward algorithm with the given parameters. """
        x_dot_parameters = np.dot(self.x, parameters.T)  # Pre-compute the dot product
        alpha = self._forward(x_dot_parameters)
        beta = self._backward(x_dot_parameters)
        I, J, _ = self.x.shape
        return gradient(alpha, beta, parameters, self.states_to_classes, self.x, self.y, I, J)

    def predict(self, parameters):
        """ Run forward algorithm to find the predicted distribution over classes. """
        x_dot_parameters = np.dot(self.x, parameters.T)  # Pre-compute the dot product
        alpha = forward_predict(self._lattice, x_dot_parameters, 
                                self.state_machine.n_states)
        I, J, _ = self.x.shape

        class_Z = {}
        Z = -np.inf

        for state, predicted_class in self.states_to_classes.items():
            weight = alpha[I - 1, J - 1, state]
            class_Z[self.states_to_classes[state]] = weight
            Z = np.logaddexp(Z, weight)

        return {label: np.exp(class_z - Z) for label, class_z in class_Z.iteritems()}

    def _forward(self, x_dot_parameters):
        """ Helper to calculate the forward weights.  """
        return forward(self._lattice, x_dot_parameters, 
                       self.state_machine.n_states)

    def _backward(self, x_dot_parameters):
        """ Helper to calculate the backward weights.  """
        I, J, _ = self.x.shape
        return backward(self._lattice, x_dot_parameters, I, J,
                        self.state_machine.n_states)



def test_fit_predict_regularized():
    incorrect = ['helloooo', 'freshh', 'ffb', 'h0me', 'wonderin', 'relaionship', 'hubby', 'krazii', 'mite', 'tropic']
    correct = ['hello', 'fresh', 'facebook', 'home', 'wondering', 'relationship', 'husband', 'crazy', 'might', 'topic']
    training = zip(incorrect, correct)

    fe = StringPairFeatureExtractor(match=True, numeric=True)
    xf = fe.fit_transform(training)

    model = Hacrf(l2_regularization=10.0)
    model.fit(xf, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    expected_parameters = np.array([[-0.0569188, 0.07413339, 0.],
                                    [0.00187709, -0.06377866, 0.],
                                    [-0.01908823, 0.00586189, 0.],
                                    [0.01721114, -0.00636556, 0.],
                                    [0.01578279, 0.0078614, 0.],
                                    [-0.0139057, -0.00862948, 0.],
                                    [-0.00623241, 0.02937325, 0.],
                                    [0.00810951, -0.01774676, 0.]])

    from numpy.testing import assert_array_almost_equal
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
    assert_array_almost_equal(actual_predict_probas, expected_probas)

    expected_predictions = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    actual_predictions = model.predict(xf)
    assert_array_almost_equal(actual_predictions, expected_predictions)

test_fit_predict_regularized()
