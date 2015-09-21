pyhacrf
=======

Hidden alignment conditional random field for classifying string pairs -
a learnable edit distance.

This package aims to implement the HACRF machine learning model with a
``sklearn``-like interface. It includes ways to fit a model to training
examples and score new example.

The model takes string pairs as input and classify them into any number
of classes. In McCallum's original paper the model was applied to the
database deduplication problem. Each database entry was paired with
every other entry and the model then classified whether the pair was a
'match' or a 'mismatch' based on training examples of matches and
mismatches.

I also tried to use it as learnable string edit distance for normalizing
noisy text. See *A Conditional Random Field for Discriminatively-trained
Finite-state String Edit Distance* by McCallum, Bellare, and Pereira,
and the report *Conditional Random Fields for Noisy text normalisation*
by Dirko Coetsee.

Example
-------

.. code:: python

    from pyhacrf import StringPairFeatureExtractor, Hacrf

    training_X = [('helloooo', 'hello'), # Matching examples
                  ('h0me', 'home'),
                  ('krazii', 'crazy'),
                  ('non matching string example', 'no really'), # Non-matching examples
                  ('and another one', 'yep')]
    training_y = ['match',
                  'match',
                  'match',
                  'non-match',
                  'non-match']

    # Extract features
    feature_extractor = StringPairFeatureExtractor(match=True, numeric=True)
    training_X_extracted = feature_extractor.fit_transform(training_X)

    # Train model
    model = Hacrf(l2_regularization=1.0)
    model.fit(training_X_extracted, training_y)

    # Evaluate
    from sklearn.metrics import confusion_matrix
    predictions = model.predict(training_X_extracted)

    print(confusion_matrix(training_y, predictions))
    > [[0 3]
    >  [2 0]]

    print(model.predict_proba(training_X_extracted))
    > [[ 0.94914812  0.05085188]
    >  [ 0.92397711  0.07602289]
    >  [ 0.86756034  0.13243966]
    >  [ 0.05438812  0.94561188]
    >  [ 0.02641275  0.97358725]]

Dependencies
------------

This package depends on ``numpy``. The LBFGS optimizer in ``pylbfgs`` is
used, but alternative optimizers can be passed.

Install
-------

Install by running:

::

    python setup.py install

or from pypi:

::

    pip install pyhacrf

Developing
----------
Clone from repository, then

::

    pip install -r requirements-dev.txt
    cython pyhacrf/*.pyx
    python setup.py install

To deploy to pypi, make sure you have compiled the \*.pyx files to \*.c