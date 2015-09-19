# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements feature extraction methods to use with HACRF models. """

import numpy as np
import functools
import itertools

class PairFeatureExtractor(object):
    """Extract features from sequence pairs.

    For each feature, a grid is constructed for a sequency pair. The 
    features are stacked, producing a 3 dimensional matrix of 
    dimensions: 

    (length of sequence 1) X (length of sequence 2) X (number of features)

    For example, a 'beginning' character feature grid for the sequences,
    'kaas' and 'cheese' could look like this.

       c h e e s e
     k 1 1 1 1 1 1
     a 1 0 0 0 0 0
     a 1 0 0 0 0 0
     s 1 0 0 0 0 0

    These grids are made from two different types of feature
    functions: real and sparse.

    Real features are functions of the form:

        def some_feature_function(array1, array2):
            ...
            return feature_grid

    Given two sequences, s1 and s1, return a numpy.array with dimensions 
    (length of array1) X (length of array2).

    For performance reasons, we take advantage of numpy broadcasting, and 
    array1 is a column array and array2 is a row array. 

    For a 'matching character' feature between 'kaas' and 'cheese', the
    sequences are transformed and then we use broadcasting

        > array1 = numpy.array([['k'],
                                ['a'],
                                ['a'],
                                ['s']])
        > array2 = numpy.array([['c', 'h', 'e', 'e', 's', 'e'])
        > array1 == array2
        numpy.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0]])

    When writing you own real feature functions, you can assume that
    the arrays will come in with the right shape.    

    Sparse feature functions look similar:

        def some_feature_function(i, j, s1, s2):
            ...
            return some_index, total_vector_length

    but they always return two ints. The first is the index of the
    element that should be 1 and the second is the total length of
    vector. So for example if (4, 5) is returned, then the feature
    vector [0, 0, 0, 0, 1] is constructed.


    Parameters
    ----------
    real: list: optional (default=[])
        List of functions of the form
            def some_feature_function(i, j, s1, s2):
                ...
                return some_float

    sparse: list: optional (default=[])
        List of functions of the form
            def some_feature_function(i, j, s1, s2):
                ...
                return some_index, total_vector_length

    """

    def __init__(self, real=None, sparse=None):
        self._binary_features = []
        if real:
            self._binary_features = real
        self._sparse_features = []
        if sparse:
            self._sparse_features = sparse

    def fit_transform(self, raw_X, y=None):
        """Like transform. Transform sequence pairs to feature arrays that can be used as input to `Hacrf` models.

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
        return self.transform(raw_X)

    def transform(self, raw_X, y=None):
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

        array1 = np.array(tuple(sequence1), ndmin=2).T
        array2 = np.array(tuple(sequence2), ndmin=2)

        K = (len(self._binary_features) 
             + sum(num_feats for _, num_feats in self._sparse_features))

        feature_array = np.zeros((array1.size, array2.size, K), dtype='float64')

        for k, feature_function in enumerate(self._binary_features):
            feature_array[..., k] = feature_function(array1, array2)

        if self._sparse_features:
            n_binary_features = len(self._binary_features)

            for i, j in np.ndindex(len(sequence1), len(sequence2)):
                k = n_binary_features

                for feature_function, num_features in self._sparse_features:
                    
                    feature_array[i, j, k + feature_function(i, j, sequence1, sequence2)] = 1.0
                    k += num_features

        return feature_array


class StringPairFeatureExtractor(PairFeatureExtractor):
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
    bias: float: optional (default=1.0)
        A bias term that is always added to every position in the lattice.

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

    

    def __init__(self, bias=1.0, start=False, end=False, match=False, numeric=False, transition=False):
        # TODO: For longer strings, tokenize and use Levenshtein
        # distance up until a lattice position.  Other (possibly)
        # useful features might be whether characters are consonant or
        # vowel, punctuation, case.
        binary_features_active = [True, start, end, match, numeric]
        binary_features = [functools.partial(biases, bias=bias),
                           starts,
                           ends,
                           matches,
                           digits]

        self._binary_features = [feature 
                                 for feature, active 
                                 in zip(binary_features, 
                                        binary_features_active)
                                 if active]
        self._sparse_features = []
        if transition:
            characters_to_index = {character: index for index, character in enumerate(self.CHARACTERS)}
            curried_charIndex = functools.partial(charIndex,
                                                  char2index = characters_to_index)
            self._sparse_features.append((curried_charIndex, 
                                          len(characters_to_index) ** 2))


def charIndex(i, j, s1, s2, char2index=None) :
    char_i, char_j = s1[i].lower(), s2[j].lower()
    index = char2index[char_j] + char2index[char_i] * len(char2index)
    return index

def biases(s1, s2, bias=1.0) :
    return np.full((s1.size, s2.size), bias)

def starts(s1, s2) :
    M = np.zeros((s1.size, s2.size))
    M[0,...] = 1
    M[...,0] = 1
    return M

def ends(s1, s2) :
    M = np.zeros((s1.size, s2.size))
    M[(s1.size-1),...] = 1
    M[...,(s2.size-1)] = 1
    return M

def matches(s1, s2) :
    return (s1 == s2)

def digits(s1, s2) :
    return np.char.isdigit(s1) & np.char.isdigit(s2)




