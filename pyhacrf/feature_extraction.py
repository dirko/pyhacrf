# Authors: Dirko Coetsee
# License: 3-clause BSD

""" Implements feature extraction methods to use with HACRF models. """

import numpy as np
import functools

class PairFeatureExtractor(object):
    """ Extract features from sequence pairs.

    A grid is constructed for each sequence pair, for example (['k', 'a', 'a', 's'], ['c', 'h', 'e', 'e', 's', 'e']):

     s * . . . @ .
     a * . . . . .
     a * . . . . .
     k * * * * * *
       c h e e s e

    For each element in the grid, a feature vector is constructed. There are two types of features - real and sparse.
    Real features are functions of the form:
        def some_feature_function(i, j, s1, s2):
            ...
            return some_float
    Given the position in the lattice (i, j) and the two sequences s1 and s2, the function returns a float/int.
    For example, if 3.2 is returned then the feature vector [3.2] is constructed.

    Sparse feature functions look similar:
        def some_feature_function(i, j, s1, s2):
            ...
            return some_index, total_vector_length
    but they always return two ints. The first is the index of the element that should be 1 and the second is the total
    length of vector. So for example if (4, 5) is returned, then the feature vector [0, 0, 0, 0, 1] is constructed.

    After evaluating all the real and sparse functions at a certain lattice position the final feature vector for that
    lattice position is constructed by concatenating all these vectors. For example if there are the real vectors
    [3.2], [0.0], and [1.0] and the sparse vector [0, 0, 0, 0, 1] then the final vector will be
    [3.2, 0.0, 1.0, 0, 0, 0, 0, 1].

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
         X : List of np ndarrays, each with shape = (I_n, J_n, K), where I_n is the length of sequence1_n, J_n is the
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
         X : List of np ndarrays, each with shape = (I_n, J_n, K), where I_n is the length of sequence1_n, J_n is the
            length of sequence2_n, and K is the number of features.
            Feature matrix list, for use with estimators or further transformers.
        """
        return [self._extract_features(sequence1, sequence2) for sequence1, sequence2 in raw_X]

    def _extract_features(self, sequence1, sequence2):
        """ Helper to extract features for one data point. """
        n_sequence1 = len(sequence1) 
        n_sequence2 = len(sequence2) 
        K = (len(self._binary_features) 
             + sum(num_feats for _, num_feats in self._sparse_features))

        feature_array = np.zeros((n_sequence1, n_sequence2, K))

        array1 = np.array(list(sequence1), ndmin=2).T
        array2 = np.array(list(sequence2), ndmin=2)

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

        self._binary_features = [feature for feature, active in zip(binary_features, binary_features_active) if active]
        self._sparse_features = []
        if transition:
            characters_to_index = {character: index for index, character in enumerate(self.CHARACTERS)}
            self._sparse_features.append(((lambda i, j, s1, s2, chars_to_index=characters_to_index:
                                           chars_to_index[s2[j].lower()] +
                                           chars_to_index[s1[i].lower()] * len(chars_to_index)),
                                          len(characters_to_index) ** 2))



def biases(s1, s2, bias=1.0) :
    return np.full((s1.shape[0], s2.shape[1]), bias)

def starts(s1, s2) :
    M = np.zeros((s1.shape[0], s2.shape[1]))
    M[0,...] = 1
    M[...,0] = 1
    return M

def ends(s1, s2) :
    M = np.zeros((s1.shape[0], s2.shape[1]))
    M[(s1.shape[0]-1),...] = 1
    M[...,(s2.shape[1]-1)] = 1
    return M

def matches(s1, s2) :
    return (s1 == s2).astype('i4')

def digits(s1, s2) :
    return (np.char.isdigit(s1) & np.char.isdigit(s2)).astype('i4')




