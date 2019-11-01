"""Utility methods for classification."""
from typing import List

# from sklearn.preprocessing import LabelEncoder
import numpy as np


class Utils:
    """Wrapper helper class."""

    @staticmethod
    def encode_features(word_features: List[str], document: List[str]) -> np.array:  # FIXME list or array?
        """
        Generate a features set for a given feature list and word list.

        :param word_features: the list of words that will make up the feature set
        :param document: the text to check (as a list of words)
        :return: a feature set for the document
        """
        doc_words = set(document)
        feature_vector = []
        for w in word_features:
            feature_vector.append(w in doc_words)

        # le = LabelEncoder()
        # features = le.fit_transform(feature_vector)
        # features = features.reshape(1, -1)
        return np.array(feature_vector).astype(int)
