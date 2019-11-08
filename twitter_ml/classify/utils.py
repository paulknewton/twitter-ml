"""Utility methods for classification."""
from typing import List

# from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Utils:
    """Wrapper helper class."""

    @staticmethod
    def encode_features(
        word_features: List[str], document: List[str]
    ) -> np.array:  # FIXME list or array?
        """
        Generate a features set for a given feature list and word list.

        :param word_features: the list of words that will make up the feature set
        :param document: the text to check (as a list of words)
        :return: a feature set for the document
        """
        doc_words = set(map(str.lower, document))
        feature_vector = []
        for w in word_features:
            feature_vector.append(w in doc_words)

        # le = LabelEncoder()
        # features = le.fit_transform(feature_vector)
        # features = features.reshape(1, -1)
        return np.array(feature_vector).astype(int)

    @staticmethod
    def get_classification_metrics(y_true, y_pred):
        """
        Get a classification report for a set of test data and results.

        :param y_true: a vector of expected categories
        :param y_pred: a vector of actual results
        :return: a string containing the report
        """
        return classification_report(y_true, y_pred, output_dict=False)

    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """
        Get a confusion matrix for a set of test data and results.

        :param y_true: a vector of expected categories
        :param y_pred: a vector of actual results
        :return: a string containing the confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
