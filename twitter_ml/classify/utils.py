"""Utility methods for classification."""
from typing import List

import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels


class Utils:
    """Helper class of utility functions."""

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
    def get_classification_metrics(y_true, y_pred) -> str:
        """
        Get a classification report for a set of test data and results.

        :param y_true: a vector of expected categories
        :param y_pred: a vector of actual results
        :return: a string containing the report
        """
        return classification_report(y_true, y_pred, output_dict=False)

    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> str:
        """
        Get a confusion matrix for a set of test data and results.

        :param y_true: a vector of expected categories
        :param y_pred: a vector of actual results
        :return: a string containing the confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        y_true, y_pred, classes, normalize=False, title: str = None, cmap=plt.cm.Blues
    ):
        """
        Create a matplotlib confusion matrix.

        Normalization can be applied by setting `normalize=True`.
        Taken from scikit examples https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return ax
