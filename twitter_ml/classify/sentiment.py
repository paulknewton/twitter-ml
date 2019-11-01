"""Main classifier logic."""
import logging
import pickle
from statistics import mode
from typing import Any, Tuple, Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

from twitter_ml.classify.utils import Utils

logger = logging.getLogger(__name__)


class VoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier based on a collection of other classifiers.

    Classifiers input based on the majority decisions of the sub-classifiers.
    Note: this could be replaced by the equivalent class in SciKit-Learn.
    """

    def __init__(self, classifiers: Dict[str, Tuple[Any, str]]):
        """
        Create voting classifier from a list of (classifier, description) tuples.

        :param classifiers: the list of sub-classifiers used in the voting
        """
        if len(classifiers) % 2 == 0:
            raise ValueError("Majority voting classifier needs an odd number of classifiers")

        self._raw_classifiers = classifiers.copy()  # copy the list in case it is changed elsewhere
        self._fitted_classifiers: Dict[str, Any] = {}

    @property
    def sub_classifiers(self):
        """Get the internal classifiers used by the voting classifier."""
        return self._raw_classifiers

    def fit(self, X, y: List[str]):
        """Fit the classifier to the test data."""
        # drop any old fitted classifiers
        self._fitted_classifiers = {}

        for label, c in self._raw_classifiers.items():
            logger.debug("Training classifiers...")
            self._fitted_classifiers[label] = c[0].fit(X, y)
            # logger.debug("%s %% accuracy: %f", desc, nltk.classify.accuracy(trained_c, testing_data) * 100)
            # Sentiment._saveit(trained_c, desc + ".pickle")

        # allow chaining
        return self

    # TODO change signature
    def predict(self, X) -> List[int]:
        """
        Classify the features using the list of internal classifiers - classification is calculated by majority vote.

        :param X: the features to classify
        :return calculated category of the features (pos/neg)
        """
        logger.info("Classifying with 'VoteClassifier'")
        classifier_list = self._fitted_classifiers

        predictions = []  # predict multiple samples at once

        for sample in X:
            votes = []
            for label, c in classifier_list.items():
                v = c.predict(X)[0]
                logger.info("%s: %s", label, v)
                votes.append(v)
            majority_vote = mode(votes)
            predictions.append(majority_vote)

        return predictions

    def confidence(self, features, sub_classifier: str = None) -> float:
        """
        Return the confidence of the classification.

        :param sub_classifier: optional label indicating the sub-classifier to use (default: use the VoteClassifier)
        :param features: the features to classify
        :return rate of +ve votes / classifiers
        """
        raise NotImplementedError()
        if sub_classifier:
            logger.info("Calculating confidence with '%s'", sub_classifier)
            classifier_list = {sub_classifier: self.sub_classifiers[sub_classifier]}
        else:
            logger.info("Calculating confidence with 'VoteClassifier'")
            classifier_list = self.sub_classifiers

        votes = []
        for label, c in classifier_list.items():
            v = c[0].predict(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


class Sentiment:
    """Class to initialise classifiers and expose common classification methods."""

    def __init__(self):
        """Instantiate the class with an internal majority voting classifier and feature set."""
        self._voting_classifier = None
        self.feature_list = None

    @property
    def sub_classifiers(self):
        """
        Return the internal classifiers used inside the majority voting classifier.

        :return: the list of sub-classifiers
        """
        return self.voting_classifier.sub_classifiers

    @property
    def voting_classifier(self) -> VoteClassifier:
        """
        Return the majority voting classifier to perform the classification.

        :return: the classifier
        """
        if not self._voting_classifier:
            # load classifiers from disk
            logger.debug("Reading classifiers from disk...")
            with open("models/voting.pickle", "rb") as classifier_f:
                self._voting_classifier = pickle.load(classifier_f)
        return self._voting_classifier

    @staticmethod
    def _saveit(classifier, filename: str) -> None:
        """
        Save a classifier to disk.

        :param classifier: the classifier
        :param filename: the filename
        """
        filename = "models/" + filename
        logger.debug("Saving classifier to %s...", filename)
        with open(filename, "wb") as classifier_f:
            pickle.dump(classifier, classifier_f)

    def init_classifiers(self, X_train, X_test, y_train, y_test) -> None:
        """
        Create a VoteClassifier from a collection of sub-classifiers and train/test them with the provided data.

        :param X_train: the data to use when training the classifier
        :param X_test: the data to use to evaluate the classifier
        """
        # create the sub classifiers, and save this to disk in case we need them later
        sub_classifiers = {
            # "naivebayes": (
            # Sentiment._create_classifier(nltk.NaiveBayesClassifier, "naivebayes", training_data,
            #                              testing_data),
            # "Naive Bayes classifier from NLTK"),
            "multinomilnb": (MultinomialNB(),
                             "Multinomial NB classifier from SciKit"),
            "bernouillinb": (BernoulliNB(),
                             "Bernouilli NB classifier from SciKit"),
            "logisticregression": (LogisticRegression(),
                                   "Logistic Regression classifier from SciKit"),
            "sgd": (SGDClassifier(),
                    "SGD classifier from SciKit"),
            "linearrsvc": (LinearSVC(),
                           "Linear SVC classifier from SciKit")
            # ,
            # "nusvc": (NuSVC(), "nusvc",
            #           "Nu SVC classifier from SciKit")
        }

        # wrap the sub classifiers in a VoteClassifier
        self._voting_classifier = VoteClassifier(sub_classifiers).fit(X_train, y_train)
        Sentiment._saveit(self._voting_classifier, "voting.pickle")

    def classify_sentiment(self, text: str, sub_classifier: str = None) -> Tuple[np.array, np.array, int]:
        """
        Classify a piece of text as positive ("pos") or negative ("neg").

        :param text: the text to classify
        :param sub_classifier: name of the specific sub-classifier to use (default: use a voting classifier)
        :return: tuple of (feature_list, feature_encoding, category)
        """
        if not self.feature_list:
            # load features used by the classifiers (are created during training)
            logger.debug("Reading features from disk...")
            with open("models/features.pickle", "rb") as features_f:
                self.feature_list = pickle.load(features_f)

        # build feature set for the text being classified
        feature_encoding = Utils.encode_features(self.feature_list, text.split()).reshape(1, -1)

        if sub_classifier:
            logger.debug("Using sub-classifier: %s", sub_classifier)
            clf = self.voting_classifier.sub_classifiers[sub_classifier]
        else:
            logger.debug("Using vote classifier")
            clf = self.voting_classifier

        return self.feature_list, feature_encoding[0], clf.predict(feature_encoding)[0]
