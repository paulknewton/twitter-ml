"""Main classifier logic."""
import logging
import pickle
import string
from importlib import import_module
from statistics import mode
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from twitter_ml.classify.config import Config
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
        # if len(classifiers) % 2 == 0:
        #     raise ValueError(
        #         "Majority voting classifier needs an odd number of classifiers"
        #     )

        self._raw_classifiers = (
            classifiers.copy()  # copy the list in case it is changed elsewhere
        )
        self._fitted_classifiers: Dict[str, Any] = {}

    @property
    def sub_classifiers(self):
        """Get the internal classifiers used by the voting classifier."""
        return self._fitted_classifiers

    def fit(self, X, y: List[str]):
        """
        Fit the classifier to the test data.

        :return the trained classifier
        """
        logger.info("Training voting classifier")

        logger.debug("len(X, y) = %d, %d", len(X), len(y))
        unique, counts = np.unique(np.array(y), return_counts=True)
        logger.debug(list(zip(unique, counts)))

        # drop any old fitted classifiers
        self._fitted_classifiers = {}

        for label, c in self._raw_classifiers.items():
            logger.debug("Training classifier %s", label)
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
        logger.debug("Classifying with 'VoteClassifier'")
        classifier_list = self._fitted_classifiers

        predictions = []  # predict multiple samples at once

        for id, sample in enumerate(X):
            votes = []
            for label, c in classifier_list.items():
                v = c.predict(sample.reshape(1, -1))[0]  # why need to reshape?
                logger.debug("%s: %s", label, v)
                votes.append(v)
            majority_vote = mode(votes)
            predictions.append(majority_vote)
            logger.debug("Classification for sample %d: %s", id, majority_vote)

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

    def __init__(self, config_filename):
        """Instantiate the class with an internal majority voting classifier and feature set."""
        self._voting_classifier = None
        self.feature_list = None

        # load list of classifiers from config file
        config = Config(config_filename)

        voting_tag = "voting"
        voting_config = config.get_config_value(voting_tag)
        if not voting_config:
            raise KeyError(
                "Could not find section '%s' in config file. Exiting" % voting_config
            )

        # a voting classifer must have an odd number of sub-classifiers to avoid a tied vote
        if len(voting_config) % 2 == 0:
            raise ValueError(
                "A voting classifer must have an odd number of sub-classifiers to avoid a tied vote"
            )

        self.sub_classifiers = {}

        for clf_label, clf_config in voting_config.items():
            logging.debug("Import %s", clf_config["module"])
            module = import_module(clf_config["module"])
            logging.debug("Class %s", clf_config["class"])
            class_ = getattr(module, clf_config["class"])

            self.sub_classifiers[clf_label] = (class_(), clf_config["description"])

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
        logger.info("Saving classifier to %s...", filename)
        with open(filename, "wb") as classifier_f:
            pickle.dump(classifier, classifier_f)

    def init_classifiers(self, X_train, y_train) -> None:
        """
        Create a VoteClassifier from a collection of sub-classifiers and train/test them with the provided data.

        :param X_train: the data to use when training the classifier
        :param X_test: the data to use to evaluate the classifier
        """
        # wrap the sub classifiers in a VoteClassifier
        self._voting_classifier = VoteClassifier(self.sub_classifiers)
        self._voting_classifier.fit(X_train, y_train)
        Sentiment._saveit(self._voting_classifier, "voting.pickle")

    def classify_sentiment(
        self, text: str, sub_classifier: str = None
    ) -> Tuple[np.array, np.array, int]:
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
        feature_encoding = Utils.encode_features(
            self.feature_list,
            text.translate(
                str.maketrans("", "", string.punctuation)
            ).split(),  # remove punctuation
        ).reshape(1, -1)

        if sub_classifier:
            logger.debug("Using sub-classifier: %s", sub_classifier)
            clf = self.voting_classifier.sub_classifiers[sub_classifier]
        else:
            logger.debug("Using vote classifier")
            clf = self.voting_classifier

        return self.feature_list, feature_encoding[0], clf.predict(feature_encoding)[0]
