"""Based on the excellent NLTK tutorial at https://pythonprogramming.net/text-classification-nltk-tutorial/"""
import logging
import pickle
from statistics import mode
from typing import Any, Tuple, Dict

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC

from twitter_ml.classify.utils import Utils

logger = logging.getLogger(__name__)


class VoteClassifier(ClassifierI):
    """Classifier based on a collection of other classifiers. Classifiers input based on the majority decisions of the sub-classifiers.
    Note: this could be replaced by the equivalent class in SciKit-Learn."""

    def __init__(self, classifiers: Dict[str, Tuple[Any, str]]):
        """
        Create voting classifier from a list of (classifier, description) tuples.
        :param classifiers: the list of sub-classifiers used in the voting
        """
        self._classifiers = classifiers.copy()  # copy the list in case it is changed elsewhere

    def classify(self, features, sub_classifier: str=None):
        """
        Classify the features using the list of internal classifiers. Classification is calculated by majority vote.
        :param features: the features to classify
        :param sub_classifier: the name of the sub-classifier to use (default: use all classifiers and take a vote
        :return calculated category of the features (pos/neg)
        """
        if sub_classifier:
            logger.info("Using sub-classifier '%s'", sub_classifier)
            classifier_list = {sub_classifier: self._classifiers[sub_classifier]}
        else:
            logger.info("Using VoteClassifier")
            classifier_list = self._classifiers

        votes = []
        for label, c in classifier_list.items():
            v = c[0].classify(features)
            logger.info("%s: %s: %s", label, c[1], v)
            votes.append(v)
        majority_vote = mode(votes)
        return majority_vote

    def confidence(self, features, sub_classifier: str=None) -> float:
        """
        Return the confidence of the classification
        :param features: the features to classify
        :return rate of +ve votes / classifiers
        """
        if sub_classifier:
            logger.info("Using sub-classifier '%s'", sub_classifier)
            classifier_list = { sub_classifier: self._classifiers[sub_classifier] }
        else:
            logger.info("Using VoteClassifier")
            classifier_list = self._classifiers

        votes = []
        for _label, c in classifier_list.items():
            v = c[0].classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    def labels(self):
        return ["pos", "neg"]


class Sentiment:

    def __init__(self):
        self.voting_classifier = None
        self.word_features = None

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

    @staticmethod
    def _create_classifier(c, desc: str, training_data, testing_data) -> None:
        """
        Create a classifier of a specific type, train the classifier, evaluate the accuracy then save it to a .pickle file.
        :param c: class of the classifier
        :param desc: label to identify the classifier
        :param training_data: data to use when training the classifier
        :param testing_data: data to use to evaluate the classifier
        """
        logger.debug("Training %s", desc)
        trained_c = c.train(training_data)
        logger.debug("%s %% accuracy: %f", desc, nltk.classify.accuracy(trained_c, testing_data) * 100)
        Sentiment._saveit(trained_c, desc + ".pickle")

        return trained_c

    def rebuild_classifiers(self, training_data, testing_data) -> None:
        """
        Create a VoteClassifier from a collection of sub-classifiers and train/test them with the provided data.
        :param training_data: the data to use when training the classifier
        :param testing_data: the data to use to evaluate the classifier
        """
        logger.debug("Training classifiers...")

        # create the sub classifiers, and save this to disk in case we need them later
        sub_classifiers = {
            "naivebayes": (
                Sentiment._create_classifier(nltk.NaiveBayesClassifier, "naivebayes", training_data,
                                             testing_data),
                "Naive Bayes classifier from NLTK"),
            "multinomilnb": (
                Sentiment._create_classifier(SklearnClassifier(MultinomialNB()), "multinomialnb",
                                             training_data,
                                             testing_data),
                "Multinomial NB classifier from SciKit"),
            "bernouillinb": (
                Sentiment._create_classifier(SklearnClassifier(BernoulliNB()), "bernouillinb",
                                             training_data,
                                             testing_data),
                "Bernouilli NB classifier from SciKit"),
            "logisticregression": (
                Sentiment._create_classifier(SklearnClassifier(LogisticRegression()),
                                             "logisticregression",
                                             training_data,
                                             testing_data),
                "Logistic Regression classifier from SciKit"),
            "sgd": (
                Sentiment._create_classifier(SklearnClassifier(SGDClassifier()), "sgd", training_data,
                                             testing_data),
                "SGD classifier from SciKit"),
            "linearrsvc": (
                Sentiment._create_classifier(SklearnClassifier(LinearSVC()), "linearsvc", training_data,
                                             testing_data),
                "Linear SVC classifier from SciKit"),
            "nusvc": (Sentiment._create_classifier(SklearnClassifier(NuSVC()), "nusvc", training_data,
                                                   testing_data),
                      "Nu SVC classifier from SciKit")
        }

        # wrap the sub classifiers in a VoteClassifier
        self.voting_classifier = VoteClassifier(sub_classifiers)
        Sentiment._saveit(self.voting_classifier, "voting.pickle")

    def classify_sentiment(self, text: str, sub_classifier: str=None) -> Tuple[Any, float]:
        """
        Classify a piece of text as positive ("pos") or negative ("neg")
        :param text: the text to classify
        :param sub_classifier: name of the specific sub-classifier to use (default: use a voting classifier)
        :return: pos or neg
        """
        if not self.voting_classifier:
            # load classifiers from disk
            logger.debug("Reading classifiers from disk...")
            with open("models/voting.pickle", "rb") as classifier_f:
                self.voting_classifier = pickle.load(classifier_f)

        if not self.word_features:
            # load features used by the classifiers (are created during training)
            logger.debug("Reading features from disk...")
            with open("models/features.pickle", "rb") as features_f:
                self.word_features = pickle.load(features_f)
                logging.debug(self.word_features)

        # build feature set for the text being classified
        feature_set = Utils.get_feature_vector(self.word_features, text.split())

        return self.voting_classifier.classify(feature_set, sub_classifier), self.voting_classifier.confidence(feature_set, sub_classifier)
