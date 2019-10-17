"""Based on the excellent NLTK tutorial at:
    https://pythonprogramming.net/text-classification-nltk-tutorial/"""

import logging
import os
import pickle
import random
from statistics import mode
from typing import Dict, Any, Tuple, List

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC

logger = logging.getLogger(__name__)

word_features = None


class VoteClassifier(ClassifierI):
    """Classifier based on a collection of other classifiers. Classifiers input based on the majority decisions of the sub-classifiers.
    Note: this could be replaced by the equivalent class in SciKit-Learn."""

    def __init__(self, classifiers: List[Any]):
        """Create voting classifier from a list of (classifier, description) tuples."""
        self._classifiers = classifiers.copy()  # copy the list in case it is changed elsewhere

    def classify(self, features):
        """Classify the features using the list of internal classifiers. Classification is calculated by majority vote."""
        logger.debug("------------")
        votes = []
        for c in self._classifiers:
            v = c[0].classify(features)
            logger.debug("%s: %s", c[1], v)
            votes.append(v)
        majority_vote = mode(votes)
        logger.debug("Voting Classifier: %s", majority_vote)

        return majority_vote

    def confidence(self, features) -> float:
        """Return the confidence of the classification (rate of +ve votes / classifiers)."""
        votes = []
        for c in self._classifiers:
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
    def find_features(word_features: List[str], document: List[str]) -> Dict[str, bool]:
        """
        Generate a features set for a given feature list and word list.
        :param word_feature: the list of words that will make up the feature set
        :param document: the text to check (as a list of words)
        :return: a feature set for the document
        """
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    @staticmethod
    def create_all_feature_sets() -> List[Tuple[Dict[str, bool], Any]]:
        """
        Create a (feature set, category) tuple for all movie reviews in the NLTK movie review dataset
        :return: a list of feature, category tuples
        """

        # get the moview reviews
        if "NLTK_PROXY" in os.environ:
            logger.debug("Using proxy %s", os.environ["NLTK_PROXY"])
            nltk.set_proxy(os.environ["NLTK_PROXY"])
        nltk.download('movie_reviews')
        from nltk.corpus import movie_reviews

        logger.debug("Building data set...")

        # build list of words (1 list per doc) and pos/neg category
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)

        # extract the most common words to use for building features
        all_words = []
        for w in movie_reviews.words():
            all_words.append(w.lower())
        all_words_dist = nltk.FreqDist(all_words)
        # logger.debug("Frequency dist of 15 most common words:%s", all_words_dist.most_common(15))
        # logger.debug("Frequency of 'stupid':%d", all_words_dist["stupid"])
        features = list(all_words_dist.keys())[:3000]

        # save the words used in the feature list (used when classifying)
        feature_fn = "models/features.pickle"
        logger.debug("Saving feature list to %s...", feature_fn)
        with open(feature_fn, "wb") as features_f:
            pickle.dump(features, features_f)

        # build list of feature, category profiles for each review in the movie DB
        return [(Sentiment.find_features(features, review), category) for (review, category) in documents]

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

    def retrain_classifiers(self, training_data, testing_data) -> None:
        """
        Create a VoteClassifier from a collection of sub-classifiers and train/test them with the provided data.
        :param training_data: the data to use when training the classifier
        :param testing_data: the data to use to evaluate the classifier
        """
        logger.debug("Training classifiers...")

        # create the sub classifiers, and save this to disk in case we need them later
        sub_classifiers = [
            (Sentiment._create_classifier(nltk.NaiveBayesClassifier, "naivebayes", training_data, testing_data),
             "Naive Bayes classifier from NLTK"), (
                Sentiment._create_classifier(SklearnClassifier(MultinomialNB()), "multinomialnb", training_data,
                                             testing_data),
                "Multinomial NB classifier from SciKit"), (
                Sentiment._create_classifier(SklearnClassifier(BernoulliNB()), "bernouillinb", training_data,
                                             testing_data),
                "Bernouilli NB classifier from SciKit"), (
                Sentiment._create_classifier(SklearnClassifier(LogisticRegression()), "logisticregression",
                                             training_data,
                                             testing_data),
                "Logistic Regression classifier from SciKit"),
            (Sentiment._create_classifier(SklearnClassifier(SGDClassifier()), "sgd", training_data, testing_data),
             "SGD classifier from SciKit"), (
                Sentiment._create_classifier(SklearnClassifier(LinearSVC()), "linearsvc", training_data, testing_data),
                "Linear SVC classifier from SciKit"),
            (Sentiment._create_classifier(SklearnClassifier(NuSVC()), "nusvc", training_data, testing_data),
             "Nu SVC classifier from SciKit")]

        # wrap the sub classifiers in a VoteClassifier
        self.voting_classifier = VoteClassifier(sub_classifiers)
        Sentiment._saveit(self.voting_classifier, "voting.pickle")

    def classify_sentiment(self, text) -> Tuple[Any, float]:
        """
        Classify a piece of text as positive ("pos") or negative ("neg")
        :param text: the text to classify
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
                logging.debug(word_features)

        # build feature set for the text being classified
        feature_set = Sentiment.find_features(self.word_features, text.split())

        return self.voting_classifier.classify(feature_set), self.voting_classifier.confidence(feature_set)
