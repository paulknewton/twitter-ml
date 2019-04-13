"""Based on the excellent NLTK tutorial at:
    https://pythonprogramming.net/text-classification-nltk-tutorial/"""
import logging
import os
import pickle
import random
import argparse

import nltk
# SciKit classifiers
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

# init logging
logger = logging.getLogger("sentiment-nltk")
for h in logger.handlers:
    logger.removeHandler(h)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

voting_classifier = None
word_features = None


class VoteClassifier(ClassifierI):
    """Classifier based on a collection of other classifiers. Classifiers input based on the majority decisions of the sub-classifiers.
    Note: this could be replaced by the equivalent class in SciKit-Learn."""

    def __init__(self, classifiers):
        """Create voting classifier from a list of (classifier, description) tuples."""
        self._classifiers = classifiers.copy()  # copy the list in case it is changed elsewhere

    def classify(self, features):
        """Classify the features using the list of internal classifiers. Classification is calculated by majority vote."""
        logger.debug("------------")
        votes = []
        for c in self._classifiers:
            v = c[0].classify(features)
            logger.debug("%s: %s" % (c[1], v))
            votes.append(v)
        majority_vote = mode(votes)
        logger.debug("Voting Classifier: %s", majority_vote)

        return majority_vote

    def confidence(self, features):
        """Return the confidence of the classification (rate of +ve votes / classifiers)."""
        votes = []
        for c in self._classifiers:
            v = c[0].classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def saveit(classifier, filename):
    """Save the classifier to a pickle."""
    filename = "models/" + filename
    logger.debug("Saving classifier to %s..." % filename)
    classifier_f = open(filename, "wb")
    pickle.dump(classifier, classifier_f)
    classifier_f.close()


def find_features(word_features, document):
    """Generate a set of features from a document."""
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def create_classifier(c, desc, training_data, testing_data):
    """Create a classifier of a specific type, train the classifier, evaluate the accuracy then save it to a .pickle file."""
    logger.debug("Training %s" % desc)
    trained_c = c.train(training_data)
    logger.debug("%s %% accuracy: %f" % (desc, nltk.classify.accuracy(trained_c, testing_data) * 100))
    saveit(trained_c, desc + ".pickle")

    return trained_c


def create_classifiers(training_data, testing_data):
    """Create a set of classifiers as a tuple of (classifer, description) pairs.. The Voting Classifier is always the first entry."""
    logger.debug("Training classifiers...")

    classifiers = []

    classifiers.append((create_classifier(nltk.NaiveBayesClassifier, "naivebayes", training_data, testing_data),
                        "Naive Bayes classifier from NLTK"))
    classifiers.append((
        create_classifier(SklearnClassifier(MultinomialNB()), "multinomialnb", training_data, testing_data),
        "Multinomial NB classifier from SciKit"))
    classifiers.append(
        (create_classifier(SklearnClassifier(BernoulliNB()), "bernouillinb", training_data, testing_data),
         "Bernouilli NB classifier from SciKit"))
    classifiers.append((
        create_classifier(SklearnClassifier(LogisticRegression()), "logisticregression", training_data, testing_data),
        "Logistic Regression classifier from SciKit"))
    classifiers.append((create_classifier(SklearnClassifier(SGDClassifier()), "sgd", training_data, testing_data),
                        "SGD classifier from SciKit"))
    classifiers.append((
        create_classifier(SklearnClassifier(LinearSVC()), "linearsvc", training_data, testing_data),
        "Linear SVC classifier from SciKit"))
    classifiers.append((create_classifier(SklearnClassifier(NuSVC()), "nusvc", training_data, testing_data),
                        "Nu SVC classifier from SciKit"))

    voting_classifier = VoteClassifier(classifiers)
    saveit(voting_classifier, "voting.pickle")

    classifiers.insert(0, (VoteClassifier(classifiers), "Voting Classifier"))

    return classifiers


def create_feature_sets():
    # get the training set
    if "NLTK_PROXY" in os.environ:
        logger.debug("Using proxy %s" % os.environ["NLTK_PROXY"])
        nltk.set_proxy(os.environ["NLTK_PROXY"])
    nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews

    logger.debug("Building data set...")

    # build list of words (1 list per doc) and pos/neg category
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)

    # extract list of features from most common words
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    logger.debug("Frequency dist of 15 most common words:%s" % all_words.most_common(15))
    logger.debug("Frequency of 'stupid':%d" % all_words["stupid"])
    word_features = list(all_words.keys())[:3000]

    # save feature list
    feature_fn = "features.pickle"
    logger.debug("Saving feature list to %s..." % feature_fn)
    features_f = open(feature_fn, "wb")
    pickle.dump(word_features, features_f)
    features_f.close()

    # build list of feature sets from data
    return [(find_features(word_features, rev), category) for (rev, category) in documents]


def classify_sentiment(text):
    global voting_classifier
    global word_features

    # load classifiers from disk
    logger.debug("Reading classifiers from disk...")
    classifier_f = open("models/voting.pickle", "rb")
    voted_classifier = pickle.load(classifier_f)
    classifier_f.close()

    # load features used by the classifiers (are re-created during training)
    logger.debug("Reading features from disk...")
    features_f = open("models/features.pickle", "rb")
    word_features = pickle.load(features_f)
    features_f.close()
    logging.debug(word_features)

    # build feature set for the text being classified
    feature_set = {}
    feature_set = find_features(word_features, text.split())

    return (voted_classifier.classify(feature_set), voted_classifier.confidence(feature_set))


if __name__ == "__main__":

    # read command-line args
    parser = argparse.ArgumentParser(description='Classifies text sentiment based on skikit and NLTK models.')
    parser.add_argument('-t', dest='train', action='store_true', default=False,
                        help='train models (output in models/*.pickle)')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("args = %s" % args)

    # build new classifiers?
    if args.train:
        logger.debug("Building new classifiers")
        feature_sets = create_feature_sets()
        training_set = feature_sets[:1900]
        testing_set = feature_sets[1900:]

        classifiers = create_classifiers(training_set, testing_set)
        voted_classifier = classifiers[0]  # 1st one in the list

    # run a few tests
    test_data = ["This is good.", "This is bad."]
    for t in test_data:
        (sentiment, confidence) = classify_sentiment(t)
        print("%s\nClassification: %s with Confidence: %f" % (t, sentiment, confidence))
