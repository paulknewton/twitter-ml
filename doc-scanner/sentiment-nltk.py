"""Based on the excellent NLTK tutorial at:
    https://pythonprogramming.net/text-classification-nltk-tutorial/"""
import pickle
import random, os, logging
import nltk

# SciKit classifiers
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# init logging
logger = logging.getLogger("x")
for h in logger.handlers:
    logger.removeHandler(h)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def saveit(classifier, filename):
    """Save the classifier to a pickle."""

    logger.debug("Saving classifier to %s..." % filename)
    classifier_f = open(filename, "wb")
    pickle.dump(classifier, classifier_f)
    classifier_f.close()


# our custom classifier
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    """Classifier based on a collection of other classifiers. Classifiers input based on the majority decisions of the sub-classifiers"""

    def __init__(self, classifiers):
        self._classifiers = classifiers.copy()  # copy the list in case it is changed elsewhere

    def classify(self, features):
        """Classify the features using the list of internal classifiers. Classification is calculated by majority vote."""

        #logger.debug("Classifying with %d classifiers" % len(self._classifiers))
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        """Return the confidence of the classification (rate of +ve votes / classifiers)."""
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


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
    logger.debug("%s accuracy percent: %f" % (desc, nltk.classify.accuracy(trained_c, testing_data) * 100))
    saveit(trained_c, desc + ".pickle")

    return trained_c


def create_classifiers(training_data, testing_data):
    """Create a set of classifiers."""
    logger.debug("Training classifiers...")

    classifiers = []

    classifiers.append(create_classifier(nltk.NaiveBayesClassifier, "naivebayes", training_set, testing_data), "Naive Bayes classifier from NLTK")
    classifiers.append(
        create_classifier(SklearnClassifier(MultinomialNB()), "multinomialnb", training_data, testing_set), "Multinomial NB classifier from SciKit")
    classifiers.append(create_classifier(SklearnClassifier(BernoulliNB()), "bernouillinb", training_data, testing_set), "Bernouilli NB classifier from SciKit")
    classifiers.append(
        create_classifier(SklearnClassifier(LogisticRegression()), "logisticregression", training_data, testing_set), "Logistic Regression classifier from SciKit")
    classifiers.append(create_classifier(SklearnClassifier(SGDClassifier()), "sgd", training_data, testing_data), "SGD classifier from SciKit")
    classifiers.append(
        create_classifier(SklearnClassifier(LinearSVC()), "linearsvc.pickle", training_data, testing_data), "Linear SVC classifier from SciKit")
    classifiers.append(create_classifier(SklearnClassifier(NuSVC()), "nusvc.pickle", training_data, testing_data), "Nu SVC classifier from SciKit")

    voted_classifier = VoteClassifier(classifiers)
    saveit(voted_classifier, "voting.pickle")

    classifiers.insert(0, VoteClassifier(classifiers), "Voting Classifier")

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

    # build list of feature sets from data
    return [(find_features(word_features, rev), category) for (rev, category) in documents]


if __name__ == "__main__":
    feature_sets = create_feature_sets()
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    train = True

    if train:
        classifiers = create_classifiers(training_set, testing_set)
        voted_classifier = classifiers[0]
    else:
        classifier_f = open("voted.pickle", "rb")
        voted_classifier = pickle.load(classifier_f)
        classifier_f.close()

    for data in testing_set[:20]:
        logger.info("Classification: %s Confidence: %f" % (
        voted_classifier.classify(data[0]), voted_classifier.confidence(data[0]) * 100))
    logger.info("voted_classifier accuracy percent: %f" % (nltk.classify.accuracy(voted_classifier, testing_set) * 100))
