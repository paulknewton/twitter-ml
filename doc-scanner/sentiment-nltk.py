"""Based on the excellent NLTK tutorial at:
    https://pythonprogramming.net/text-classification-nltk-tutorial/"""
import pickle
import random, os, logging
import nltk

# SciKit classifiers
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# init logging
logger = logging.getLogger("x")
for h in logger.handlers:
    logger.removeHandler(h)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def saveit(classifier, filename):
    """Save the classifier to a pickle."""
    
    logger.debug("Saving classifier to %s..." % filename)
    classifier_f = open(filename,"wb")
    pickle.dump(classifier, classifier_f)
    classifier_f.close()

    
# our custom classifier
from nltk.classify import ClassifierI
from statistics import mode
class VoteClassifier(ClassifierI):
    """Classifier based on a collection of other classifiers. Classifiers input based on the majority decisions of the sub-classifiers"""

    def __init__(self, classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

def find_features(word_features, document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def create_classifiers():
    logger.debug("Training classifiers...")
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    saveit(classifier, "naivebayes.pickle")
    logger.debug("Naive Bayes accuracy percent: %f" % (nltk.classify.accuracy(classifier, testing_set)*100))
    #classifier.show_most_informative_features(15)
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    saveit(MNB_classifier, "multinomialnb.pickle")
    logger.debug("MNB_classifier accuracy percent: %f" % (nltk.classify.accuracy(MNB_classifier, testing_set)*100))
    
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    saveit(BernoulliNB_classifier, "bernouillinb.pickle")
    logger.debug("BernoulliNB_classifier accuracy percent: %f", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)*100))
    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    saveit(LogisticRegression_classifier, "logisticregression.pickle")
    logger.debug("LogisticRegression_classifier accuracy percent: %f" % (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100))
    
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    saveit(SGDClassifier_classifier, "sgd.pickle")
    logger.debug("SGDClassifier_classifier accuracy percent: %f" % (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)*100))
    
    ##SVC_classifier = SklearnClassifier(SVC())
    ##SVC_classifier.train(training_set)
    ##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
    
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    saveit(LinearSVC_classifier, "linearsvc.pickle")
    logger.debug("LinearSVC_classifier accuracy percent: %f" % (nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100))
    
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    saveit(NuSVC_classifier, "nusvc.pickle")
    logger.debug("NuSVC_classifier accuracy percent: %f" % (nltk.classify.accuracy(NuSVC_classifier, testing_set)*100))
    
    # aggregate these into our custom classifier
    classifiers = (classifier,
                   NuSVC_classifier,
                   LinearSVC_classifier,
                   SGDClassifier_classifier,
                   MNB_classifier,
                   BernoulliNB_classifier,
                   LogisticRegression_classifier)

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
    
    # build feature sets from data
    return [(find_features(word_features, rev), category) for (rev, category) in documents]
    

feature_sets = create_feature_sets()        
training_set = feature_sets[:1900]
testing_set =  feature_sets[1900:]

if False:
    classifiers = create_classifiers()
    voted_classifier = VoteClassifier(classifiers)
    saveit(voted_classifier, "voted.pickle")

classifier_f = open("voted.pickle","rb")
voted_classifier = pickle.load(classifier_f)
classifier_f.close()
    
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[1][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[2][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[3][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[4][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[5][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("Classification: %s Confidence: %f" % (voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0])*100))
logger.info("voted_classifier accuracy percent: %f" % (nltk.classify.accuracy(voted_classifier, testing_set)*100))
