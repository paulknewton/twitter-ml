"""Unit tests for the CustomVoteClassifier."""
import pytest
from twitter_ml.classify.sentiment import Sentiment


def test_even_classifiers_in_voting_classifier():
    """Test that a Voting Classifier must have an odd number of classifiers (to avoid a tied vote)."""
    with pytest.raises(ValueError):
        Sentiment("tests/test_even_classifiers.yaml")


def test_accessors():
    """Test that the accessors return the same types as sklearn estimators."""
    clfs = {
        "multinomilnb": "sklearn.naive_bayes.MultinomialNB",
        "bernouillinb": "sklearn.naive_bayes.BernoulliNB",
        "logisticregression": "sklearn.linear_model.LogisticRegression",
        "sgd": "sklearn.linear_model.SGDClassifier",
        "linearrsvc": "sklearn.svm.LinearSVC",
    }

    st = Sentiment("tests/test_voting.yaml")

    eclf = st.voting_classifier
    assert eclf

    # raw estimators
    estimators = eclf.estimators
    assert len(estimators) == len(clfs)
    labels = [label for label, _ in estimators]
    assert labels - clfs.keys() == set()

    # fitted estimators
    estimators_ = eclf.estimators_
    assert len(estimators_) == len(clfs)

    # dict
    named_estimators_ = eclf.named_estimators_
    assert len(named_estimators_) == len(clfs)
    assert named_estimators_.keys() - clfs.keys() == set()
