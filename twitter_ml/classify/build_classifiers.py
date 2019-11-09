#! /usr/bin/env python3
"""Re-create classifiers based on training data."""
import argparse
import logging.config
import sys

import numpy as np
import yaml
from twitter_ml.classify.movie_reviews import MovieReviews
from twitter_ml.classify.sentiment import Sentiment
from twitter_ml.classify.utils import Utils

with open("logging.yaml", "rt") as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)


def do_report(X, y):
    """
    Print key matrics for a set of test data.

    :param X: a matrix of samples
    :param y: a vector of categories
    """
    logger.debug("Samples: len(X, y) = %d, %d" % (len(X), len(y)))
    unique, counts = np.unique(np.array(y), return_counts=True)
    logger.debug("Categories: " + str(list(zip(unique, counts))))

    y_pred = classifier.voting_classifier.predict(X)
    print("Metrics:\n" + Utils.get_classification_metrics(y, y_pred))
    print("Confusion matrix:")
    print(Utils.get_confusion_matrix(y, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds scikit-learn/nltk classifiers based on training data."
    )
    parser.add_argument(
        "--features", action="store_true", default=False, help="list features and exit"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="print classification report and exit",
    )
    args = parser.parse_args()

    data = MovieReviews(3000)

    if args.features:
        print("Features:")
        features = data.features
        for i, feat in enumerate(features):
            print("%d - %s" % (i, feat))
        sys.exit(0)

    classifier = Sentiment()

    logger.info("Loading feature sets and training data...")
    X, y = data.get_samples()

    # TODO split data into k-fold samples
    X_train = X[:1900]
    y_train = y[:1900]
    X_test = X[1900:]
    y_test = y[1900:]

    if args.report:
        do_report(X_test, y_test)
        sys.exit(0)

    # building classifiers is time-consuming so only do this if we get here
    logger.info("Creating classifiers...")
    classifier.init_classifiers(X_train, y_train)
    logger.info("Done.")
