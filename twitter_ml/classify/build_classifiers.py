#! /usr/bin/env python3
"""Re-create classifiers based on training data."""
import argparse
import logging.config
import sys

import yaml

from twitter_ml.classify.sentiment import Sentiment
from twitter_ml.classify.movie_reviews import MovieReviews

with open("logging.yaml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Builds scikit-learn/nltk classifiers based on training data.')
    parser.add_argument("--features", action="store_true", default=False, help="list features and exit")
    args = parser.parse_args()

    data = MovieReviews()

    if args.features:
        print("Features:")
        features = data.features
        for i, f in enumerate(features):
            print("%d - %s" % (i, f))
        sys.exit(0)

    classifier = Sentiment()

    logger.info("Loading feature sets...")
    X, y = data.get_samples()

    # TODO split data into k-fold samples
    X_train = X[:1900]
    y_train = y[:1900]
    X_test = X[1900:]
    y_test = y[1900:]

    logger.info("Creating classifiers...")
    classifier.init_classifiers(X_train, X_test, y_train, y_test)

    logger.info("Done.")
