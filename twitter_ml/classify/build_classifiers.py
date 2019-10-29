"""
Re-create classifiers based on training data.
"""
import argparse
import logging.config

import yaml

from twitter_ml.classify.sentiment import Sentiment
from twitter_ml.classify.movie_reviews import MovieReviews

with open("logging.yaml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Builds scikit-learn/nltk classifiers based on training data.')
    args = parser.parse_args()

    classifier = Sentiment()

    logger.info("Loading feature sets...")
    X, y = MovieReviews.create_all_feature_sets()

    # TODO split data into k-fold samples
    X_train = X[:1900]
    y_train = y[:1900]
    X_test = X[1900:]
    y_test = y[1900:]

    logger.info("Creating classifiers...")
    c = classifier.init_classifiers(X_train, X_test, y_train, y_test)

    logger.info("Done.")
