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
    feature_sets = MovieReviews.create_all_feature_sets()

    # TODO split data into 3 samples
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    logger.info("Creating classifiers...")
    classifier.rebuild_classifiers(training_set, testing_set)

    logger.info("Done.")
