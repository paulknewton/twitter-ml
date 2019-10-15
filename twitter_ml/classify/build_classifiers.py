import argparse
import logging.config

import yaml

import twitter_ml.classify.sentiment as sentiment

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Builds scikit-learn/nltk classifiers based on training data.')
    args = parser.parse_args()

    logger.info("Loading feature sets...")
    feature_sets = sentiment.create_feature_sets()
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    logger.info("Creating classifiers...")
    classifiers = sentiment.create_classifiers(training_set, testing_set)

    logger.info("Done.")
