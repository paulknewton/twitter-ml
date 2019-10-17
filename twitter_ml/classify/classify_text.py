"""
Classify a text as +ve or -ve using classifiers.
"""
import argparse
import logging.config

import yaml

from twitter_ml.classify.sentiment import Sentiment

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifies text sentiment based on scikit and NLTK models.')
    parser.add_argument(dest='text', help='text to classify')
    args = parser.parse_args()

    classifier = Sentiment()
    sentiment, confidence = classifier.classify_sentiment(args.text)
    print("%s\nClassification: %s with Confidence: %f" % (args.text, sentiment, confidence))
