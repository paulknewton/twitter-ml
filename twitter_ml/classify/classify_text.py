"""
Classify a text as +ve or -ve using classifiers.
"""
import argparse
import logging.config
import sys

import yaml

from twitter_ml.classify.sentiment import Sentiment

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies text sentiment based on scikit and NLTK models")
    parser.add_argument("--text", help="text to classify")
    parser.add_argument("--files", nargs="+", help="files to classify")
    args = parser.parse_args()

    classifier = Sentiment()

    if args.text:
        sentiment, confidence = classifier.classify_sentiment(args.text)
        print("Classification: %s; Confidence: %f" % (sentiment, confidence))

    elif args.files:
        for f in args.files:
            print("Processing %s" % f)
            print("---")
            with open(f, "rb") as file:
                sentiment, confidence = classifier.classify_sentiment(file.read())
                print("%s:Classification = %s; Confidence = %f" % (f, sentiment, confidence))

    else:
        print("Nothing to do. Exiting.")
        sys.exit(0)
