import argparse
import logging.config

import yaml

import twitter_ml.classify.sentiment as sentiment

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifies text sentiment based on scikit and NLTK models.')
    parser.add_argument(dest='text', help='text to classify')
    args = parser.parse_args()

    sentiment, confidence = sentiment.classify_sentiment(args.text)
    print("%s\nClassification: %s with Confidence: %f" % (args.text, sentiment, confidence))
