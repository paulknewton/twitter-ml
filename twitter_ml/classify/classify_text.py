"""
Classify a text as +ve or -ve using classifiers.
"""
import argparse
import logging.config
import sys
from collections import Counter

import yaml

from twitter_ml.classify.sentiment import Sentiment

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies text sentiment based on scikit and NLTK models")
    parser.add_argument("--text", nargs="+", help="text to classify")
    parser.add_argument("--files", nargs="+", help="files to classify")
    parser.add_argument("--classifier", nargs=1,
                        help="name of the specific classifier to use (default: a voting classifier")
    parser.add_argument("--waffle", action="store_true", default=False)
    args = parser.parse_args()

    classifier = Sentiment()
    results = []

    if args.text:
        for t in args.text:
            sentiment, confidence = classifier.classify_sentiment(t, args.classifier)
            print("Classification: %s; Confidence: %f" % (sentiment, confidence))
            results.append(sentiment)

    elif args.files:
        for f in args.files:
            print("Processing %s" % f)
            print("---")
            with open(f, "r") as file:
                sentiment, confidence = classifier.classify_sentiment(file.read(), args.classifier)
                print("%s:Classification = %s; Confidence = %f" % (f, sentiment, confidence))
                results.append(sentiment)

    else:
        print("Nothing to do. Exiting.")
        sys.exit(0)

    # plot the results
    import matplotlib.pyplot as plt
    from pywaffle import Waffle

    fig = plt.figure(
        FigureClass=Waffle,
        rows=5,
        columns=10,
        values=Counter(results),
        figsize=(5, 3)
    )
    plt.show()
