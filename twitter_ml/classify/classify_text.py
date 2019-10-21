#! /usr/bin/env python3
"""
Classify a text as +ve or -ve using classifiers.
"""
import argparse
import logging.config
import sys
from collections import Counter

import matplotlib.pyplot as plt
import yaml
from pywaffle import Waffle

from twitter_ml.classify.sentiment import Sentiment
from wordcloud import WordCloud

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies text sentiment based on scikit and NLTK models")
    parser.add_argument("--text", nargs="+", help="text to classify")
    parser.add_argument("--files", nargs="+", help="files to classify")
    parser.add_argument("--classifier", help="name of the specific classifier to use (default: a voting classifier")
    parser.add_argument("--waffle", action="store_true", default=False, help="create a waffle picture of the results")
    parser.add_argument("--wordcloud", action="store_true", default=False, help="create a wordcloud of the text")
    parser.add_argument("--list", action="store_true", default=False, help="list the individual sub-classifers")
    args = parser.parse_args()

    classifier = Sentiment()
    results = []


    if args.list:
        print("Available classifiers:")
        for label, c in classifier.sub_classifiers.items():
            print("- %s: %s" % (label, c[1]))

        sys.exit(0)

    if args.text:
        for t in args.text:
            sentiment, confidence = classifier.classify_sentiment(t, args.classifier)
            print("Classification: %s; Confidence: %f" % (sentiment, confidence))
            results.append(sentiment)
        if args.wordcloud:
            all_words = " ".join(args.text)

    elif args.files:
        all_words = ""
        for f in args.files:
            print("Processing %s" % f)
            print("---")
            with open(str(f), "rb") as file:    # read as bytes in case there are any unusual chars
                text = file.read().decode(errors="ignore")  # ignore chars that cannot be converted to UTF

                if args.wordcloud:
                    all_words += text   # store the text for a wordcloud later

                sentiment, confidence = classifier.classify_sentiment(text, args.classifier)
                print("%s:Classification = %s; Confidence = %f" % (f, sentiment, confidence))
                results.append(sentiment)

    else:
        print("Nothing to do. Exiting.")
        sys.exit(0)

    # plot the results
    if args.waffle:
        fig = plt.figure(
            FigureClass=Waffle,
            rows=10,
            columns=20,
            values=Counter(results),
            figsize=(5, 3)
        )

        plt.savefig("waffle.png", bbox_inches='tight')
        plt.show()

    if args.wordcloud:
        logger.debug(all_words)
        wordcloud = WordCloud(stopwords={}).generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
