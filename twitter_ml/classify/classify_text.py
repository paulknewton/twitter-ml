#! /usr/bin/env python3
"""
Classify a text as +ve or -ve using classifiers.
"""
import argparse
import logging.config
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from PIL import Image
from bs4 import BeautifulSoup
from pywaffle import Waffle
from wordcloud import WordCloud

from twitter_ml.classify.sentiment import Sentiment

with open("logging.yaml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)


def print_feature_encoding(feature_list, feature_encoding):
    print("Feature encoding:")

    # zip feature, encoding then sort or filter by encoding values
    # for feature, encoding in sorted(zip(feature_list, feature_encoding), key=lambda tup: tup[1]):
    for feature, encoding in filter(lambda tup: tup[1] == 1, zip(feature_list, feature_encoding)):
        print("%s: %s" % (feature, encoding))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies text sentiment based on scikit and NLTK models")
    parser.add_argument("--text", nargs="+", help="text to classify")
    parser.add_argument("--files", nargs="+", help="files to classify")
    parser.add_argument("--classifier", help="name of the specific classifier to use (default: a voting classifier")
    parser.add_argument("--waffle", action="store_true", default=False, help="create a waffle picture of the results")
    parser.add_argument("--wordcloud", action="store_true", default=False, help="create a wordcloud of the text")
    parser.add_argument("--list", action="store_true", default=False, help="list the individual sub-classifers")
    parser.add_argument("--features", action="store_true", default=False, help="list features")
    args = parser.parse_args()

    sentiment = Sentiment()
    results = []

    if args.list:
        print("Available classifiers:")
        for label, c in sentiment.sub_classifiers.items():
            print("- %s: %s" % (label, c[1]))

        sys.exit(0)

    if args.text:
        for t in args.text:
            feature_list, feature_encoding, category = sentiment.classify_sentiment(t, args.classifier)

            if args.features:
                print_feature_encoding(feature_list, feature_encoding)
            print("Classification: %s" % category)
            results.append(category)

        if args.wordcloud:
            all_words = " ".join(args.text)

    elif args.files:
        all_words = ""
        for url in args.files:
            print("Processing %s" % url)
            print("---")

            # URLs via http/https
            if url.lower().startswith("http") or url.lower().startswith("https"):
                with requests.get(url) as resp:
                    soup = BeautifulSoup(resp.text, features="html.parser")

                    # kill all script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()  # rip it out

                    # get text
                    text = soup.get_text()
            # files
            else:
                with open(str(url), "rb") as file:  # read as bytes in case there are any unusual chars
                    text = file.read().decode(errors="ignore")  # ignore chars that cannot be converted to UTF

            logger.info(text)
            if args.wordcloud:
                all_words += text  # store the text for a wordcloud later

            feature_list, feature_encoding, category = sentiment.classify_sentiment(text, args.classifier)

            if args.features:
                print_feature_encoding(feature_list, feature_encoding)
            print("%s:Classification = %s" % (url, category))
            results.append(category)

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
        mask = np.array(Image.open("wordcloud_mask.png"))
        logger.debug(all_words)

        # choose the stop words here
        stopwords = None
        # stopwords={}

        wordcloud = WordCloud(background_color="white", stopwords=stopwords, contour_width=3, contour_color="firebrick",
                              mask=mask).generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
