"""Wrapper around the NLTK movie view databas."""
import logging
import os
import pickle
import random
from typing import Tuple, List

import nltk
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from twitter_ml.classify.utils import Utils

logger = logging.getLogger(__name__)


class MovieReviews:
    """
    Wrapper around the movie review data shipped with the NLTK library.

    Used to generate test/training data by extracting features.
    """

    def __init__(self):
        """Instantiate the class and download the NLTK stopwords and review data."""
        if "NLTK_PROXY" in os.environ:
            logger.info("Using proxy %s", os.environ["NLTK_PROXY"])
            nltk.set_proxy(os.environ["NLTK_PROXY"])

        nltk.download("stopwords")
        from nltk.corpus import stopwords
        self._stopwords = stopwords.words('english')

        nltk.download('movie_reviews')
        from nltk.corpus import movie_reviews
        self._data = movie_reviews

        self._features = []  #  lazy load

    @property
    def stopwords(self) -> List[str]:
        """
        Get the NLTK stopwords used internally when identifying features.

        :return: the list of stopwords
        """
        return self._stopwords

    @property
    def reviews(self):
        """Get the moview review data."""
        return self._data

    @property
    def features(self) -> List[str]:
        """
        Get the list of features used to train the classifier.

        :return: the list of feature words
        """
        # TODO load from pickle file if available
        if self._features:
            return self._features

        # lazy load the most common words to use for building features
        all_words = []
        for w in tqdm(self.reviews.words(), desc="Feature identification"):
            w = w.lower()
            if w not in self.stopwords:
                all_words.append(w.lower())

        all_words_dist = nltk.FreqDist(all_words)
        # logger.debug("Frequency dist of 15 most common words:%s", all_words_dist.most_common(15))
        # logger.debug("Frequency of 'stupid':%d", all_words_dist["stupid"])
        # TODO tune word bag size
        self._features = list(all_words_dist.keys())[:3000]

        # save the words used in the feature list (used when classifying)
        feature_fn = "models/features.pickle"
        logger.debug("Saving feature list to %s...", feature_fn)
        with open(feature_fn, "wb") as features_f:
            pickle.dump(self._features, features_f)
        return self._features

    def get_samples(self) -> Tuple[List[int], List[int]]:
        """
        Create a (feature set, category) tuple for all movie reviews in the NLTK movie review dataset.

        :return: a tuple of features (matrix) and categories (vector)
        """
        logger.debug("Building data set...")

        # build list of words (1 list per doc) and pos/neg category
        documents = [(list(self.reviews.words(fileid)), category)
                     for category in self.reviews.categories()
                     for fileid in tqdm(self.reviews.fileids(category), desc="Review extraction")]
        random.shuffle(documents)

        # build a feature encoding and category for each review in the movie DB

        # encode the categories as integers via a LabelEncoder
        le = LabelEncoder()
        X, y = zip(*[(Utils.encode_features(self.features, review), category) for (review, category) in
                     tqdm(documents, desc="Feature encoding")])

        X = list(X)
        y = list(le.fit_transform(y))

        return X, y
