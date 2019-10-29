import logging
import os
import pickle
import random
from typing import Any, Tuple, List, Dict

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

    @staticmethod
    def get_stopwords() -> List[str]:
        # get the moview reviews
        if "NLTK_PROXY" in os.environ:
            logger.debug("Using proxy %s", os.environ["NLTK_PROXY"])
            nltk.set_proxy(os.environ["NLTK_PROXY"])
        nltk.download('movie_reviews')
        from nltk.corpus import stopwords

        x = stopwords.words('english')
        return x

    @staticmethod
    def create_all_feature_sets() -> Tuple[List[int], List[int]]:
        """
        Create a (feature set, category) tuple for all movie reviews in the NLTK movie review dataset
        :return: a tuple of features and categories
        """

        # get the moview reviews
        if "NLTK_PROXY" in os.environ:
            logger.debug("Using proxy %s", os.environ["NLTK_PROXY"])
            nltk.set_proxy(os.environ["NLTK_PROXY"])
        nltk.download('movie_reviews')
        from nltk.corpus import movie_reviews

        logger.debug("Building data set...")

        # build list of words (1 list per doc) and pos/neg category
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in tqdm(movie_reviews.fileids(category), desc="Review extraction")]
        random.shuffle(documents)

        # extract the most common words to use for building features
        all_words = []
        stopwords = MovieReviews.get_stopwords()
        for w in tqdm(movie_reviews.words(), desc="Feature identification"):
            w = w.lower()
            if w not in stopwords:
                all_words.append(w.lower())

        all_words_dist = nltk.FreqDist(all_words)
        # logger.debug("Frequency dist of 15 most common words:%s", all_words_dist.most_common(15))
        # logger.debug("Frequency of 'stupid':%d", all_words_dist["stupid"])
        # TODO tune word bag size
        features = list(all_words_dist.keys())[:3000]

        # save the words used in the feature list (used when classifying)
        feature_fn = "models/features.pickle"
        logger.debug("Saving feature list to %s...", feature_fn)
        with open(feature_fn, "wb") as features_f:
            pickle.dump(features, features_f)

        # build a feature encoding and category for each review in the movie DB
        le = LabelEncoder()
        X, y = zip(*[(Utils.encode_features(features, review), category) for (review, category) in
                     tqdm(documents, desc="Feature encoding")])

        X = list(X)
        y = list(le.fit_transform(y))

        return X, y
