import logging
import os
import pickle
import random
from typing import Any, Tuple, List, Dict

import nltk

from twitter_ml.classify.utils import Utils

logger = logging.getLogger(__name__)


class MovieReviews:
    """
    Wrapper around the movie review data shipped with the NLTK library.
    Used to generate test/training data by extracting features.
    """

    @staticmethod
    def create_all_feature_sets() -> List[Tuple[Dict[str, bool], Any]]:
        """
        Create a (feature set, category) tuple for all movie reviews in the NLTK movie review dataset
        :return: a list of feature, category tuples
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
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)

        # extract the most common words to use for building features
        all_words = []
        for w in movie_reviews.words():
            all_words.append(w.lower())
        all_words_dist = nltk.FreqDist(all_words)
        # logger.debug("Frequency dist of 15 most common words:%s", all_words_dist.most_common(15))
        # logger.debug("Frequency of 'stupid':%d", all_words_dist["stupid"])
        features = list(all_words_dist.keys())[:3000]

        # save the words used in the feature list (used when classifying)
        feature_fn = "models/features.pickle"
        logger.debug("Saving feature list to %s...", feature_fn)
        with open(feature_fn, "wb") as features_f:
            pickle.dump(features, features_f)

        # build list of feature, category profiles for each review in the movie DB
        return [(Utils.get_feature_vector(features, review), category) for (review, category) in documents]
