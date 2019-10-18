from typing import List, Dict


class Utils:
    @staticmethod
    def get_feature_vector(word_features: List[str], document: List[str]) -> Dict[str, bool]:
        """
        Generate a features set for a given feature list and word list.
        :param word_features: the list of words that will make up the feature set
        :param document: the text to check (as a list of words)
        :return: a feature set for the document
        """
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features
