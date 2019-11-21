"""Test the movie review wrapper."""
import string

from twitter_ml.data.movie_reviews import MovieReviews


def test_features():
    """Test that features can be retrieved from the movie reviews."""
    data = MovieReviews(3000)
    assert len(data.features) == 3000


def test_features_no_punctuation():
    """Test that feature identification ignores punctuation."""
    data = MovieReviews(3000)
    punctuation = set(string.punctuation)

    common_words = punctuation & set(data.features)

    assert not common_words
