"""Test the movie review wrapper."""
from twitter_ml.classify.movie_reviews import MovieReviews


def test_features():
    """Test that features can be retrieved from the movie reviews."""
    data = MovieReviews()
    assert len(data.features) == 3000
