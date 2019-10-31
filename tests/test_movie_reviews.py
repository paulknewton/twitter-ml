from twitter_ml.classify.movie_reviews import MovieReviews

def test_features():
    """
    Check features can be retrieved from the movie reviews
    """
    data = MovieReviews()
    assert len(data.features) == 3000
