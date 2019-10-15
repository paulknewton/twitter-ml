[![pypi](https://img.shields.io/pypi/v/twitter_ml.svg)](https://pypi.python.org/pypi/twitter_ml)
[![Build Status](https://travis-ci.org/paulknewton/twitter_ml.svg?branch=master)](https://travis-ci.org/paulknewton/twitter_ml)
[![codecov](https://codecov.io/gh/paulknewton/twitter_ml/branch/master/graph/badge.svg)](https://codecov.io/gh/paulknewton/twitter_ml)
[![pyup](https://pyup.io/repos/github/paulknewton/twitter_ml/shield.svg)](https://pyup.io/account/repos/github/paulknewton/twitter_ml)
[![python3](https://pyup.io/repos/github/paulknewton/twitter_ml/python-3-shield.svg)](https://pyup.io/account/repos/github/paulknewton/twitter_ml)
[![readthedocs](https://readthedocs.org/projects/twitter_ml/badge/?version=latest)](https://twitter_ml.readthedocs.io/en/latest/?badge=latest)

[![DeepSource](https://static.deepsource.io/deepsource-badge-light.svg)](https://deepsource.io/gh/paulknewton/twitter_ml/?ref=repository-badge)

# Welcome to TwitterML
Project to analyse text streams (tweets or docs) using big data and machine learning. Uses Apache Spark to built textual metrics, then processes the text via various classification models to evaluate the sentiment (models via SciKit-Learn).

* Free software: GNU General Public License v3
* Documentation: https://twitter_ml.readthedocs.io

## Features
* Document Scanner - a standalone program that reads a text document and analyses it using NLTK and Spark.
* Twitter-Kafka Publisher - reads tweets from Twitter and pumps them into a Kafka server (where they can be consumed by out Twitter Consumer programs).
* Twitter Analyser - reads tweets from Kafka and performs analysis of the text using the Spark platform.
