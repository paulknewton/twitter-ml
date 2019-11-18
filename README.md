[![pypi](https://img.shields.io/pypi/v/twitter-ml.svg)](https://pypi.python.org/pypi/twitter-ml)
[![Build Status](https://travis-ci.org/paulknewton/twitter-ml.svg?branch=master)](https://travis-ci.org/paulknewton/twitter-ml)
[![codecov](https://codecov.io/gh/paulknewton/twitter-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/paulknewton/twitter-ml)
[![pyup](https://pyup.io/repos/github/paulknewton/twitter-ml/shield.svg)](https://pyup.io/account/repos/github/paulknewton/twitter-ml)
[![python3](https://pyup.io/repos/github/paulknewton/twitter-ml/python-3-shield.svg)](https://pyup.io/account/repos/github/paulknewton/twitter-ml)
[![Documentation Status](https://readthedocs.org/projects/twitter-ml/badge/?version=latest)](https://twitter-ml.readthedocs.io/en/latest/?badge=latest)

[![DeepSource](https://static.deepsource.io/deepsource-badge-light.svg)](https://deepsource.io/gh/paulknewton/twitter-ml/?ref=repository-badge)

# Welcome to TwitterML
Project to analyse text streams (tweets or docs) using big data and machine learning. Uses Apache Spark to built textual metrics, then processes the text via various classification models to evaluate the sentiment (models via SciKit-Learn).

* Free software: GNU General Public License v3
* Documentation: https://twitter-ml.readthedocs.io

![waffle](sample_waffle.png)

![wordcloud](wordcloud.png)

![learning_curve](learning_curve.png)

![roc_kfolds](roc_kfolds.png)

## Features
* Classifier Builder - standalone tool to configure classifiers and train them using pre-classified samples
* Text Classify - a standalone program for classifying the sentiment of text using NLTK and SciKit-Learn classifiers
* Document Scanner - a program for classifying text documents on the Spark platform
* Twitter-Kafka Publisher - reads tweets from Twitter and pumps them into a Kafka server (where they can be consumed by out Twitter Consumer programs).
* Twitter Analyser - reads tweets from Kafka and performs analysis of the text using the Spark platform.
