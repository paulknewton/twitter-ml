#! /usr/bin/env python3
"""Re-create classifiers based on training data."""
import argparse
import logging.config
import math
import sys
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm
from twitter_ml.classify.movie_reviews import MovieReviews
from twitter_ml.classify.sentiment import Sentiment
from twitter_ml.classify.utils import Utils

with open("logging.yaml", "rt") as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)


def do_graphs(classifiers: List[Tuple[str, Any]], X, y):
    """
    Command method: output graphs.

    :param classifiers: list of classifiers
    :param X: test data
    :param y: test categories
    """
    if len(classifiers) < 1:
        raise ValueError("Must provide at least 1 classifier")

    # plot confusion matrices in a grid
    logger.info("Calculating confusion matrices with %d samples", len(X_test))
    subplots_cols = 4
    subplots_rows = math.ceil(len(classifiers) / subplots_cols)
    logger.debug("Plot dimensions: %d x %d", subplots_rows, subplots_cols)

    fig, axs = plt.subplots(subplots_rows, subplots_cols, figsize=(15, 8))
    graph_data = list(zip(axs.flat, classifiers))
    for ax, (label, clf) in graph_data:
        y_pred = clf.predict(X)

        Utils.plot_confusion_matrix(y, y_pred, unique_labels(y), label, ax)
    plt.show()

    # plot ROC curve and calculate AUC
    logging.info("Calculating ROC curves for %d sanples", len(X))
    for label, clf in classifiers:
        y_pred = clf.predict(X)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label="ROC %s (AUC = %0.2f)" % (label, roc_auc))
        plt.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="_Chance",
        )
    plt.title("Receiver Operating Characteristics")
    plt.legend()
    plt.show()


def do_report(classifiers: List[Tuple[str, Any]], X, y):
    """
    Print key matrics for a set of test data.

    :param classifiers: list of label, classifier tuples
    :param X: a matrix of test samples
    :param y: a vector of categories
    """
    logger.info("Samples: len(X, y) = %d, %d" % (len(X), len(y)))
    unique, counts = np.unique(np.array(y), return_counts=True)
    logger.info("Categories: " + str(list(zip(unique, counts))))

    for label, clf in classifiers:
        _dump_metrics(label, clf, X, y)


def do_learning_curve(classifiers: List[Tuple[str, Any]], X, y):
    """
    Plot learning curves for varying sample sizes.

    :param classifiers: list of label, classifier tuples
    :param X: a matrix of samples. Note this is both training data and test data.
    :param y: a vector of categories
    """
    fig, ax = plt.subplots()
    logging.info("Calculating learning curves over 0-%d samples", len(X))
    for label, clf in tqdm(classifiers, desc="Calculating learning curves"):
        train_sizes = np.linspace(0.05, 1, 50)
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, train_sizes=train_sizes, cv=5
        )

        # calculate the average across CV cycle results
        train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        ax.plot(train_sizes, train_mean, marker="o")  # no label
        ax.plot(train_sizes, test_mean, marker="x", label=label)
        # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD"
        )

    ax.set_xlabel("Training samples")
    ax.set_ylabel("Score")
    ax.set_title("Learning curve for classifiers")
    ax.legend()
    plt.show()


def do_roc_k_fold(label, clf, X, y, k):
    """
    Plot ROC curves for varying sample sizes using k-fold cross validation.

    :param label: name of the classifier
    :param clf: classifier to plot
    :param X: a matrix of samples. Note this is both training data and test data.
    :param y: a vector of categories
    :param k: number of folds to evaluate
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []  # TPR results (interpolated)
    aucs = []  # AUC results

    cv = StratifiedKFold(n_splits=k)
    fold_cnt = 0
    for train_index, test_index in cv.split(X, y):
        fold_cnt += 1
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # interpolate tpr values across 100 measurements
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        # tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # plot each ROC curve
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label="ROC %s - fold %d (AUC = %0.2f)" % (label, fold_cnt, roc_auc),
        )

    # plot random classifier ("chance")
    plt.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="_Chance",  # no label
    )

    # calculate average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # plot std dev either side of the mean tpr curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.title("ROC curve with various k-folds")
    plt.legend()
    plt.show()


def _dump_metrics(label, clf, X, y):
    print("-----------------\nSUMMARY FOR CLASSIFIER: %s" % label)
    y_pred = clf.predict(X)
    print("Metrics:\n" + Utils.get_classification_metrics(y, y_pred))
    print("Confusion matrix:")
    print(Utils.get_confusion_matrix(y, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds scikit-learn/nltk classifiers based on training data."
    )
    parser.add_argument(
        "--features", action="store_true", default=False, help="list features and exit"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="print classifier/sub-classifier metrics",
    )
    parser.add_argument(
        "--graphs", action="store_true", default=False, help="plot classifier graphs",
    )
    parser.add_argument(
        "--learning",
        action="store_true",
        default=False,
        help="plot classifier learning curves",
    )
    parser.add_argument(
        "--roc-kfold",
        help="plot ROC curves with specified k-fold value",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-k", help="value to use for k-folding data (default 5)", default=5
    )
    args = parser.parse_args()
    logger.info(args)

    data = MovieReviews(3000)

    if args.features:
        print("Features:")
        features = data.features
        for i, feat in enumerate(features):
            print("%d - %s" % (i, feat))
        sys.exit(0)

    sentiment = Sentiment("voting.yaml")

    logger.info("Loading feature sets and training data...")
    X, y = data.get_samples()

    # split data into k-fold samples (StratifiedKFold will balance the pos/neg samples)
    k_fold = int(args.k)
    skf = StratifiedKFold(n_splits=k_fold)
    train_index, test_index = next(skf.split(X, y))
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    if args.report:
        classifiers = [("voting", sentiment.voting_classifier)] + list(
            sentiment.voting_classifier.sub_classifiers.items()
        )
        do_report(classifiers, X_test, y_test)
        sys.exit(0)

    if args.graphs:
        do_graphs(
            [("voting", sentiment.voting_classifier)]
            + list(sentiment.voting_classifier.sub_classifiers.items()),
            X_test,
            y_test,
        )

    if args.roc_kfold:
        label, clf = (
            "voting",
            sentiment.voting_classifier,
        )  # list(sentiment.voting_classifier.sub_classifiers.items())[0]
        do_roc_k_fold(label, clf, X, y, k_fold)

    if args.learning:
        classifiers = list(sentiment.voting_classifier.sub_classifiers.items())
        do_learning_curve(classifiers, X, y)

    # building classifiers is time-consuming so only do this if we get here
    logger.info("Creating classifiers...")
    sentiment.init_classifiers(X_train, y_train)
    logger.info("Done.")
