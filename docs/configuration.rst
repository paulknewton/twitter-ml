=============
Configuration
=============

Voting Sub-classifiers
----------------------
The main classifier is a Voting classifier that uses a series of sub-classifiers to classify the text, then chooses the most-popular result.
The sub-classifiers used inside the Voting classifier are defined in the ``voting.yaml`` file:

.. code-block:: console

    voting:
        multinomilnb:
            module: sklearn.naive_bayes
            class: MultinomialNB
            description: Multinomial NB classifier from SciKit
        bernouillinb:
            module: sklearn.naive_bayes
            class: BernoulliNB
            description: Bernouilli NB classifier from SciKit
        logisticregression:
            module: sklearn.linear_model
            class: LogisticRegression
            description: Logistic Regression classifier from SciKit
        sgd:
            module: sklearn.linear_model
            class: SGDClassifier
            description: SGD classifier from SciKit
        linearrsvc:
            module: sklearn.svm
            class: LinearSVC
            description: Linear SVC classifier from SciKit

Note how there is a single ``voting`` section containing a list (actually a dictionary) of sub-classifiers.
Each entry is of the form:

.. code-block:: console

    some label:
        module: the python module containing the classifer class
        class: the class name
        description: some textual description
