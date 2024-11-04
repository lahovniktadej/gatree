Getting started
===============

This section demonstrates the usage of ``GATree`` for machine learning tasks.

Installation
------------

To install ``GATree`` with pip, use:

..  code:: bash

    pip install gatree

Usage
-----

The following example demonstrates how to perform classification of the iris dataset using ``GATree``.

..  code:: python

    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from gatree.methods.gatreeclassifier import GATreeClassifier

    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    # Create and fit the GATree classifier
    gatree = GATreeClassifier(n_jobs=16, random_state=10)
    gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

    # Make predictions on the testing set
    y_pred = gatree.predict(X_test)

    # Evaluate the accuracy of the classifier
    print(accuracy_score(y_test, y_pred))