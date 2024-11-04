"""
Experiment to compare the performance of the GATree
classifier with competing classifiers on the adult dataset.

Before running this example, install the following libraries:
poetry add aif360
poetry add gplearn
"""
import datetime
import pandas as pd
import numpy as np
from aif360.sklearn.datasets import fetch_adult
from gplearn.genetic import SymbolicClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from gatree.methods.gatreeclassifier import GATreeClassifier

# Load the adult dataset
adult = fetch_adult(numeric_only=True)
X = adult.X
y = adult.y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)

# Evaluate a classifier
def evaluate_classifier(clf_class, clf_params, X_train, X_test, y_train, y_test, fit_params=None, n_iter=100):
    results = []

    for i in range(1, n_iter + 1):
        clf = clf_class(**clf_params, random_state=i)
        if fit_params is None:
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, **fit_params)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({'classifier': clf_class.__name__,
                       'run': i, 'accuracy': accuracy, 'f1': f1})

    avg_accuracy = np.mean([result['accuracy'] for result in results])
    avg_f1 = np.mean([result['f1'] for result in results])

    print(datetime.datetime.now())
    print(clf_class.__name__)
    print('Average accuracy: ', avg_accuracy)
    print('Average F1 score: ', avg_f1)
    print('')

    return results

# Dictionary of classifiers and their parameters
classifiers = {
    'GATree (configuration 1)': {
        'class': GATreeClassifier,
        'params': {
            'max_depth': 5,
            'n_jobs': 16
        },
        'fit_params': {
            'mutation_probability': 0.10,
            'population_size': 25,
            'elite_size': 1,
            'max_iter': 50
        }
    },
    'GATree (configuration 2)': {
        'class': GATreeClassifier,
        'params': {
            'max_depth': 5,
            'n_jobs': 16
        },
        'fit_params': {
            'mutation_probability': 0.15,
            'population_size': 50,
            'elite_size': 2,
            'max_iter': 100
        }
    },
    'GATree (configuration 3)': {
        'class': GATreeClassifier,
        'params': {
            'max_depth': 5,
            'n_jobs': 16
        },
        'fit_params': {
            'mutation_probability': 0.20,
            'population_size': 50,
            'elite_size': 5,
            'max_iter': 200
        }
    },
    'DecisionTreeClassifier': {
        'class': DecisionTreeClassifier,
        'params': {
            'criterion': 'gini',
            'splitter': 'random',
            'max_depth': 5
        }
    },
    'SymbolicClassifier': {
        'class': SymbolicClassifier,
        'params': {
            'parsimony_coefficient': 0.01,
            'population_size': 50,
            'generations': 50,
            'init_depth': (5, 5)
        }
    }
}

# Evaluate each classifier
results = []
for clf_name, clf_info in classifiers.items():
    clf_results = evaluate_classifier(
        clf_info['class'], clf_info['params'], X_train, X_test, y_train, y_test, clf_info.get('fit_params'))
    results.extend(clf_results)

# Extract results to CSV
pd.DataFrame(results).to_csv('joss_experiment.csv', index=False)
