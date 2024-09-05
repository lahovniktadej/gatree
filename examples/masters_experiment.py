"""
The following example was used to perform 
the experiment for the master's thesis.

Before running this example, install the following packages:
poetry add tqdm
poetry add imblearn
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from gatree.gatree import GATree
from imblearn.datasets import fetch_datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def csv_preparation(CLASSIFIERS, results_dir, wine_quality_long, wine_quality_wide):
    """
    Prepare CSV files for storing the experimental results.

    Args:
        CLASSIFIERS (dict): Dictionary of classifiers.
        results_dir (str): Path to the directory for storing the results.
        wine_quality_long (str): Path to the long format CSV file.
        wine_quality_wide (str): Path to the wide format CSV.

    Returns:
        None
    """
    # Long format
    if not os.path.exists(os.path.join(results_dir, 'wine_quality_long.csv')):
        # Write header to the results file
        header = ['run', 'fold', 'algorithm', 'accuracy', 'accuracy_majority',
                  'accuracy_minority', 'f1_macro', 'f1_majority', 'f1_minority']
        with open(wine_quality_long, 'w') as f:
            f.write(','.join(header) + '\n')

    # Wide format
    if not os.path.exists(os.path.join(results_dir, 'wine_quality_wide.csv')):
        # Write header to the results file
        header = ['run', 'fold']
        for clf_name, _ in CLASSIFIERS.items():
            header.extend([f'{clf_name}_accuracy', f'{clf_name}_accuracy_majority', f'{clf_name}_accuracy_minority',
                          f'{clf_name}_f1_macro', f'{clf_name}_f1_majority', f'{clf_name}_f1_minority'])
        with open(wine_quality_wide, 'w') as f:
            f.write(','.join(header) + '\n')


def evaluate_classifier(run, fold, clf_name, clf_class, clf_params, X_train, X_test, y_train, y_test, fit_params):
    """
    Evaluate the classifier.

    Args:
        fold (int): Fold number.
        clf_name (str): Classifier name.
        clf_class (class): Classifier class.
        clf_params (dict): Classifier parameters.
        X_train (DataFrame): Training data.
        X_test (DataFrame): Testing data.
        y_train (Series): Training target.
        y_test (Series): Testing target.
        fit_params (dict): Fit parameters.

    Returns:
        dict: Dictionary of results.
    """
    # Model fitting
    clf = clf_class(**clf_params)
    clf.fit(X_train, y_train, **fit_params)

    # Evaluation
    y_pred = clf.predict(X_test)

    # Convert y_test and y_pred to numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    return {
        'run': run + 1,
        'fold': fold + 1,
        'algorithm': clf_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'accuracy_majority': accuracy_score(y_test[y_test == 0], y_pred[y_test == 0]),
        'accuracy_minority': accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_majority': f1_score(y_test[y_test == 0], y_pred[y_test == 0], zero_division=0),
        'f1_minority': f1_score(y_test[y_test == 1], y_pred[y_test == 1], zero_division=0)
    }


# Data preparation
wine_quality = fetch_datasets()['wine_quality']
df = pd.DataFrame(wine_quality.data)
df['target'] = wine_quality.target
# -1 majority (4715), 1 minority (183)

# Convert target values to 0 and 1
df['target'] = df['target'].apply(lambda x: 0 if x == -1 else 1)

# Configuration
RUNS = 3
FOLDS = 10
GENERATIONS = 500
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.2
ELITE_SIZE = 1
CLASSIFIERS = {
    'standard': {
        'class': GATree,
        'params': {
            'n_jobs': 16
        },
        'fit_params': {
            'algorithm': 'standard',
            'max_iter': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'mutation_probability': MUTATION_PROBABILITY,
            'elite_size': ELITE_SIZE
        }
    },
    'blocal': {
        'class': GATree,
        'params': {
            'n_jobs': 16
        },
        'fit_params': {
            'algorithm': 'balanced',
            'scope': 'local',
            'max_iter': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'mutation_probability': MUTATION_PROBABILITY,
            'elite_size': ELITE_SIZE
        }
    },
    'binherit': {
        'class': GATree,
        'params': {
            'n_jobs': 16
        },
        'fit_params': {
            'algorithm': 'balanced',
            'scope': 'inherit',
            'max_iter': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'mutation_probability': MUTATION_PROBABILITY,
            'elite_size': ELITE_SIZE
        }
    },
    'bglobal': {
        'class': GATree,
        'params': {
            'n_jobs': 16
        },
        'fit_params': {
            'algorithm': 'balanced',
            'scope': 'global',
            'max_iter': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'mutation_probability': MUTATION_PROBABILITY,
            'elite_size': ELITE_SIZE
        }
    }
}

# Experimental results
restart = False
results_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'results')
wine_quality_long = os.path.join(results_dir, 'wine_quality_long.csv')
wine_quality_wide = os.path.join(results_dir, 'wine_quality_wide.csv')
if input('Restart the experiment and discard existing results? [y/n]: ') == 'y':
    if os.path.exists(wine_quality_long):
        os.remove(wine_quality_long)
    if os.path.exists(wine_quality_wide):
        os.remove(wine_quality_wide)
else:
    if input('Continue the experiment from a specific checkpoint? [y/n]: ') == 'y':
        restart = True
        last_run = int(input('Run: ')) - 1
        last_fold = int(input('Fold: ')) - 1
        last_clf = int(
            input('Algorithm [1 standard, 2 local, 3 inherit, 4 global]: ')) - 1
csv_preparation(CLASSIFIERS, results_dir, wine_quality_long, wine_quality_wide)

# Runs
for run in tqdm(range(RUNS)):
    # Skip runs
    if restart and run < last_run:
        continue

    # Cross-validation
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)
    for fold, (train_index, test_index) in enumerate(tqdm(skf.split(df, df['target']), total=FOLDS)):
        # Skip folds
        if restart and fold < last_fold:
            continue

        # Split data
        train, test = df.iloc[train_index], df.iloc[test_index]
        X_train, y_train = train.drop('target', axis=1), train['target']
        X_test, y_test = test.drop('target', axis=1), test['target']

        # Iteration over classifiers
        for i, (clf_name, clf_info) in enumerate(CLASSIFIERS.items()):
            # Skip classifiers
            if restart and i < last_clf:
                continue
            restart = False

            clf_info['params']['random_state'] = run
            clf_results = evaluate_classifier(
                run,
                fold,
                clf_name,
                clf_info['class'],
                clf_info['params'],
                X_train,
                X_test,
                y_train,
                y_test,
                clf_info.get('fit_params')
            )

            # Export results to CSV files
            with open(wine_quality_long, 'a') as f:
                f.write(','.join(map(str, clf_results.values())) + '\n')
            if i == 0:
                with open(wine_quality_wide, 'a') as f:
                    f.write(str(clf_results['fold']) +
                            ',' + str(clf_results['run']) + ',')
            with open(wine_quality_wide, 'a') as f:
                f.write(
                    ','.join(map(str, list(clf_results.values())[3:])) + ',')
            if i == len(CLASSIFIERS) - 1:
                with open(wine_quality_wide, 'a') as f:
                    f.write('\n')
