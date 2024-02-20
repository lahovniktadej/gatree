# GATree
[![PyPI version](https://img.shields.io/pypi/v/gatree.svg)](https://img.shields.io/pypi/v/gatree.svg)
[![GATree](https://github.com/lahovniktadej/gatree/actions/workflows/test.yml/badge.svg)](https://github.com/lahovniktadej/gatree/actions/workflows/test.yml)
[![Documentation status](https://readthedocs.org/projects/gatree/badge/?version=latest)](https://gatree.readthedocs.io/en/latest/?badge=latest)
![Open issues](https://isitmaintained.com/badge/open/lahovniktadej/gatree.svg)
![Repository size](https://img.shields.io/github/repo-size/lahovniktadej/gatree)
![License](https://img.shields.io/github/license/lahovniktadej/gatree.svg)

* **Free software:** MIT license
* **Documentation**: [http://gatree.readthedocs.io](http://gatree.readthedocs.io)
* **Python**: 3.9.x, 3.10.x
* **Operating systems**: Windows, Ubuntu, macOS

## About 📋
GATree is a Python library designed for implementing evolutionary decision trees using a genetic algorithm approach. The library provides functionalities for selection, mutation, and crossover operations within the decision tree structure, allowing users to evolve and optimise decision trees for various classification tasks. 🌲🧬

The library's core objective is to empower users in creating and fine-tuning decision trees through an evolutionary process, opening avenues for innovative approaches to classification problems. GATree enables the dynamic growth and adaptation of decision trees, offering a flexible and powerful tool for machine learning enthusiasts and practitioners. 🚀🌿

## Installation 📦
### pip
To install GATree using pip, run the following command:
```bash
pip install gatree
```

## Usage 🚀
The following example demonstrates how to perform classification of the iris dataset using `GATree`. More examples can be found in the [examples](./examples) directory.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree.gatree import GATree

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Create and fit the GATree classifier
gatree = GATree(n_jobs=16, random_state=123)
gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
print(accuracy_score(y_test, y_pred))
```

## License
This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!