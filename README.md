<h1 align="center">
    GATree
</h1>

<p align="center">
    <img alt="PyPI version" src="https://img.shields.io/pypi/v/gatree.svg" />
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gatree.svg">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/niaaml.svg" href="https://pepy.tech/project/gatree">
    <img alt="Downloads" src="https://pepy.tech/badge/gatree">
    <img alt="GATree" src="https://github.com/lahovniktadej/gatree/actions/workflows/test.yml/badge.svg" />
    <img alt="Documentation status" src="https://readthedocs.org/projects/gatree/badge/?version=latest" />
</p>

<p align ="center">
    <img alt="Repository size" src="https://img.shields.io/github/repo-size/lahovniktadej/gatree" />
    <img alt="License" src="https://img.shields.io/github/license/lahovniktadej/gatree.svg" />
    <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/w/lahovniktadej/gatree.svg">
    <img alt="Percentage of issues still open" src="http://isitmaintained.com/badge/open/lukapecnik/niaaml.svg" href="http://isitmaintained.com/project/lahovniktadej/gatree">
    <img alt="Average time to resolve an issue" src="http://isitmaintained.com/badge/resolution/lukapecnik/niaaml.svg" href="http://isitmaintained.com/project/lahovniktadej/gatree">
    <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/lahovniktadej/gatree.svg"/>
</p>

<p align="center">
    <a href="#-about">📋 About </a>  •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-usage">🚀 Usage</a> •
    <a href="#-genetic-operators-in-gatree">🧬 Genetic Operators</a> •
    <a href="#-community-guidelines">🫂 Community Guidelines</a> •
    <a href="#-license">📜 License</a>
</p>

## 📋 About
GATree is a Python library designed for implementing evolutionary decision trees using a standard genetic algorithm approach. The library provides functionalities for selection, mutation, and crossover operations within the decision tree structure, allowing users to evolve and optimise decision trees for various classification tasks. 🌲🧬

The library's core objective is to empower users in creating and fine-tuning decision trees through an evolutionary process, opening avenues for innovative approaches to classification problems. GATree enables the dynamic growth and adaptation of decision trees, offering a flexible and powerful tool for machine learning enthusiasts and practitioners. 🚀🌿

GATree is currently limited to classification tasks, with support for regression tasks planned for future releases. 💡

* **Free software:** MIT license
* **Documentation**: [http://gatree.readthedocs.io](http://gatree.readthedocs.io)
* **Python**: 3.9, 3.10, 3.11, 3.12
* **Dependencies**: listed in [CONTRIBUTING.md](./CONTRIBUTING.md#dependencies)
* **Operating systems**: Windows, Ubuntu, macOS

## 📦 Installation
### pip
To install `GATree` using pip, run the following command:
```bash
pip install gatree
```

## 🚀 Usage
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

## 🧬 Genetic Operators in `GATree`
The genetic algorithm for decision trees in `GATree` involves several key operators: _selection_, _elitism_, _crossover_, and _mutation_. Each of these operators plays a crucial role in the evolution and optimisation of the decision trees. Below is a detailed description of each operator within the context of the `GATree` class.

### Selection
**Selection** is the process of choosing parent trees from the current population to produce offspring for the next generation. By default, `GATree` class uses tournament selection, a method where a subset of the population is randomly chosen, and the best individual from this subset is selected.

### Elitism
**Elitism** ensures that the best-performing individuals (trees) from the current generation are carried over to the next generation without any modification. This guarantees that the quality of the population does not decrease from one generation to the next.

### Crossover

**Crossover** is a genetic operator used to combine the genetic information of two parent trees to generate new offspring. This enables exploration, which helps in creating diversity in the population and combining good traits from both parents.

### Mutation
**Mutation** introduces random changes to a tree to maintain genetic diversity and explore new solutions. This helps in avoiding local optima by introducing new genetic structures.

## 🫂 Community Guidelines
### Contributing
To contribure to the software, please read the [contributing guidelines](./CONTRIBUTING.md).

### Reporting Issues
If you encounter any issues with the library, please report them using the [issue tracker](https://github.com/lahovniktadej/gatree/issues). Include a detailed description of the problem, including the steps to reproduce the problem, the stack trace, and details about your operating system and software version.

### Seeking Support
If you need support, please first refer to the [documentation](http://gatree.readthedocs.io). If you still require assistance, please open an issue on the [issue tracker](https://github.com/lahovniktadej/gatree/issues) with the `question` tag. For private inquiries, you can contact us via e-mail at [tadej.lahovnik1@um.si](mailto:tadej.lahovnik1@um.si) or [saso.karakatic@um.si](mailto:saso.karakatic@um.si).

## 📜 License
This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!