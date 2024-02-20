---
title: 'GATree: A Python package for an evolutionary decision tree classifier'
tags:
  - Python
  - genetic algorithm
  - evolutionary algorithm
  - classifier
  - machine learning
authors:
  - name: Tadej Lahovnik
    orcid: 0009-0005-9689-2991
    equal-contrib: true
    affiliation: 1
  - name: Sašo Karakatič
    orcid: 0000-0003-4441-9690
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: University of Maribor, Maribor, Slovenia
    index: 1
date: XX February 2024
bibliography: paper.bib
---

# Summary



# Statement of need

GATree, an evolutionary decision tree classifier, is a Python library with a modular and extensible architecture comprised of two classes: _GATree_ and _Node_. The _GATree_ class is responsible for the genetic algorithm by utilising operator classes, such as _Selection_, _Crossover_, and _Mutation_. The _Node_ class handles the decision tree structure and its operations. The library is user-friendly and highly customisable - users can easily define custom fitness functions and other parameters to meet their needs. While primarily intended for classification tasks, GATree can also perform regression tasks by modifying the fitness function.

The following example shows how to perform classification of the iris dataset using the GATree package. The iris dataset is a well-known dataset in the machine learning community, often used for testing and benchmarking classification algorithms. The dataset consists of 150 samples of iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The samples belong to one of the three classes: setosa, versicolor, and virginica. The goal is to classify the samples into the correct class based on the four features.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree import GATree

# Load the iris dataset as DataFrame
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

In this example, we load the iris dataset and split it into training and testing sets. Next, we create an instance of the GATree classifier and define its parameters, such as the number of jobs to run in parallel and the random state for reproducibility. We then fit the classifier to the training data using a population size of 100 and a maximum of 100 iterations. Finally, we make predictions on the testing set and evaluate the accuracy of the classifier. The GATree classifier uses a genetic algorithm to evolve and optimise the decision tree structure for the classification task. This configuration achieves an accuracy of 0.93 on the testing set, demonstrating the effectiveness of GATree for classification tasks.

\autoref{fig:fitness_plot} depicts the average fitness value at each iteration of the genetic algorithm for the iris dataset, demonstrating how the algorithm converges towards an optimal solution.

![Average fitness value at each iteration of the genetic algorithm for the iris dataset.\label{fig:fitness_plot}](./images/fitness_value.png){ width=60% }

\autoref{fig:decision_tree} shows the final decision tree obtained by the GATree classifier after fitting it to the iris dataset.

![Final decision tree obtained by the GATree classifier.\label{fig:decision_tree}](./images/decision_tree.png){ width=60% }

The fitness function can be customised to suit the specific requirements of the classification task. For example, we can define a custom fitness function that takes into account the size of the decision tree, penalising larger trees to encourage simplicity and interpretability. The following example demonstrates how to define and use a custom fitness function with the GATree classifier.

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

# Custom fitness function
def fitness_function(root):
    return accuracy_score(root.y_true, root.y_pred) - 0.05 * root.size()

# Create and fit the GATree classifier
gatree = GATree(fitness_function=fitness_function, n_jobs=16, random_state=123)
gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
print(accuracy_score(y_test, y_pred))
```

# References
