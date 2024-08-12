---
title: 'GATree: Evolutionary decision tree classifier in Python'
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
date: 06 March 2024
bibliography: paper.bib
---

# Summary

_GATree_ is a Python library that simplifies the way decision trees are constructed and optimised for classification machine learning tasks[^1]. Leveraging the principles of standard genetic algorithms, _GATree_ allows for the dynamic evolution of decision tree structures, providing a flexible and powerful tool for machine learning practitioners. Unlike traditional decision tree algorithms that follow a deterministic path based on statistical models or information theory, _GATree_ introduces an evolutionary process where selection, mutation, and crossover operations guide the development of optimised trees. This method enhances the adaptability and performance of decision trees and opens new possibilities for addressing complex classification problems. _GATree_ stands out as a user-friendly, highly customisable solution, enabling users to tailor fitness functions and algorithm parameters to meet specific project needs, whether in academic research or practical applications.

[^1]: _GATree_ is limited to classification tasks, with support for regression tasks planned for future releases.

# Overview

At the heart of _GATree_'s methodology lies the integration of genetic algorithms with decision tree construction, a process inspired by natural evolution [@koza1990concept]. This evolutionary approach begins with the random generation of an initial population of decision trees, each evaluated for their fitness[^2] in solving a given supervised task on the training data. Fitness evaluation typically considers factors such as classification accuracy and tree complexity, striving for a balance that rewards both the quality of decisions and the generalisability of the decisions [@bot2000application; @barros2012survey].

[^2]: Fitness is the estimation of the quality of the individual decision trees, which determines whether a decision tree survives into the next generation or not.

![Overview of the evolution process.\label{fig:ga}](./images/ga.png){ width=85% }

\newpage

Following the principles of natural selection, trees that perform better are more likely to contribute to the next generation, either through direct selection or by producing offspring via crossover and mutation operations. Crossover involves the exchange of genetic material (i.e., tree nodes or branches) between two parent trees, while mutation introduces random changes to a tree's structure, promoting genetic diversity within the population. This iterative process of selection, crossover, and mutation (presented in \autoref{fig:ga}) continues across generations, with the algorithm converging towards more effective decision tree solutions over time.

# Statement of need

The development of decision tree classifiers has long been a focal point in machine learning due to their interpretability and efficacy in various machine learning tasks. Traditional algorithms, however, often fall short when dealing with complex data structures or require extensive fine-tuning to avoid overfitting or underfitting. _GATree_ addresses these challenges by introducing an evolutionary approach to decision tree optimisation, allowing for a more nuanced exploration of the solution space than is possible with conventional methods [@RIVERALOPEZ2022101006; @karakativc2018building].

This evolutionary strategy ensures that _GATree_ can adaptively fine-tune decision trees, exploring a broader range of potential solutions and dynamically adjusting to achieve optimal performance. Such flexibility is precious in fields where classification tasks are complex, and data can exhibit varied and unpredictable patterns. Furthermore, _GATree_'s ability to customise fitness functions allows for incorporating domain-specific knowledge into the evolutionary process, enhancing the relevance and quality of the resulting decision trees.

Even though there are existing Python libraries that use various meta-heuristic approaches to form machine learning tree models (i.e., _gplearn_[^3], _tinyGP_[^4] and _TensorGP_ [@baeta2021tensorgp]), they use symbolic regression and not decision trees. In the broader context of machine learning and data mining, _GATree_ represents a significant advancement, offering a novel solution to the limitations of existing libraries. By integrating the principles of standard genetic algorithms with decision tree construction, _GATree_ not only enhances the adaptability and performance of these classifiers but also provides a rich platform for further research and development in evolutionary computing and its applications in machine learning.

[^3]: \url{https://github.com/trevorstephens/gplearn}
[^4]: \url{https://github.com/moshesipper/tiny_gp}

## Architecture

_GATree_ is a Python library with a modular and extensible architecture. The package is implemented using two classes: _GATree_ and _Node_. The _GATree_ class is responsible for the genetic algorithm by utilising operator classes, such as _Selection_ (with optional elitism), _Crossover_, and _Mutation_. The _Node_ class handles the decision tree structure and its operations, such as tree generation, tree evaluation, and class prediction.

The library is user-friendly and highly customisable - users can easily define custom fitness functions[^5] and other parameters to meet their needs. It is implemented to be compatible with the de-facto standard _scikit-learn_ machine learning library; thus, the main methods of use (i.e., _fit()_ and _predict()_) are present in _GATree_.

[^5]: The default fitness function is calculated as the combination of accuracy on the test set (preferring better/higher accuracy) and the tree size (preferring smaller, more generalisable trees).

## Usage and customisation

The following example shows how to perform classification of the _iris_ dataset using the _GATree_ library.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree import GATree

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

# Create and fit the GATree classifier
gatree = GATree(n_jobs=16, random_state=32)
gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
acc = accuracy_score(y_test, y_pred)
```

In this example, we load the iris dataset and split it into training and testing sets. Next, we create an instance of the _GATree_ classifier and define its parameters, such as the number of jobs to run in parallel and the random state for reproducibility. We then fit the classifier to the training data using a population size of 100 and a maximum of 100 iterations. Finally, we make predictions on the testing set and evaluate the accuracy of the classifier. The _GATree_ classifier uses a genetic algorithm to evolve and optimise the decision tree structure for the classification task. This configuration achieves an accuracy of 100% on the testing set, demonstrating the effectiveness of GATree for classification tasks.

![Average fitness value and best fitness value at each iteration of the genetic algorithm for the iris dataset.\label{fig:fitness_plot}](./images/fitness_value.png){ width=100% }

\newpage

\autoref{fig:fitness_plot} provides a comprehensive visualisation of the genetic algorithm's progress on the _iris_ dataset. The line graph on the left showcases the average fitness value[^6] of each decision tree in the population across iterations, offering insight into the algorithm's overall performance over time. We can observe the most significant improvement in the average fitness value in the first 50 iterations. We can see a slight decline in average fitness values after the 50th iteration, indicating getting stuck in the local optimum while building the decision trees. The slight variations in the final iterations indicate that the population is still changing due to crossover and mutation. However, the average quality of the decision trees in the population stays roughly the same. On the right half, a similar line graph displays the best fitness value[^7] at each iteration, providing a more detailed view of the algorithm's progress. The graph shows that the best fitness value improves rapidly in the first 30 iterations. The best decision tree is unaffected by evolving local optimums around the 70th iteration as the average decision tree does but remains near the global optimum, mainly due to the elitism operator.

[^6]: The average fitness is the actual average value of all the fitness values of the entire population.
[^7]: The best fitness is only the one fitness value - the one from the best individual in the population.

\autoref{fig:decision_tree} shows the final decision tree built with the _GATree_ classifier after fitting it to the _iris_ dataset.

![Final decision tree built with the GATree classifier.\label{fig:decision_tree}](./images/decision_tree.png){ width=80% }

The fitness function can be customised to suit the specific requirements of the classification task. For example, we can define a custom fitness function that considers the decision tree's size, penalising larger trees to encourage simplicity and interpretability. The following example demonstrates defining and using a custom fitness function with the _GATree_ classifier.

```python
# Custom fitness function
def fitness_function(root):
    return 1 - accuracy_score(root.y_true, root.y_pred) + (0.05 * root.size())

# Create and fit the GATree classifier
gatree = GATree(fitness_function=fitness_function, n_jobs=16, random_state=10)
gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)
```

\newpage

## Experiment

To test the performance of the _GATree_ classifier, we conducted a series of experiments on the _adult_ dataset. The _adult_ dataset contains 48.842 instances with 14 attributes (e.g., sex, age, native country, marital status, education, work class, occupation, etc.). The outcome variable is the income level, which is divided into two classes: <=50K and >50K (binary outcome, suitable for classification tasks). The dataset is imbalanced, with 76% of instances belonging to the <=50K class and 24% to the >50K class.

We evaluated the classifier's accuracy and F1-score across 100 runs with different parameter settings (see \autoref{tab:experiment_results}) and compared the results with other classifiers, such as _DecisionTreeClassifier_[^8] (scikit-learn) and _SymbolicClassifier_[^9] (gplearn). The _DecisionTreeClassifier_ is a standard decision tree classifier, while the _SymbolicClassifier_ is a symbolic regression classifier that uses genetic programming to evolve symbolic expressions. The code used to conduct the experiment is available in the _GATree_ repository[^10].

[^8]: \url{https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html}
[^9]: \url{https://gplearn.readthedocs.io/en/stable/reference.html\#symbolic-classifier}
[^10]: \url{https://github.com/lahovniktadej/gatree/blob/main/examples/joss_experiment.py}

The results demonstrate that _GATree_ achieves competitive performance in terms of accuracy and F1-score. _GATree_'s performance improves with more generations and higher population sizes, indicating the importance of these parameters in the evolutionary process.

:Results of the conducted experiment.\label{tab:experiment_results}

+------------------------+-----------------------------+----------------+----------------+
| Classifier             | Parameters                  | Avg. max\      | Avg. max\      |
|                        |                             | accuracy\      | F1-score\      |
|                        |                             | (95% CI)       | (95% CI)       |
+========================+=============================+================+:===============+
| GATree                 | mutation_probability=0.10\  | 0.800\         | 0.351\         |
|                        | population_size=25\         | (0.799, 0.801) | (0.341, 0.362) |
|                        | elite_size=1\               |                |                |
|                        | max_depth=5\                |                |                |
|                        | max_iter=50\                |                |                |
+------------------------+-----------------------------+----------------+----------------+
| GATree                 | mutation_probability=0.15\  | 0.807\         | 0.379\         |
|                        | population_size=50\         | (0.806, 0.809) | (0.368, 0.390) |
|                        | elite_size=2\               |                |                |
|                        | max_depth=5\                |                |                |
|                        | max_iter=100\               |                |                |
+------------------------+-----------------------------+----------------+----------------+
| GATree                 | mutation_probability=0.20\  | 0.810\         | 0.392\         |
|                        | population_size=50\         | (0.808, 0.811) | (0.382, 0.403) |
|                        | elite_size=5\               |                |                |
|                        | max_depth=5\                |                |                |
|                        | max_iter=200\               |                |                |
+------------------------+-----------------------------+----------------+----------------+
| DecisionTreeClassifier | criterion='gini'            | 0.806\         | 0.451\         |
|                        | splitter='random'\          | (0.804, 0.806) | (0.430, 0.473) |
|                        | max_depth=5\                |                |                |
+------------------------+-----------------------------+----------------+----------------+
| SymbolicClassifier     | parsimony_coefficient=0.01\ | 0.739\         | 0.034\         |
|                        | population_size=50\         | (0.722, 0.756) | (0.013, 0.055) |
|                        | generations=100\            |                |                |
|                        | init_depth=\(5, 5)          |                |                |
+------------------------+-----------------------------+----------------+----------------+

\newpage

# References
