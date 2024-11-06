"""
The following example shows how to perform classification
of the iris dataset using GATree

Before running this example, install matplotlib:
poetry add matplotlib
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from gatree.methods.gatreeclustering import GATreeClustering

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

min_clusters = 5

fitness_function_kwargs = {
    'min_clusters': min_clusters,
    'fitness_X': X_train
}

# Create and fit the GATree clustering
gatree = GATreeClustering(n_jobs=16, random_state=32, min_clusters=min_clusters)
gatree.fit(X=X_train, population_size=100, max_iter=100, fitness_function_kwargs=fitness_function_kwargs)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
print(silhouette_score(X_test, y_pred))

# Plot fitness values over iterations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(gatree._avg_fitness, linestyle='-')
ax1.set_title('Average fitness values over iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Average fitness')
ax1.grid(True)
ax2.plot(gatree._best_fitness, linestyle='-', color='green')
ax2.set_title('Best fitness values over iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Best fitness')
ax2.grid(True)
plt.tight_layout()
plt.show()

gatree.plot()
