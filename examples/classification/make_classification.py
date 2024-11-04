"""
The following example shows how to perform classification 
on a randomly generated dataset using GATree
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree.methods.gatreeclassifier import GATreeClassifier

# Generate a random dataset
data = make_classification(
    n_samples=1500, n_features=10, n_classes=2, random_state=32)
X = pd.DataFrame(data[0], columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(data[1], name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

# Create and fit the GATree classifier
gatree = GATreeClassifier(n_jobs=16, random_state=32)
gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
print(accuracy_score(y_test, y_pred))
