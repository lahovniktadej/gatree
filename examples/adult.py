"""
The following example shows how to perform classification 
of the adult dataset using GATree

Before running this example, install the aif360 library:
poetry add aif360
"""

from aif360.sklearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree.gatree import GATree

# Load the adult dataset
adult = fetch_adult(numeric_only=True)
X = adult.X
y = adult.y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)

# Create and fit the GATree classifier
gatree = GATree(n_jobs=16, random_state=1)
gatree.fit(X=X_train, y=y_train, population_size=25, max_iter=25)

# Make predictions on the testing set
y_pred = gatree.predict(X_test)

# Evaluate the accuracy of the classifier
print(accuracy_score(y_test, y_pred))
