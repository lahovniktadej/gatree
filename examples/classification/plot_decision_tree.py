""" 
Example of plotting the final decision 
tree after fitting the model.
"""

from gatree.methods.gatreeclassifier import GATreeClassifier
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Initialise the tree
    gatree = GATreeClassifier()

    # Load the iris dataset and split it into X and y
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # Fit the model
    gatree.fit(X=X_train, y=y_train, population_size=100, max_iter=100)

    # Plot the final decision tree
    gatree.plot()
