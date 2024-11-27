"""
The following example shows how to perform fair clustering
of the adult dataset using GATree

Before running this example, install matplotlib and fairlearn:
poetry add matplotlib
poetry add fairlearn
"""

import pandas as pd
from fairlearn.datasets import fetch_adult
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from gatree.methods.gatreeclustering import GATreeClustering
from sklearn.metrics import silhouette_samples


# Load the adult dataset
adult = fetch_adult()

X = pd.DataFrame(adult.data, columns=adult.feature_names)
y = pd.Series(adult.target, name='target')

X.dropna(inplace=True)

sample_size = min(3000, len(X))

X.drop(columns=['workclass', 'occupation', 'native-country'], axis=1, inplace=True)
X = X.sample(n=sample_size, random_state=32)

sensitive_data = X['sex']

X = X.drop(columns=['sex'], axis=1)

# Preprocessing
X = pd.get_dummies(X)

label_to_number = {label: i for i, label in enumerate(sensitive_data.unique())}

print('sens', sensitive_data)

Z = sensitive_data.map(label_to_number)

print('Z len', len(Z))

min_clusters = 5

print('X', X, 'Z', Z)

Z = Z.reset_index(drop=True)
X = X.reset_index(drop=True)

fitness_function_kwargs = {
    'min_clusters': min_clusters,
    'fitness_X': X,
    'fitness_Z': Z
}


def fair_fitness_function_SS(root, **fitness_function_kwargs):
    if len(set(root.y_pred)) < fitness_function_kwargs['min_clusters']:
        return 1

    y_pred = pd.Series(root.y_pred)
    SSS = []
    Z = fitness_function_kwargs['fitness_Z']
    X = fitness_function_kwargs['fitness_X']

    for z_i in Z.unique():
        y_i = y_pred.loc[Z == z_i]
        X_i = X.loc[Z == z_i]
        SS_i = silhouette_score(X_i, y_i)
        SSS.append(SS_i)

    SS_max = (max(SSS) + 1) / 2
    SS_min = (min(SSS) + 1) / 2
    SS_diff = abs(SS_max - SS_min)

    return 1 - ((silhouette_score(fitness_function_kwargs['fitness_X'], root.y_pred) + 1) / 2) + (0.002 * root.size()) \
        + (SS_diff / 2)

# Create and fit the GATree clustering
gatree_SS = GATreeClustering(n_jobs=16, random_state=32, min_clusters=min_clusters, fitness_function=fair_fitness_function_SS)
gatree_SS.fit(X=X, population_size=20, max_iter=20, fitness_function_kwargs=fair_fitness_function_SS)

# Plot fitness values over iterations
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.plot(gatree._avg_fitness, linestyle='-')
# ax1.set_title('Average fitness values over iterations')
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('Average fitness')
# ax1.grid(True)
# ax2.plot(gatree._best_fitness, linestyle='-', color='green')
# ax2.set_title('Best fitness values over iterations')
# ax2.set_xlabel('Iterations')
# ax2.set_ylabel('Best fitness')
# ax2.grid(True)
# plt.tight_layout()
# plt.show()

gatree_SS.plot()

y_pred = gatree_SS._tree.y_pred

sil_scores = silhouette_samples(X, y_pred)
silhouette_df = pd.DataFrame({
    'Z': Z,
    'Silhouette_Score': sil_scores
})

avg_silhouette_per_Z = silhouette_df.groupby('Z')['Silhouette_Score'].mean()

print('avg_sil', avg_silhouette_per_Z)
print('silhouette normal', silhouette_score(X, y_pred))
