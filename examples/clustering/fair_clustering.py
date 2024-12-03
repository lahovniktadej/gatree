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

Z = sensitive_data.map(label_to_number)

min_clusters = 5

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
gatree_SS.fit(X=X, population_size=20, max_iter=20, fitness_function_kwargs=fitness_function_kwargs)

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


def fair_fitness_function_ratios(root, **fitness_function_kwargs):
    if len(set(root.y_pred)) < fitness_function_kwargs['min_clusters']:
        return 1

    fitness_Z = fitness_function_kwargs['fitness_Z']
    y_pred = root.y_pred

    global_ratios = fitness_Z.value_counts(normalize=True).reset_index()
    global_ratios.columns = ['Z', 'global_ratio']

    data = pd.DataFrame({
        'y_pred': y_pred,
        'Z': fitness_Z
    })

    cluster_counts = data.groupby(['y_pred', 'Z']).size().reset_index(name='count')
    total_counts = cluster_counts.groupby('y_pred')['count'].transform('sum')
    cluster_counts['cluster_ratio'] = cluster_counts['count'] / total_counts

    merged = cluster_counts.merge(global_ratios, on='Z', how='left')

    merged['abs_diff'] = (merged['cluster_ratio'] - merged['global_ratio']).abs()

    aggregated_diff = merged.groupby('y_pred')['abs_diff'].sum().reset_index()
    aggregated_diff.columns = ['y_pred', 'sum_abs_diff']

    # print("Merged ratios with absolute differences:\n", merged)
    # print("Aggregated mean absolute differences by cluster:\n", aggregated_diff)

    abs_diff_sum = aggregated_diff['sum_abs_diff'].sum()

    # print('abs_diff_sum', abs_diff_sum)

    return 1 - ((silhouette_score(fitness_function_kwargs['fitness_X'], root.y_pred) + 1) / 2) + (0.002 * root.size()) + abs_diff_sum


gatree_ratios = GATreeClustering(n_jobs=16, random_state=32, min_clusters=min_clusters, fitness_function=fair_fitness_function_ratios)
gatree_ratios.fit(X=X, population_size=20, max_iter=20, fitness_function_kwargs=fitness_function_kwargs)

gatree_ratios.plot()

y_pred_ratios = gatree_ratios._tree.y_pred

sil_scores_ratios = silhouette_samples(X, y_pred_ratios)
silhouette_df_ratios = pd.DataFrame({
    'Z': Z,
    'Silhouette_Score': sil_scores_ratios
})

avg_silhouette_per_Z_ratios = silhouette_df_ratios.groupby('Z')['Silhouette_Score'].mean()

print('avg_sil', avg_silhouette_per_Z_ratios)
print('silhouette normal', silhouette_score(X, y_pred_ratios))
