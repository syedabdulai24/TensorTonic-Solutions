import numpy as np

def gini_impurity(y):
    """Calculate Gini impurity of a label array."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data
    based on minimum Gini impurity.
    """
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape

    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    # Try every feature
    for feature in range(n_features):
        values = X[:, feature]
        unique_values = np.unique(values)

        # Try thresholds between consecutive values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            left_mask = values <= threshold
            right_mask = values > threshold

            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                continue

            gini_left = gini_impurity(y[left_mask])
            gini_right = gini_impurity(y[right_mask])

            # weighted Gini
            weighted_gini = (
                len(y[left_mask]) / n_samples * gini_left +
                len(y[right_mask]) / n_samples * gini_right
            )

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return [best_feature, best_threshold]