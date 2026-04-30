import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute squared distances
    train_sq = np.sum(X_train**2, axis=1)
    test_sq = np.sum(X_test**2, axis=1)

    dists = test_sq[:, None] + train_sq[None, :] - 2 * np.dot(X_test, X_train.T)
    dists = np.maximum(dists, 0)

    sorted_idx = np.argsort(dists, axis=1)

    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    # Output array filled with -1
    result = -1 * np.ones((n_test, k), dtype=int)

    usable_k = min(k, n_train)
    result[:, :usable_k] = sorted_idx[:, :usable_k]

    return result   # ✅ return NumPy array