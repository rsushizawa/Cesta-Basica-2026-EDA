import numpy as np


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, inertia: float
) -> dict[str, float]:
    # Compute inertia, silhouette, Davies-Bouldin and Calinski-Harabasz; handle degenerate partitions explicitly.
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    result = {"inertia": float(inertia)}
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        result.update(
            {
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
            }
        )
    else:
        result.update(
            {
                "silhouette": float(silhouette_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )
    return result


def partition_agreement(a: np.ndarray, b: np.ndarray) -> float:
    # Compute adjusted Rand index; invariant to label permutation.
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))
