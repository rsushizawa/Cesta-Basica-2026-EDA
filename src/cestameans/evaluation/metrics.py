import numpy as np


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, inertia: float) -> dict[str, float]:
    # Compute inertia, silhouette, Davies-Bouldin and Calinski-Harabasz; handle degenerate partitions explicitly.
    raise NotImplementedError


def partition_agreement(a: np.ndarray, b: np.ndarray) -> float:
    # Compute adjusted Rand index; invariant to label permutation.
    raise NotImplementedError


