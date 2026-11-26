import numpy as np
import polars as pl
from sklearn.metrics import silhouette_samples


def score_cluster_outliers(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    quantile: float = 0.99,
) -> pl.DataFrame:
    """Score observations by distance to their assigned KMeans centroid.

    A point is flagged when its distance is in the furthest ``quantile`` of
    distances within its own cluster. Silhouette is reported separately so
    distant points and ambiguous assignments are not conflated.
    """
    assigned_centroids = centroids[labels]
    distances = np.linalg.norm(X - assigned_centroids, axis=1)
    thresholds = np.zeros(len(distances), dtype=float)
    for cluster in np.unique(labels):
        cluster_distances = distances[labels == cluster]
        thresholds[labels == cluster] = np.quantile(cluster_distances, quantile)

    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
        silhouettes = silhouette_samples(X, labels)
    else:
        silhouettes = np.full(len(labels), np.nan)

    return pl.DataFrame(
        {
            "distance_to_centroid": distances,
            "cluster_distance_threshold": thresholds,
            "is_cluster_outlier": distances >= thresholds,
            "silhouette_sample": silhouettes,
            "is_ambiguous_assignment": silhouettes < 0,
        }
    )
