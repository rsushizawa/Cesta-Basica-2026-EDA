import numpy as np
from sklearn.cluster import KMeans

from cestameans.config import ModelConfig


class SklearnKMeans:
    def __init__(self, config: ModelConfig) -> None:
        # Store hyperparameters; initialize unfitted state. Configure sklearn KMeans.
        self.config = config
        self.model = KMeans(
            n_clusters=config.n_clusters,
            max_iter=config.max_iter,
            n_init=config.n_init,
            tol=config.tolerance,
            random_state=config.seed,
        )

    def fit(self, X: np.ndarray) -> "SklearnKMeans":
        # Delegate to sklearn; map config parameters correctly.
        # Validate finite 2D input and feasible K; return self.
        self._validate(X)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return nearest-centroid integer labels; reject calls before fit.
        if not hasattr(self.model, "cluster_centers_"):
            raise RuntimeError("Model is not fitted")
        self._validate(X)
        return self.model.predict(X).astype(int)

    @property
    def centroids(self) -> np.ndarray:
        # Return fitted centroid matrix (K, D) in CPU NumPy format.
        if not hasattr(self.model, "cluster_centers_"):
            raise RuntimeError("Model is not fitted")
        return self.model.cluster_centers_

    @property
    def inertia(self) -> float:
        # Return fitted within-cluster squared error.
        if not hasattr(self.model, "inertia_"):
            raise RuntimeError("Model is not fitted")
        return float(self.model.inertia_)

    def _validate(self, X: np.ndarray) -> None:
        if (
            X.ndim != 2
            or X.shape[0] < self.config.n_clusters
            or not np.isfinite(X).all()
        ):
            raise ValueError("X must be finite 2D data with at least n_clusters rows")
