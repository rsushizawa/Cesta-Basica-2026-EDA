import numpy as np
from icb.config import ModelConfig

class SklearnKMeans:
    def __init__(self, config: ModelConfig) -> None:
        # Store hyperparameters; initialize unfitted state. Configure sklearn KMeans.
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> 'SklearnKMeans':
        # Delegate to sklearn; map config parameters correctly.
        # Validate finite 2D input and feasible K; return self.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return nearest-centroid integer labels; reject calls before fit.
        raise NotImplementedError

    @property
    def centroids(self) -> np.ndarray:
        # Return fitted centroid matrix (K, D) in CPU NumPy format.
        raise NotImplementedError

    @property
    def inertia(self) -> float:
        # Return fitted within-cluster squared error.
        raise NotImplementedError
