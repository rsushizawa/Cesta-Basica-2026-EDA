import numpy as np
import polars as pl

from cestameans.config import ModelConfig
from cestameans.evaluation.metrics import evaluate_clustering
from cestameans.models.sklearn_kmeans import SklearnKMeans


def evaluate_k_range(
    X: np.ndarray, config: ModelConfig, values: tuple[int, ...]
) -> pl.DataFrame:
    # Fit fresh model for each K; collect metrics and seeds; do not select solely by inertia.
    rows = []
    for k in values:
        c = ModelConfig(**{**config.__dict__, "n_clusters": k})
        m = SklearnKMeans(c).fit(X)
        rows.append({"k": k, **evaluate_clustering(X, m.predict(X), m.inertia)})
    return pl.DataFrame(rows)
