import numpy as np
import polars as pl

from cestameans.config import ModelConfig
from cestameans.evaluation.metrics import partition_agreement
from cestameans.models.sklearn_kmeans import SklearnKMeans


def evaluate_seed_stability(
    X: np.ndarray, config: ModelConfig, seeds: tuple[int, ...]
) -> pl.DataFrame:
    # Fit fresh models for each seed; compare partitions on identical rows with ARI.
    fitted = [
        SklearnKMeans(ModelConfig(**{**config.__dict__, "seed": seed}))
        .fit(X)
        .predict(X)
        for seed in seeds
    ]
    return pl.DataFrame(
        {
            "seed": list(seeds),
            "agreement_to_first": [partition_agreement(fitted[0], y) for y in fitted],
        }
    )
