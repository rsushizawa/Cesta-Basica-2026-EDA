import numpy as np
import polars as pl
from icb.config import ModelConfig


def evaluate_seed_stability(X: np.ndarray, config: ModelConfig, seeds: tuple[int, ...]) -> pl.DataFrame:
    # Fit fresh models for each seed; compare partitions on identical rows with ARI.
    raise NotImplementedError


