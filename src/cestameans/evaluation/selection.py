import numpy as np
import polars as pl
from icb.config import ModelConfig


def evaluate_k_range(X: np.ndarray, config: ModelConfig, values: tuple[int, ...]) -> pl.DataFrame:
    # Fit fresh model for each K; collect metrics and seeds; do not select solely by inertia.
    raise NotImplementedError


