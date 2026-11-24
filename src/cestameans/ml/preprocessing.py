import numpy as np
import polars as pl
from icb.config import FeatureSpec
from sklearn.pipeline import Pipeline

class MLPreprocessor:
    def __init__(self, spec: FeatureSpec, scaler: str = 'robust') -> None:
        # Store ordered feature specification and configure an unfitted sklearn pipeline.
        raise NotImplementedError

    def select(self, df: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        # Return numeric matrix and aligned store/month keys; exclude IDs and quality columns.
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> 'MLPreprocessor':
        # Fit imputation/scaling only on training data; reject entirely missing columns.
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Apply fitted parameters; validate shape and finite output; never refit.
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Compose fit and transform.
        raise NotImplementedError
