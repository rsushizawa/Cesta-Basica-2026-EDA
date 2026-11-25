import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from cestameans.config import FeatureSpec


class MLPreprocessor:
    def __init__(self, spec: FeatureSpec, scaler: str = "robust") -> None:
        # Store ordered feature specification and configure an unfitted sklearn pipeline.
        self.spec = spec
        scaler_obj = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }[scaler]
        self.pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_obj)]
        )

    def select(self, df: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        # Return numeric matrix and aligned store/month keys; exclude IDs and quality columns.
        missing = [c for c in self.spec.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        keys = [c for c in ("establishment_id", "reference_month") if c in df.columns]
        return df.select(list(self.spec.columns)).to_numpy().astype(float), df.select(
            keys
        )

    def fit(self, X: np.ndarray) -> "MLPreprocessor":
        # Fit imputation/scaling only on training data; reject entirely missing columns.
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        self.pipeline.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Apply fitted parameters; validate shape and finite output; never refit.
        if not hasattr(self.pipeline, "n_features_in_"):
            raise RuntimeError("Preprocessor is not fitted")
        out = self.pipeline.transform(X)
        if not np.isfinite(out).all():
            raise ValueError("Preprocessing produced non-finite values")
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Compose fit and transform.
        return self.fit(X).transform(X)
