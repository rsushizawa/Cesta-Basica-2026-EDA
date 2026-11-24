from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class DataConfig:
    input_path: Path
    output_dir: Path
    aliases_path: Path
    outlier_threshold: float = 3.5
    min_outlier_group: int = 10
    min_market_stores: int = 5
    unit_price_tolerance: float = 0.01

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    columns: tuple[str, ...]
    min_products: int = 10
    min_matched_products: int = 0

@dataclass(frozen=True)
class ModelConfig:
    backend: Literal['sklearn', 'numpy', 'torch'] = 'sklearn'
    n_clusters: int = 4
    max_iter: int = 300
    n_init: int = 10
    tolerance: float = 1e-4
    seed: int = 42
    device: str = 'cpu'

@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    features: FeatureSpec
    model: ModelConfig
    scaler: Literal['standard', 'robust', 'minmax'] = 'robust'
