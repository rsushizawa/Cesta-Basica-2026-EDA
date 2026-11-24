from dataclasses import dataclass
import polars as pl
from icb.config import ExperimentConfig
from icb.pipelines.data import DataPipeline
from icb.pipelines.features import FeaturePipeline
from icb.ml.preprocessing import MLPreprocessor
from icb.models.base import ClusteringModel
from icb.infrastructure.artifacts import ArtifactRepository

@dataclass(frozen=True)
class ExperimentResult:
    assignments: pl.DataFrame
    metrics: dict[str, float]

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, data: DataPipeline, features: FeaturePipeline, preprocessor: MLPreprocessor, model: ClusteringModel, artifacts: ArtifactRepository) -> None:
        # Inject dependencies; avoid constructing concrete backends here.
        raise NotImplementedError

    def run(self) -> ExperimentResult:
        # Build/persist bronze and silver (including rejected rows), then gold.
        # Apply coverage filters; select aligned feature matrix and keys.
        # Fit preprocessing and model; predict; evaluate and profile.
        # Save fitted objects, assignments, metrics and provenance.
        # This is descriptive full-period clustering; future evaluation needs a time split.
        raise NotImplementedError
