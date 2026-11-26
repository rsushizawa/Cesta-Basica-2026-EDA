from cestameans.config import ModelConfig
from cestameans.models.base import ClusteringModel
from cestameans.models.sklearn_kmeans import SklearnKMeans


def create_model(config: ModelConfig) -> ClusteringModel:
    # Select sklearn, numpy or torch backend; reject unknown values; no training here.
    if config.backend != "sklearn":
        raise ValueError("Only the sklearn backend is supported")
    return SklearnKMeans(config)
