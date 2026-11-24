from icb.config import ModelConfig
from icb.models.base import ClusteringModel


def create_model(config: ModelConfig) -> ClusteringModel:
    # Select sklearn, numpy or torch backend; reject unknown values; no training here.
    raise NotImplementedError


