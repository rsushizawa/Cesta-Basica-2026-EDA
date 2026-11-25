from cestameans.config import ModelConfig
from cestameans.models.sklearn_kmeans import SklearnKMeans


class NumpyKMeans(SklearnKMeans):
    """Compatibility wrapper; the supported implementation is sklearn KMeans."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
