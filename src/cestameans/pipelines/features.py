import polars as pl
from icb.pipelines.transforms import TransformationPipeline

class FeaturePipeline:
    def __init__(self, shared: TransformationPipeline, entity: TransformationPipeline) -> None:
        # Inject shared economic transformations and entity-specific aggregation.
        raise NotImplementedError

    def run(self, silver: pl.LazyFrame) -> pl.LazyFrame:
        # Execute shared then entity-specific features; enforce output grain.
        raise NotImplementedError
