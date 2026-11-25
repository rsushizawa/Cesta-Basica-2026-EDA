import polars as pl

from cestameans.pipelines.transforms import TransformationPipeline


class FeaturePipeline:
    def __init__(
        self, shared: TransformationPipeline, entity: TransformationPipeline
    ) -> None:
        # Inject shared economic transformations and entity-specific aggregation.
        self.shared, self.entity = shared, entity

    def run(self, silver: pl.LazyFrame) -> pl.LazyFrame:
        # Execute shared then entity-specific features; enforce output grain.
        return self.entity.run(self.shared.run(silver))
