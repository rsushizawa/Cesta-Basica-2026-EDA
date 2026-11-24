import polars as pl
from icb.ingestion.base import DataSource
from icb.pipelines.transforms import TransformationPipeline

class DataPipeline:
    def __init__(self, source: DataSource, bronze: TransformationPipeline, silver: TransformationPipeline) -> None:
        # Inject source and both pipelines.
        raise NotImplementedError

    def build_bronze(self) -> pl.LazyFrame:
        # Read source, rename and parse while preserving every row and raw values.
        raise NotImplementedError

    def build_silver(self, bronze: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        # Validate, normalize and split accepted/rejected data; retain quality flags.
        raise NotImplementedError
