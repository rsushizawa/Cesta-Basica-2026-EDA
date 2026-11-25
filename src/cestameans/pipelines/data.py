import polars as pl

from cestameans.ingestion.base import DataSource
from cestameans.pipelines.transforms import TransformationPipeline
from cestameans.validation.rules import split_valid_rejected


class DataPipeline:
    def __init__(
        self,
        source: DataSource,
        bronze: TransformationPipeline,
        silver: TransformationPipeline,
    ) -> None:
        # Inject source and both pipelines.
        self.source, self.bronze, self.silver = source, bronze, silver

    def build_bronze(self) -> pl.LazyFrame:
        # Read source, rename and parse while preserving every row and raw values.
        return self.bronze.run(self.source.read())

    def build_silver(self, bronze: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        # Validate, normalize and split accepted/rejected data; retain quality flags.
        return split_valid_rejected(self.silver.run(bronze))
