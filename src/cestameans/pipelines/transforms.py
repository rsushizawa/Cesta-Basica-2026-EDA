from collections.abc import Callable, Sequence

import polars as pl

Transformation = Callable[[pl.LazyFrame], pl.LazyFrame]


class TransformationPipeline:
    def __init__(self, steps: Sequence[Transformation]) -> None:
        # Store ordered transformations; avoid mutation of caller's list.
        self.steps = tuple(steps)

    def run(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Apply each transformation in order; remain lazy.
        for step in self.steps:
            df = step(df)
        return df
