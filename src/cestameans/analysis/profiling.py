import polars as pl


def profile_clusters(features: pl.DataFrame, assignments: pl.DataFrame, feature_columns: tuple[str, ...]) -> pl.DataFrame:
    # Join using entity keys with one-to-one validation; summarize original unscaled features by cluster.
    raise NotImplementedError


