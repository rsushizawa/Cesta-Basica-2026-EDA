import polars as pl


def profile_clusters(
    features: pl.DataFrame, assignments: pl.DataFrame, feature_columns: tuple[str, ...]
) -> pl.DataFrame:
    # Join using entity keys with one-to-one validation; summarize original unscaled features by cluster.
    joined = features.join(
        assignments, on=["establishment_id", "reference_month"], how="inner"
    )
    return (
        joined.group_by("cluster")
        .agg(
            [pl.col(c).mean().alias(f"mean_{c}") for c in feature_columns]
            + [pl.len().alias("rows")]
        )
        .sort("cluster")
    )
