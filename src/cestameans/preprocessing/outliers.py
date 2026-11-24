import polars as pl


def flag_price_outliers(df: pl.LazyFrame, threshold: float = 3.5, min_group_size: int = 10) -> pl.LazyFrame:
    # Group by product and reference_month; compute median, MAD and modified z-score.
    # Leave small or zero-MAD groups unassessed; flag rather than automatically delete.
    raise NotImplementedError


