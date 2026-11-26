import polars as pl


def summarize_quality(df: pl.LazyFrame) -> pl.DataFrame:
    # Count rows, nulls, malformed values, duplicates and flags; report by month where useful.
    x = df.collect()
    metrics = {"rows": x.height, "columns": x.width}
    flag_columns = [
        "missing_required_field",
        "invalid_price",
        "unit_price_mismatch",
        "month_mismatch",
        "duplicate_observation",
        "is_price_outlier",
    ]
    for column in flag_columns:
        if column in x:
            metrics[column] = int(x.get_column(column).fill_null(False).sum())
    return pl.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def assert_unique_key(df: pl.LazyFrame, columns: tuple[str, ...]) -> None:
    # Raise a descriptive error if keys are null or repeated.
    x = df.select(list(columns)).collect()
    if any(x.get_column(c).is_null().any() for c in columns):
        raise ValueError(f"Null values found in key {columns}")
    if x.height != x.unique(subset=list(columns)).height:
        raise ValueError(f"Duplicate key found for {columns}")
