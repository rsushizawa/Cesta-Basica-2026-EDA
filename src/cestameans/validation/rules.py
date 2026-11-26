import polars as pl


def flag_required_fields(df: pl.LazyFrame) -> pl.LazyFrame:
    # Flag missing month, store or product; do not drop observations.
    return df.with_columns(
        (
            (pl.col("reference_month").is_null())
            | (pl.col("establishment_id").is_null())
            | (pl.col("product").is_null())
        ).alias("missing_required_field")
    )


def flag_invalid_prices(df: pl.LazyFrame) -> pl.LazyFrame:
    # Flag null, nonfinite or nonpositive price and quantity.
    return df.with_columns(
        (
            (pl.col("unit_price").is_null())
            | (~pl.col("unit_price").is_finite())
            | (pl.col("unit_price") <= 0)
            | (pl.col("quantity_value").is_null())
            | (pl.col("quantity_value") <= 0)
        ).alias("invalid_price")
    )


def flag_unit_price_mismatch(df: pl.LazyFrame, tolerance: float = 0.01) -> pl.LazyFrame:
    # Compare verified unit_price with reported_unit_price; distinguish missing PPK from mismatch.
    diff = (pl.col("unit_price") - pl.col("reported_unit_price_value")).abs()
    return df.with_columns(
        pl.col("reported_unit_price_value")
        .is_null()
        .alias("missing_reported_unit_price"),
        (
            pl.col("reported_unit_price_value").is_not_null()
            & (
                diff
                > tolerance
                * pl.max_horizontal([pl.col("unit_price").abs(), pl.lit(1.0)])
            )
        ).alias("unit_price_mismatch"),
    )


def flag_date_quality(df: pl.LazyFrame) -> pl.LazyFrame:
    # Flag missing collection dates and month mismatches separately.
    return df.with_columns(
        pl.col("collection_date").is_null().alias("missing_collection_date"),
        (
            pl.col("collection_date").is_not_null()
            & pl.col("reference_month").is_not_null()
            & (
                pl.col("collection_date").dt.month()
                != pl.col("reference_month").dt.month()
            )
        ).alias("month_mismatch"),
    )


def flag_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    # Compare original business columns, not observation IDs; preserve duplicates for review.
    keys = [
        c
        for c in (
            "reference_month",
            "establishment_id",
            "product",
            "brand_name",
            "package_price_value",
            "quantity_value",
        )
        if c in df.collect_schema().names()
    ]
    return df.with_columns(
        pl.struct(keys).is_duplicated().alias("duplicate_observation")
    )


def split_valid_rejected(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    # Split using explicit hard rejection flags; retain reason columns; no overlap or loss.
    flags = [
        "missing_required_field",
        "invalid_price",
        "unit_price_mismatch",
        "month_mismatch",
    ]
    rejected = pl.any_horizontal([pl.col(c).fill_null(False) for c in flags])
    return (
        df.with_columns(rejected.alias("hard_reject")).partition_by(
            "hard_reject", as_dict=False
        )
        if False
        else (df.filter(~rejected), df.filter(rejected))
    )
