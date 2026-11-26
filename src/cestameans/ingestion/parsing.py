import polars as pl

from cestameans.ingestion.schema import COLUMN_MAPPING


def rename_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    # Use COLUMN_MAPPING; verify required headers; preserve observation ID.
    missing = set(COLUMN_MAPPING) - set(df.collect_schema().names())
    if missing:
        raise ValueError(f"Missing required headers: {sorted(missing)}")
    return df.rename(COLUMN_MAPPING)


def parse_dates(df: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve raw strings; parse dd/mm/yy (and verified four-digit variants).
    # Normalize reference month to first day; distinguish missing from malformed dates.
    return df.with_columns(
        pl.col("reference_month")
        .str.strptime(pl.Date, "%d/%m/%y", strict=False)
        .dt.month_start(),
        pl.col("collection_date").str.strptime(pl.Date, "%d/%m/%y", strict=False),
    ).with_columns(
        pl.col("reference_month").is_null().alias("reference_month_parse_failed"),
        pl.col("collection_date").is_null().alias("collection_date_parse_failed"),
    )


def parse_numbers(df: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve raw price and quantity strings; cast to Float64; flag parse failures.
    return df.with_columns(
        pl.col("package_price")
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
        .alias("package_price_value"),
        pl.col("quantity")
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
        .alias("quantity_value"),
        pl.col("reported_unit_price")
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
        .alias("reported_unit_price_value"),
    )
