from pathlib import Path

import polars as pl


def load_national_basket(path: Path) -> pl.DataFrame:
    """Load the national basket quantities used for cost calculations."""
    return pl.read_csv(path, separator=",", infer_schema_length=100)


def build_basket_cube(
    monthly_prices: pl.LazyFrame,
    basket_definition: pl.DataFrame,
    min_stores: int = 5,
) -> pl.LazyFrame:
    """Build product facts and establishment-month basket totals.

    The input median_price is a price per product base unit. When a basket
    category contains variants, the category price is the median variant price
    for that establishment and month. Incomplete baskets remain visible but
    basket_cost is null until all national categories are observed.
    """
    product_map = (
        basket_definition.with_columns(pl.col("source_products").str.split("|"))
        .explode("source_products")
        .rename({"source_products": "product"})
    )
    expected_categories = basket_definition.height
    category_prices = (
        monthly_prices.join(product_map.lazy(), on="product", how="inner")
        .group_by(["establishment_id", "reference_month", "category"])
        .agg(
            pl.col("median_price").median().alias("category_unit_price"),
            pl.col("product").n_unique().alias("matched_products"),
            pl.first("basket_quantity").alias("basket_quantity"),
            pl.first("basket_unit").alias("basket_unit"),
        )
        .with_columns(
            (pl.col("category_unit_price") * pl.col("basket_quantity")).alias(
                "basket_line_cost"
            )
        )
    )
    market = (
        category_prices.group_by(["reference_month", "category"])
        .agg(
            pl.col("category_unit_price").median().alias("market_category_unit_price"),
            pl.col("establishment_id").n_unique().alias("category_store_count"),
        )
        .with_columns(
            pl.when(pl.col("category_store_count") >= min_stores)
            .then(pl.col("market_category_unit_price"))
            .otherwise(None)
            .alias("market_category_unit_price")
        )
    )
    return (
        category_prices.join(market, on=["reference_month", "category"], how="left")
        .with_columns(
            (pl.col("category_unit_price") / pl.col("market_category_unit_price") - 1)
            .mul(100)
            .alias("category_vs_market_pct")
        )
        .group_by(["establishment_id", "reference_month"])
        .agg(
            pl.col("basket_line_cost").sum().alias("basket_cost_partial"),
            pl.col("category").n_unique().alias("basket_categories_observed"),
            pl.col("category").sort().alias("basket_categories"),
            pl.col("matched_products").sum().alias("basket_products_matched"),
        )
        .with_columns(
            (pl.col("basket_categories_observed") / expected_categories * 100).alias(
                "basket_coverage_pct"
            ),
            (pl.col("basket_categories_observed") == expected_categories).alias(
                "complete_basket"
            ),
            pl.when(pl.col("basket_categories_observed") == expected_categories)
            .then(pl.col("basket_cost_partial"))
            .otherwise(None)
            .alias("basket_cost"),
        )
        .sort(["establishment_id", "reference_month"])
        .with_columns(
            pl.col("basket_cost")
            .pct_change()
            .over("establishment_id")
            .alias("basket_cost_mom_change")
        )
        .with_columns(
            pl.col("basket_cost")
            .median()
            .over("reference_month")
            .alias("market_basket_cost")
        )
        .with_columns(
            (pl.col("basket_cost") / pl.col("market_basket_cost") - 1)
            .mul(100)
            .alias("basket_cost_vs_market_pct")
        )
    )
