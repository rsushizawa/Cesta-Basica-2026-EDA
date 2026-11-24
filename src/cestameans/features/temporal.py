import polars as pl


def add_monthly_changes(df: pl.LazyFrame) -> pl.LazyFrame:
    # Sort by store/product/month; create mom_change and log_price_return only for consecutive calendar months.
    raise NotImplementedError


def add_yearly_changes(df: pl.LazyFrame) -> pl.LazyFrame:
    # Join to exact month 12 months earlier, not the 12th previous row; create yoy_change.
    raise NotImplementedError


def add_rolling_features(df: pl.LazyFrame, window_months: int = 3) -> pl.LazyFrame:
    # Use calendar-aware trailing windows; no future observations; document minimum valid periods.
    raise NotImplementedError


