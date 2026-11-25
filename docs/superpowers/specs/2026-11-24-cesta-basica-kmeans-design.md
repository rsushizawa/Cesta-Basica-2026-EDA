# Cesta Básica EDA and scikit-learn KMeans design

## Intent

Build a reproducible Polars data pipeline for the supplied cesta básica CSV. The pipeline preserves raw observations, separates accepted and rejected records, creates auditable gold features, performs exploratory analysis, and clusters establishments with scikit-learn KMeans.

## Goals

- Preserve raw values and source row identifiers in bronze.
- Keep validation flags and rejected rows for auditability in silver.
- Create one establishment-level feature row per clustering entity in gold.
- Produce EDA tables and figures for data quality, distributions, and clusters.
- Use only scikit-learn KMeans for the active model backend.
- Save Parquet, JSON, PNG, and joblib artifacts under a timestamped run directory.

## Non-goals

- No Torch or custom NumPy KMeans implementation.
- No forecasting or train/test claims; clustering describes the observed period.
- No silent deletion of malformed, duplicate, or outlier observations.

## Constraints and decisions

- Python 3.12, Polars, NumPy, scikit-learn, Matplotlib, and joblib remain the implementation stack.
- The package name is `cestameans`; stale `icb` imports are corrected.
- Product aliases are optional. When no alias file exists, normalized source product names are used as canonical products.
- Invalid records are retained in `silver_rejected.parquet`; only hard-invalid rows are excluded from gold features.
- KMeans features are robustly scaled and median-imputed through a scikit-learn pipeline.
- The default entity is establishment-month, allowing temporal price behavior to remain visible to EDA and clustering.

## Approaches considered

1. Focused scikit-learn pipeline: selected because it matches the request and minimizes unsupported backends.
2. Multi-backend clustering: rejected because NumPy/Torch are outside scope and add maintenance.
3. Notebook-only analysis: rejected because it would not provide reusable bronze/silver/gold artifacts or a repeatable CLI.

## Implementation shape

Ingestion scans the CSV as strings and adds provenance. Parsing and normalization create typed values and flags. Validation creates accepted/rejected data. Feature transformations aggregate monthly prices, calculate market-relative values and temporal changes, then summarize establishment-month records. EDA computes quality and clustering diagnostics. The runner persists every stage and model artifact.

## Builder decisions

- Missing optional aliases do not stop execution.
- The default cluster count is four and can be overridden by CLI/configuration.
- K values and seed stability are evaluated as diagnostics; the configured K is used for the final model.
