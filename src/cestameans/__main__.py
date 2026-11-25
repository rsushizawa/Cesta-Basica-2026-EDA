import argparse
from pathlib import Path

from cestameans.config import DataConfig, ExperimentConfig, FeatureSpec, ModelConfig
from cestameans.experiment import ExperimentRunner
from cestameans.features.aggregation import aggregate_monthly_prices
from cestameans.features.economic import add_market_reference, add_relative_prices
from cestameans.features.establishment import build_establishment_month_features
from cestameans.features.temporal import add_monthly_changes
from cestameans.infrastructure.artifacts import ArtifactRepository
from cestameans.ingestion.csv import CSVDataSource
from cestameans.ingestion.parsing import parse_dates, parse_numbers, rename_columns
from cestameans.ml.preprocessing import MLPreprocessor
from cestameans.models.factory import create_model
from cestameans.pipelines.data import DataPipeline
from cestameans.pipelines.features import FeaturePipeline
from cestameans.pipelines.transforms import TransformationPipeline
from cestameans.preprocessing.cleaning import calculate_unit_price, normalize_text
from cestameans.preprocessing.normalization import normalize_brands, normalize_products
from cestameans.preprocessing.outliers import flag_price_outliers
from cestameans.validation.rules import (
    flag_date_quality,
    flag_duplicates,
    flag_invalid_prices,
    flag_required_fields,
    flag_unit_price_mismatch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cesta Básica EDA and scikit-learn KMeans"
    )
    parser.add_argument("--input", type=Path, default=Path("data/ICB_DADOS_2026.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_config = DataConfig(
        input_path=args.input,
        output_dir=args.output,
        aliases_path=args.input.parent / "aliases.csv",
    )
    model_config = ModelConfig(n_clusters=args.k)
    config = ExperimentConfig(
        data=data_config,
        features=FeatureSpec(
            name="establishment_month",
            columns=(
                "basket_cost",
                "basket_cost_vs_market_pct",
                "basket_cost_mom_change",
                "mean_log_relative_price",
                "mean_relative_iqr",
            ),
            min_products=1,
        ),
        model=model_config,
    )

    bronze_steps = TransformationPipeline([rename_columns, parse_dates, parse_numbers])
    silver_steps = TransformationPipeline(
        [
            normalize_text,
            calculate_unit_price,
            normalize_brands,
            lambda df: normalize_products(df, None),
            flag_required_fields,
            flag_invalid_prices,
            flag_unit_price_mismatch,
            flag_date_quality,
            flag_duplicates,
            lambda df: flag_price_outliers(df),
        ]
    )
    data = DataPipeline(
        CSVDataSource(data_config.input_path), bronze_steps, silver_steps
    )
    features = FeaturePipeline(
        TransformationPipeline(
            [
                aggregate_monthly_prices,
                lambda df: add_market_reference(df, data_config.min_market_stores),
                add_relative_prices,
                add_monthly_changes,
            ]
        ),
        TransformationPipeline([build_establishment_month_features]),
    )
    runner = ExperimentRunner(
        config,
        data,
        features,
        MLPreprocessor(config.features, config.scaler),
        create_model(model_config),
        ArtifactRepository(data_config.output_dir),
    )
    result = runner.run()
    print(result.metrics)


if __name__ == "__main__":
    main()
