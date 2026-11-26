from dataclasses import asdict, dataclass

import polars as pl

from cestameans.analysis.profiling import profile_clusters
from cestameans.config import ExperimentConfig
from cestameans.evaluation.metrics import evaluate_clustering
from cestameans.evaluation.outliers import score_cluster_outliers
from cestameans.evaluation.selection import evaluate_k_range
from cestameans.features.aggregation import aggregate_monthly_prices
from cestameans.features.basket import build_basket_cube
from cestameans.infrastructure.artifacts import ArtifactRepository
from cestameans.ml.preprocessing import MLPreprocessor
from cestameans.models.base import ClusteringModel
from cestameans.pipelines.data import DataPipeline
from cestameans.pipelines.features import FeaturePipeline
from cestameans.validation.quality import summarize_quality
from cestameans.visualization.clusters import (
    plot_cluster_profiles,
    plot_clusters,
    plot_k_diagnostics,
    plot_outlier_scores,
)


@dataclass(frozen=True)
class ExperimentResult:
    assignments: pl.DataFrame
    metrics: dict[str, float]


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        data: DataPipeline,
        features: FeaturePipeline,
        preprocessor: MLPreprocessor,
        model: ClusteringModel,
        artifacts: ArtifactRepository,
    ) -> None:
        # Inject dependencies; avoid constructing concrete backends here.
        self.config = config
        self.data = data
        self.features = features
        self.preprocessor = preprocessor
        self.model = model
        self.artifacts = artifacts

    def run(self) -> ExperimentResult:
        # Build/persist bronze and silver (including rejected rows), then gold.
        # Apply coverage filters; select aligned feature matrix and keys.
        # Fit preprocessing and model; predict; evaluate and profile.
        # Save fitted objects, assignments, metrics and provenance.
        # This is descriptive full-period clustering; future evaluation needs a time split.
        bronze = self.data.build_bronze()
        self.artifacts.save_table("bronze", bronze)
        valid, rejected = self.data.build_silver(bronze)
        silver_with_flags = self.data.silver.run(bronze)
        self.artifacts.save_table(
            "quality_summary", summarize_quality(silver_with_flags)
        )
        self.artifacts.save_table("silver", valid)
        self.artifacts.save_table("silver_rejected", rejected)

        monthly_prices = aggregate_monthly_prices(valid).collect()
        basket_definition = (
            self.config.data.input_path.parent / "reference" / "national_basket.csv"
        )
        basket_cube = build_basket_cube(
            monthly_prices.lazy(),
            pl.read_csv(basket_definition),
            self.config.data.min_market_stores,
        ).collect()
        self.artifacts.save_table("basket_cube", basket_cube)

        gold = self.features.run(valid).collect()
        gold = gold.join(
            basket_cube,
            on=["establishment_id", "reference_month"],
            how="left",
        )
        self.artifacts.save_table("gold", gold)
        model_gold = gold.filter(pl.col("complete_basket").fill_null(False))
        self.artifacts.save_table("gold_model_input", model_gold)
        feature_columns = [
            column
            for column in self.config.features.columns
            if column in model_gold.columns
        ]
        self.artifacts.save_table(
            "gold_feature_summary", gold.select(feature_columns).describe()
        )
        X, keys = self.preprocessor.select(model_gold)
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled)
        labels = self.model.predict(X_scaled)
        assignments = keys.with_columns(pl.Series("cluster", labels))
        outlier_scores = score_cluster_outliers(X_scaled, labels, self.model.centroids)
        outliers = keys.hstack(outlier_scores)
        metrics = evaluate_clustering(X_scaled, labels, self.model.inertia)
        profiles = profile_clusters(
            model_gold, assignments, self.config.features.columns
        )
        diagnostics = evaluate_k_range(
            X_scaled,
            self.config.model,
            tuple(range(2, min(8, len(X_scaled) - 1) + 1)),
        )
        self.artifacts.save_table("k_diagnostics", diagnostics)
        self.artifacts.save_table("assignments", assignments)
        self.artifacts.save_table("outliers", outliers)
        self.artifacts.save_table("cluster_profiles", profiles)
        self.artifacts.save_model("preprocessor", self.preprocessor)
        self.artifacts.save_model("kmeans", self.model)
        self.artifacts.save_metadata("metrics", metrics)
        self.artifacts.save_metadata(
            "run",
            {
                "config": asdict(self.config),
                "feature_columns": list(self.config.features.columns),
                "n_rows_gold": gold.height,
                "n_clusters": self.config.model.n_clusters,
            },
        )
        self.artifacts.save_figure("clusters", plot_clusters(X_scaled, labels))
        self.artifacts.save_figure("k_diagnostics", plot_k_diagnostics(diagnostics))
        self.artifacts.save_figure("cluster_profiles", plot_cluster_profiles(profiles))
        self.artifacts.save_figure("outliers", plot_outlier_scores(outliers))
        return ExperimentResult(assignments=assignments, metrics=metrics)
