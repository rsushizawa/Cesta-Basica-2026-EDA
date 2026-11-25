import numpy as np
import polars as pl
from matplotlib.figure import Figure


def plot_clusters(X: np.ndarray, labels: np.ndarray) -> Figure:
    # Plot 2D features or PCA projection; label axes and explain projection.
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    projection = (
        X if X.shape[1] == 2 else PCA(n_components=2, random_state=42).fit_transform(X)
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", alpha=0.8)
    axis.set(
        xlabel="Feature 1 / PCA 1", ylabel="Feature 2 / PCA 2", title="KMeans clusters"
    )
    return figure


def plot_k_diagnostics(metrics: pl.DataFrame) -> Figure:
    # Plot inertia and silhouette versus K; return figure without saving.
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o")
    axes[0].set(xlabel="K", ylabel="Inertia", title="Inertia by K")
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o")
    axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette by K")
    figure.tight_layout()
    return figure


def plot_cluster_profiles(profiles: pl.DataFrame) -> Figure:
    # Visualize interpretable cluster feature summaries; label units.
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    feature_names = [
        column for column in profiles.columns if column not in {"cluster", "rows"}
    ]
    positions = np.arange(len(feature_names))
    for row in profiles.iter_rows(named=True):
        axis.plot(
            positions,
            [row[name] for name in feature_names],
            marker="o",
            label=f"Cluster {row['cluster']}",
        )
    axis.set_xticks(positions, feature_names, rotation=45, ha="right")
    axis.legend()
    axis.set(title="Cluster profiles", xlabel="Feature", ylabel="Mean value")
    figure.tight_layout()
    return figure


def plot_outlier_scores(outliers: pl.DataFrame) -> Figure:
    """Plot centroid distance and individual silhouette scores."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(
        np.arange(outliers.height),
        outliers["distance_to_centroid"],
        c=outliers["is_cluster_outlier"].cast(pl.Int8),
        cmap="coolwarm",
        alpha=0.7,
    )
    axes[0].set(
        title="Distance to assigned centroid", xlabel="Observation", ylabel="Distance"
    )
    axes[1].scatter(
        np.arange(outliers.height),
        outliers["silhouette_sample"],
        c=outliers["is_ambiguous_assignment"].cast(pl.Int8),
        cmap="coolwarm",
        alpha=0.7,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        title="Individual silhouette", xlabel="Observation", ylabel="Silhouette"
    )
    figure.tight_layout()
    return figure
