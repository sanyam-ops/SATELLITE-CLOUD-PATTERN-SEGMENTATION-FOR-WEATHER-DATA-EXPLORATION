"""
Evaluation Module
=================
Comprehensive evaluation and comparison of dimensionality reduction and clustering methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


def compare_dimensionality_reduction(pca_result: Dict, tsne_result: Dict) -> Dict:
    """Compare PCA and t-SNE methods across multiple criteria."""
    comparison = {
        "PCA": {
            "type": "Linear",
            "preserves_global_structure": True,
            "preserves_local_structure": False,
            "information_retention": (
                f"{pca_result.get('variance', {}).get('total_variance_explained', 0)*100:.1f}% "
                "variance retained"
            ),
            "interpretable_components": True,
            "deterministic": True,
            "scalability": "Excellent — O(n·p²) where n=samples, p=features",
            "use_case": "Feature reduction, noise removal, visualization",
            "limitations": [
                "Only captures linear relationships",
                "May miss complex non-linear patterns",
                "Components may be hard to interpret for high-dimensional data",
            ],
        },
        "t-SNE": {
            "type": "Non-linear",
            "preserves_global_structure": False,
            "preserves_local_structure": True,
            "information_retention": "Not quantifiable — no explained variance metric",
            "interpretable_components": False,
            "deterministic": False,
            "scalability": "Poor for large datasets — O(n² log n)",
            "use_case": "Visualization of clusters and local neighborhoods",
            "limitations": [
                "Not reproducible without fixed random state",
                "Distances between clusters are meaningless",
                "Cluster sizes do not reflect actual data density",
                "Sensitive to hyperparameter choices (perplexity, learning_rate)",
                "Cannot transform new data points",
                "Primarily for visualization, not analytical reduction",
            ],
        },
    }

    if tsne_result and "elapsed_seconds" in tsne_result:
        comparison["t-SNE"]["computation_time"] = f"{tsne_result['elapsed_seconds']:.1f}s"
    if tsne_result and "kl_divergence" in tsne_result:
        comparison["t-SNE"]["kl_divergence"] = f"{tsne_result['kl_divergence']:.4f}"

    return comparison


def compare_clustering_configs(
    results: Dict[int, Dict],
) -> pd.DataFrame:
    """Compare clustering results across different k values."""
    rows = []
    for k, result in results.items():
        rows.append({
            "K": k,
            "Silhouette Score": result.get("silhouette_score", None),
            "Inertia": result.get("inertia", None),
            "Cluster Size Range": (
                f"{min(result.get('sizes', {}).values(), default=0)} - "
                f"{max(result.get('sizes', {}).values(), default=0)}"
            ),
            "Size Imbalance": (
                round(
                    max(result.get("sizes", {}).values(), default=1)
                    / max(min(result.get("sizes", {}).values(), default=1), 1),
                    2,
                )
                if result.get("sizes")
                else None
            ),
        })
    return pd.DataFrame(rows)


def generate_findings_summary(
    pca_summary: Dict,
    clustering_analysis: Dict,
    n_records: int,
    n_features: int,
) -> List[str]:
    """Generate key findings as a list of interpretive statements."""
    findings = []

    # PCA findings
    if pca_summary and "variance" in pca_summary:
        var = pca_summary["variance"]
        total_var = var.get("total_variance_explained", 0) * 100
        n_comp = var.get("n_components", 0)
        findings.append(
            f"PCA with {n_comp} components retains {total_var:.1f}% of the total variance "
            f"from the original {n_features} features."
        )

    if pca_summary and "optimal_components_95" in pca_summary:
        n_opt = pca_summary["optimal_components_95"]
        findings.append(
            f"Only {n_opt} principal components are needed to capture 95% of the variance, "
            f"suggesting significant redundancy in the original feature space."
        )

    # Clustering findings
    if clustering_analysis:
        sil = clustering_analysis.get("silhouette_score", 0)
        n_clusters = clustering_analysis.get("n_clusters", 0)
        if sil and sil > 0.5:
            findings.append(
                f"K-Means with k={n_clusters} yields a silhouette score of {sil:.3f}, "
                f"indicating well-separated cluster structure in the data."
            )
        elif sil and sil > 0.25:
            findings.append(
                f"K-Means with k={n_clusters} yields a silhouette score of {sil:.3f}, "
                f"suggesting moderate cluster structure with some overlap between groups."
            )
        elif sil:
            findings.append(
                f"K-Means with k={n_clusters} yields a silhouette score of {sil:.3f}, "
                f"indicating weak cluster separation. The data may not form distinct groups."
            )

    findings.append(
        f"Analysis performed on {n_records:,} observations across {n_features} features."
    )

    return findings


def generate_limitations() -> List[str]:
    """Generate comprehensive list of project limitations."""
    return [
        "This analysis is exploratory — clusters are mathematical groupings, not confirmed cloud types.",
        "The NOAA LCD dataset represents surface weather stations, not direct satellite pixel observations.",
        "Geographical coverage is limited to station locations, introducing spatial sampling bias.",
        "Temporal sampling depends on station operating hours and reporting frequency.",
        "Missing observations may be systematically absent (not missing at random).",
        "Satellite sensor calibration differences across instruments may affect consistency.",
        "K-Means assumes spherical clusters, which may not match actual weather pattern geometry.",
        "PCA captures only linear relationships; non-linear patterns may be lost.",
        "t-SNE visualizations should not be used to infer actual distances or cluster sizes.",
        "Weather data exhibits temporal autocorrelation that is not accounted for in this analysis.",
        "Dimensionality reduction necessarily loses some information from the original feature space.",
        "Results may differ significantly with different dataset subsets or time periods.",
    ]


def generate_ethical_considerations() -> List[str]:
    """Generate ethical considerations for the project."""
    return [
        "This application is for exploratory data analysis, NOT operational weather forecasting.",
        "Predictions or cluster assignments should never be used for safety-critical decisions.",
        "The dataset may underrepresent certain geographical regions or weather conditions.",
        "Satellite observations have known biases in cloud detection over bright surfaces (snow, desert).",
        "Temporal gaps in observations may correlate with extreme weather events (sensor damage).",
        "Climate change may render historical patterns unrepresentative of current conditions.",
        "Users should validate any patterns against authoritative meteorological references.",
        "The unsupervised approach intentionally avoids imposing human-defined categories on the data.",
    ]

