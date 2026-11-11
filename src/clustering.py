"""
Clustering Module
=================
K-Means clustering with evaluation metrics, elbow method, and silhouette analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """K-Means clustering with comprehensive evaluation and analysis."""

    def __init__(self, random_state: int = 42, max_iter: int = 300, n_init: int = 10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.model = None
        self.labels = None

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Fit K-Means and return cluster labels."""
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        self.labels = self.model.fit_predict(X)
        logger.info(
            f"K-Means fitted with k={n_clusters}. "
            f"Inertia: {self.model.inertia_:.2f}"
        )
        return self.labels

    def compute_elbow(
        self, X: np.ndarray, k_range: Tuple[int, int] = (2, 15),
    ) -> Dict:
        """
        Compute inertia for range of k values (Elbow Method).

        Returns dict with k_values, inertias, and suggested elbow point.
        """
        k_values = list(range(k_range[0], k_range[1] + 1))
        inertias = []
        times = []

        for k in k_values:
            start = time.time()
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
            )
            km.fit(X)
            inertias.append(float(km.inertia_))
            times.append(round(time.time() - start, 3))
            logger.info(f"Elbow: k={k}, inertia={km.inertia_:.2f}")

        # Heuristic elbow detection using rate of change
        elbow_k = self._detect_elbow(k_values, inertias)

        return {
            "k_values": k_values,
            "inertias": inertias,
            "times": times,
            "suggested_elbow": elbow_k,
        }

    def compute_silhouette_scores(
        self, X: np.ndarray, k_range: Tuple[int, int] = (2, 15),
    ) -> Dict:
        """Compute silhouette scores for range of k values."""
        k_values = list(range(k_range[0], k_range[1] + 1))
        scores = []

        for k in k_values:
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
            )
            labels = km.fit_predict(X)
            score = float(
                silhouette_score(
                    X, labels,
                    sample_size=min(3000, len(X)) if len(X) > 3000 else None,
                    random_state=self.random_state,
                )
            )
            scores.append(score)
            logger.info(f"Silhouette: k={k}, score={score:.4f}")

        best_k = k_values[np.argmax(scores)]

        return {
            "k_values": k_values,
            "silhouette_scores": scores,
            "best_k": best_k,
            "best_score": max(scores),
        }

    def get_cluster_analysis(
        self, X: np.ndarray, labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        original_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Comprehensive cluster analysis including sizes, centroids, and feature profiles."""
        n_clusters = len(np.unique(labels))

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        sizes = {f"Cluster {int(k)}": int(c) for k, c in zip(unique, counts)}

        # Centroids in original feature space
        if original_data is not None and feature_names:
            centroid_df = original_data.copy()
            centroid_df["cluster"] = labels
            centroids = centroid_df.groupby("cluster").mean()
            centroid_dict = centroids.to_dict()
        elif feature_names:
            centroids_array = np.zeros((n_clusters, X.shape[1]))
            for k in range(n_clusters):
                mask = labels == k
                centroids_array[k] = X[mask].mean(axis=0)
            centroid_dict = {
                feature_names[i]: {
                    f"Cluster {k}": round(centroids_array[k, i], 4)
                    for k in range(n_clusters)
                }
                for i in range(len(feature_names))
            }
        else:
            centroid_dict = {}

        # Silhouette analysis (using representative sample if dataset is large for interactive speed)
        try:
            if len(X) > 3000:
                rng = np.random.RandomState(self.random_state)
                sample_idx = rng.choice(len(X), size=3000, replace=False)
                X_sample = X[sample_idx]
                labels_sample = labels[sample_idx]
            else:
                X_sample = X
                labels_sample = labels

            sil_score = float(silhouette_score(X_sample, labels_sample))
            sil_samples = silhouette_samples(X_sample, labels_sample)
            sil_per_cluster = {}
            for k in range(n_clusters):
                mask = labels_sample == k
                if mask.any():
                    sil_per_cluster[f"Cluster {k}"] = round(float(sil_samples[mask].mean()), 4)
                else:
                    sil_per_cluster[f"Cluster {k}"] = 0.0
        except Exception as e:
            logger.warning(f"Silhouette calculation error: {e}")
            sil_score = None
            sil_per_cluster = {}

        return {
            "n_clusters": n_clusters,
            "sizes": sizes,
            "size_percentages": {
                k: round(v / sum(sizes.values()) * 100, 1)
                for k, v in sizes.items()
            },
            "centroids": centroid_dict,
            "silhouette_score": sil_score,
            "silhouette_per_cluster": sil_per_cluster,
            "inertia": float(self.model.inertia_) if self.model else None,
        }

    def _detect_elbow(self, k_values: List[int], inertias: List[float]) -> int:
        """Detect elbow point using the maximum curvature method."""
        if len(k_values) < 3:
            return k_values[0]

        # Compute second derivative (rate of change of change)
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)

        # Elbow is where the second derivative is maximum (most negative to less negative transition)
        if len(diffs2) > 0:
            elbow_idx = int(np.argmax(np.abs(diffs2))) + 1  # +1 because of diff offset
            return k_values[elbow_idx]
        return k_values[1]

    def get_interpretation_guidelines(self) -> Dict:
        """Return guidelines for interpreting K-Means clustering results."""
        return {
            "cluster_meaning": (
                "Clusters represent groups of weather observations with similar feature profiles. "
                "These should be interpreted as potential cloud-pattern regimes or weather states, "
                "NOT as confirmed meteorological cloud types."
            ),
            "elbow_method": (
                "The Elbow Method plots inertia (within-cluster sum of squares) against k. "
                "The 'elbow' point is where adding more clusters yields diminishing returns. "
                "This is a heuristic — the 'best' k may not always be obvious."
            ),
            "silhouette_score": (
                "Silhouette Score ranges from -1 to 1. Values above 0.5 indicate well-separated "
                "clusters. Values 0.25-0.5 suggest overlapping structure. Values below 0.25 "
                "indicate weak or arbitrary clustering."
            ),
            "limitations": [
                "K-Means assumes spherical, equally-sized clusters — weather data may not satisfy this.",
                "Results depend on the chosen number of clusters (k).",
                "K-Means is sensitive to feature scaling (ensure StandardScaler is applied).",
                "Cluster assignments may not correspond to meaningful physical categories.",
                "Different random initializations can produce different clusterings.",
                "The algorithm minimizes inertia, which may not align with domain-relevant groupings.",
            ],
            "validation_note": (
                "Independent meteorological validation is required to confirm whether discovered "
                "clusters correspond to actual cloud types or weather patterns."
            ),
        }

