"""
PCA Analysis Module
===================
Principal Component Analysis implementation with variance analysis and visualization support.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """PCA analysis with comprehensive variance and loading analysis."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_model = None
        self.fitted = False
        self.feature_names = []

    def fit_transform(
        self, X: pd.DataFrame, n_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fit PCA and transform data.

        Parameters
        ----------
        X : Scaled feature DataFrame
        n_components : Number of components. If None, fit with all components first.
        """
        self.feature_names = list(X.columns)

        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])

        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        transformed = self.pca_model.fit_transform(X)
        self.fitted = True

        logger.info(
            f"PCA fitted with {n_components} components. "
            f"Total variance explained: {self.pca_model.explained_variance_ratio_.sum():.4f}"
        )
        return transformed

    def get_explained_variance(self) -> Dict:
        """Get explained variance ratio and cumulative variance."""
        if not self.fitted:
            raise ValueError("PCA model not fitted yet.")

        evr = self.pca_model.explained_variance_ratio_
        cumulative = np.cumsum(evr)

        return {
            "explained_variance_ratio": evr.tolist(),
            "cumulative_variance": cumulative.tolist(),
            "n_components": len(evr),
            "total_variance_explained": float(evr.sum()),
        }

    def get_component_loadings(self) -> pd.DataFrame:
        """Get PCA component loadings (feature contributions to each component)."""
        if not self.fitted:
            raise ValueError("PCA model not fitted yet.")

        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            columns=[f"PC{i+1}" for i in range(self.pca_model.n_components_)],
            index=self.feature_names,
        )
        return loadings

    def get_top_features_per_component(self, n_top: int = 5) -> Dict:
        """Get the top contributing features for each principal component."""
        loadings = self.get_component_loadings()
        top_features = {}

        for col in loadings.columns:
            abs_loadings = loadings[col].abs().sort_values(ascending=False)
            top = abs_loadings.head(n_top)
            top_features[col] = {
                "features": top.index.tolist(),
                "loadings": [round(loadings.loc[f, col], 4) for f in top.index],
                "abs_loadings": [round(v, 4) for v in top.values],
            }

        return top_features

    def find_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """Find minimum components needed to retain given variance threshold."""
        if not self.fitted:
            raise ValueError("PCA model not fitted yet.")

        cumulative = np.cumsum(self.pca_model.explained_variance_ratio_)
        indices = np.where(cumulative >= variance_threshold)[0]
        if len(indices) == 0:
            # Threshold can't be reached with available components
            n_optimal = len(cumulative)
        else:
            n_optimal = int(indices[0] + 1)
        logger.info(
            f"Optimal components for {variance_threshold*100}% variance: {n_optimal}"
        )
        return n_optimal

    def transform_with_components(self, X: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Transform data using specified number of components."""
        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(X)
        columns = [f"PC{i+1}" for i in range(n_components)]
        result = pd.DataFrame(transformed, columns=columns, index=X.index)
        return result

    def get_pca_summary(self) -> Dict:
        """Get comprehensive PCA analysis summary."""
        if not self.fitted:
            raise ValueError("PCA model not fitted yet.")

        variance_info = self.get_explained_variance()
        top_features = self.get_top_features_per_component()
        optimal_95 = self.find_optimal_components(0.95)
        optimal_90 = self.find_optimal_components(0.90)
        optimal_80 = self.find_optimal_components(0.80)

        return {
            "variance": variance_info,
            "top_features": top_features,
            "optimal_components_95": optimal_95,
            "optimal_components_90": optimal_90,
            "optimal_components_80": optimal_80,
            "n_original_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

