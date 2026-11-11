"""
t-SNE Analysis Module
=====================
t-Distributed Stochastic Neighbor Embedding implementation with parameter experimentation.
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class TSNEAnalyzer:
    """t-SNE analysis with parameter experimentation and PCA preprocessing."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}

    def run_tsne(
        self,
        X: np.ndarray,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        n_components: int = 2,
        pca_preprocess: bool = True,
        pca_components: int = 50,
        sample_size: Optional[int] = None,
    ) -> Dict:
        """
        Run t-SNE with specified parameters.

        Parameters
        ----------
        X : Input data (scaled features or PCA-reduced)
        perplexity : t-SNE perplexity parameter (5-100)
        learning_rate : t-SNE learning rate (10-1000)
        n_iter : Number of iterations (250-5000)
        n_components : Output dimensions (2 or 3)
        pca_preprocess : Whether to apply PCA before t-SNE
        pca_components : Number of PCA components for preprocessing
        sample_size : If set, randomly sample this many points

        Returns
        -------
        Dict with embedding, parameters, timing, and metadata
        """
        data = X.copy() if isinstance(X, np.ndarray) else X.values.copy()

        # Sampling for large datasets
        sampled = False
        sample_indices = None
        if sample_size and len(data) > sample_size:
            rng = np.random.RandomState(self.random_state)
            sample_indices = rng.choice(len(data), size=sample_size, replace=False)
            sample_indices.sort()
            data = data[sample_indices]
            sampled = True
            logger.info(f"Sampled {sample_size} points from {len(X)} for t-SNE")

        # PCA preprocessing (recommended for high-dimensional data)
        pca_applied = False
        if pca_preprocess and data.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=self.random_state)
            data = pca.fit_transform(data)
            pca_applied = True
            logger.info(f"PCA preprocessing: {X.shape[1]} → {pca_components} dimensions")

        # Adjust perplexity if too large for sample size
        effective_perplexity = min(perplexity, (len(data) - 1) / 3.0)
        if effective_perplexity != perplexity:
            logger.warning(
                f"Perplexity adjusted from {perplexity} to {effective_perplexity:.1f} "
                f"(must be < n_samples/3)"
            )

        # Run t-SNE
        start_time = time.time()
        # scikit-learn >= 1.6 renamed n_iter to max_iter
        tsne_kwargs = dict(
            n_components=n_components,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            random_state=self.random_state,
            init="pca" if data.shape[1] >= n_components else "random",
        )
        import inspect
        tsne_params = inspect.signature(TSNE.__init__).parameters
        if "max_iter" in tsne_params:
            tsne_kwargs["max_iter"] = n_iter
        else:
            tsne_kwargs["n_iter"] = n_iter
        tsne = TSNE(**tsne_kwargs)
        embedding = tsne.fit_transform(data)
        elapsed = time.time() - start_time

        result = {
            "embedding": embedding,
            "parameters": {
                "perplexity": effective_perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "n_components": n_components,
            },
            "kl_divergence": float(tsne.kl_divergence_),
            "n_iter_actual": n_iter,
            "elapsed_seconds": round(elapsed, 2),
            "n_samples": len(data),
            "sampled": sampled,
            "sample_indices": sample_indices,
            "pca_preprocessed": pca_applied,
        }

        return result

    def compare_perplexities(
        self, X: np.ndarray,
        perplexities: List[float] = [5, 15, 30, 50, 100],
        **kwargs,
    ) -> Dict:
        """Run t-SNE with multiple perplexity values for comparison."""
        results = {}
        for p in perplexities:
            try:
                logger.info(f"Running t-SNE with perplexity={p}")
                result = self.run_tsne(X, perplexity=p, **kwargs)
                results[f"perplexity_{p}"] = result
            except Exception as e:
                logger.error(f"t-SNE failed for perplexity={p}: {e}")
                results[f"perplexity_{p}"] = {"error": str(e)}
        return results

    def get_tsne_interpretation(self) -> Dict:
        """Return educational interpretation of t-SNE parameters and limitations."""
        return {
            "perplexity": {
                "description": (
                    "Perplexity controls the balance between local and global aspects of the data. "
                    "It loosely represents the number of nearest neighbors considered. "
                    "Lower values (5-15) emphasize local structure; higher values (30-100) "
                    "capture more global patterns."
                ),
                "recommendation": "Try values between 5 and 50. Start with 30.",
            },
            "learning_rate": {
                "description": (
                    "Controls the step size during optimization. Too low causes points to crowd; "
                    "too high causes instability. Typical range: 10 to 1000."
                ),
                "recommendation": "Default of 200 works well. Adjust if points appear overly crowded or spread.",
            },
            "n_iter": {
                "description": (
                    "Number of optimization iterations. More iterations allow convergence but increase "
                    "computation time. Minimum recommended: 250."
                ),
                "recommendation": "1000 iterations is generally sufficient. Increase to 2000-5000 for complex datasets.",
            },
            "limitations": [
                "t-SNE distances between clusters are NOT meaningful — clusters may appear equidistant even when they are not.",
                "Cluster sizes in t-SNE are NOT proportional to actual data density.",
                "t-SNE is stochastic — different runs may produce different layouts.",
                "Global structure is not reliably preserved; only local neighborhoods are meaningful.",
                "t-SNE is primarily a visualization tool, not suitable for downstream ML tasks.",
                "Perplexity significantly affects the resulting visualization — always compare multiple values.",
            ],
        }

