"""
Page 8: Conclusions & Report
=============================
Key findings, limitations, ethical considerations, and downloadable report.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pca_analysis import PCAAnalyzer
from src.clustering import KMeansClusterer
from src.evaluation import (
    generate_findings_summary,
    generate_limitations,
    generate_ethical_considerations,
)
from config import VIZ_CONFIG, PCA_CONFIG, KMEANS_CONFIG

TEMPLATE = VIZ_CONFIG["plotly_template"]


@st.cache_data
def get_conclusions_data(scaled_df, original_df):
    pca_analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
    pca_data = pca_analyzer.fit_transform(scaled_df)
    pca_summary = pca_analyzer.get_pca_summary()

    clusterer = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
    labels = clusterer.fit_predict(pca_data[:, :5], n_clusters=5)
    cluster_analysis = clusterer.get_cluster_analysis(
        pca_data[:, :5], labels,
        feature_names=list(original_df.columns),
        original_data=original_df,
    )
    return pca_summary, cluster_analysis


def render_conclusions(scaled_df, original_df, raw_df):
    """Render the Conclusions & Report page."""

    st.markdown("""
    <h1>📋 Conclusions & Report</h1>
    <p style='color: #8892A0;'>Key findings, limitations, ethical considerations, and downloadable project report.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # Run PCA and Clustering for findings (cached)
    pca_summary, cluster_analysis = get_conclusions_data(scaled_df, original_df)

    tabs = st.tabs(["🔍 Key Findings", "⚠️ Limitations", "🔒 Ethics", "📥 Download Report"])

    # ── TAB 1: Key Findings ────────────────────────────────
    with tabs[0]:
        st.markdown("### 🔍 Key Observations & Findings")

        findings = generate_findings_summary(
            pca_summary, cluster_analysis,
            n_records=len(scaled_df),
            n_features=len(scaled_df.columns),
        )

        for i, finding in enumerate(findings, 1):
            st.markdown(f"""
            <div class='custom-card'>
                <p style='color: #CBD5E1;'><b style='color: #00F0FF;'>{i}.</b> {finding}</p>
            </div>
            """, unsafe_allow_html=True)

        # Distinction of evidence levels
        st.markdown("### 📐 Levels of Evidence")
        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #10B981;'>1. Observed Mathematical Patterns</h4>
            <p style='color: #CBD5E1;'>
                PCA identifies principal axes of variance. K-Means partitions the feature space
                into K groups that minimize within-cluster distance. These are <b>mathematical facts</b>
                about the dataset structure.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #FBBF24;'>2. Possible Meteorological Interpretations</h4>
            <p style='color: #CBD5E1;'>
                Clusters with high humidity, low visibility, and low cloud ceiling <i>may</i>
                correspond to fog or stratus conditions. Clusters with high temperature and low
                humidity <i>might</i> represent clear-sky arid conditions. These are <b>plausible
                interpretations</b> that require domain expert validation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #F43F5E;'>3. Conclusions Requiring Independent Validation</h4>
            <p style='color: #CBD5E1;'>
                Any claim that specific clusters represent actual cloud types (cumulus, stratus,
                cirrus, etc.) requires independent validation against labeled meteorological data,
                satellite imagery classification, or expert analysis. This project <b>does not</b>
                make such claims.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # PCA Findings
        st.markdown("### 📊 PCA Analysis Summary")
        var_info = pca_summary["variance"]
        evr = var_info["explained_variance_ratio"]
        cumvar = var_info["cumulative_variance"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Components", len(evr))
        c2.metric("95% Variance Components", pca_summary["optimal_components_95"])
        c3.metric("PC1 Variance", f"{evr[0]*100:.1f}%")

        st.markdown("**Top contributing features to PC1:**")
        if "top_features" in pca_summary and "PC1" in pca_summary["top_features"]:
            pc1_info = pca_summary["top_features"]["PC1"]
            for feat, loading in zip(pc1_info["features"][:5], pc1_info["loadings"][:5]):
                direction = "↑" if loading > 0 else "↓"
                st.markdown(f"- **{feat}**: {loading:.4f} {direction}")

        # Clustering Findings
        st.markdown("### 🎯 K-Means Clustering Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("K Selected", cluster_analysis["n_clusters"])
        c2.metric("Silhouette Score", f"{cluster_analysis['silhouette_score']:.4f}" if cluster_analysis["silhouette_score"] else "—")
        c3.metric("Inertia", f"{cluster_analysis['inertia']:.0f}" if cluster_analysis["inertia"] else "—")

        st.markdown("**Cluster Size Distribution:**")
        for cluster, size in cluster_analysis["sizes"].items():
            pct = cluster_analysis["size_percentages"][cluster]
            st.markdown(f"- {cluster}: {size:,} observations ({pct}%)")

    # ── TAB 2: Limitations ─────────────────────────────────
    with tabs[1]:
        st.markdown("### ⚠️ Project Limitations")
        limitations = generate_limitations()

        for i, lim in enumerate(limitations, 1):
            st.markdown(f"""
            <div class='custom-card'>
                <p style='color: #B0B8C8;'><b style='color: #FFE66D;'>{i}.</b> {lim}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔮 Future Improvements")
        improvements = [
            "Integrate actual INSAT-3D/3DR satellite data from MOSDAC when download access is available.",
            "Apply DBSCAN and Gaussian Mixture Models for density-based and soft clustering.",
            "Incorporate temporal analysis — identify how cloud patterns evolve across seasons.",
            "Add geographic clustering to discover region-specific weather regimes.",
            "Implement UMAP as a modern alternative to t-SNE for dimensionality reduction.",
            "Validate clusters against actual meteorological cloud classification databases.",
            "Deploy the application on a cloud platform for wider accessibility.",
            "Add real-time data ingestion from NOAA APIs for live analysis.",
        ]
        for imp in improvements:
            st.markdown(f"- 🚀 {imp}")

    # ── TAB 3: Ethics ──────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🔒 Ethical Considerations")
        ethics = generate_ethical_considerations()

        for i, eth in enumerate(ethics, 1):
            st.markdown(f"""
            <div class='custom-card'>
                <p style='color: #B0B8C8;'><b style='color: #A855F7;'>{i}.</b> {eth}</p>
            </div>
            """, unsafe_allow_html=True)

        st.error(
            "🚨 **Critical Notice:** This application is for **educational and exploratory purposes only**. "
            "It must NOT be used for operational weather forecasting, aviation decisions, disaster management, "
            "or any safety-critical application."
        )

    # ── TAB 4: Download Report ─────────────────────────────
    with tabs[3]:
        st.markdown("### 📥 Download Project Report")

        report = _generate_report(pca_summary, cluster_analysis, scaled_df, original_df, raw_df)

        st.download_button(
            label="📄 Download Full Report (Markdown)",
            data=report,
            file_name=f"satellite_cloud_segmentation_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

        with st.expander("👀 Preview Report"):
            st.markdown(report[:5000] + "\n\n... *(truncated for preview)*")

        # References
        st.markdown("### 📚 References")
        refs = [
            "Jolliffe, I. T., & Cadima, J. (2016). Principal Component Analysis: A Review and Recent Developments. *Philosophical Transactions of the Royal Society A*, 374(2065).",
            "van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.",
            "Lloyd, S. P. (1982). Least Squares Quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.",
            "Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.",
            "NOAA NCEI. Local Climatological Data (LCD). https://www.ncei.noaa.gov/data/local-climatological-data/",
            "ISRO MOSDAC. INSAT-3D Satellite Data Products. https://mosdac.gov.in/",
            "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.",
            "Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to Use t-SNE Effectively. *Distill*. https://distill.pub/2016/misread-tsne/",
        ]
        for i, ref in enumerate(refs, 1):
            st.markdown(f"{i}. {ref}")


def _generate_report(pca_summary, cluster_analysis, scaled_df, original_df, raw_df):
    """Generate a comprehensive markdown project report."""
    evr = pca_summary["variance"]["explained_variance_ratio"]
    cumvar = pca_summary["variance"]["cumulative_variance"]

    report = f"""# Satellite Cloud Pattern Segmentation for Weather Data Exploration
## Using PCA, t-SNE, and K-Means Clustering

**Project Type:** Unsupervised Machine Learning Research Project
**Domain:** Satellite Meteorology & Atmospheric Pattern Segmentation
**Date:** {datetime.now().strftime('%B %d, %Y')}

---

## Abstract

This project applies unsupervised machine learning techniques to explore naturally
occurring patterns in satellite-derived cloud and weather observation data. Using
Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE),
and K-Means Clustering, we identify potential cloud-pattern regimes without imposing
predefined meteorological categories. The analysis is performed on {len(raw_df):,}
weather observations containing {len(original_df.columns)} numerical features,
covering 15 weather stations across the Indian subcontinent.

## 1. Introduction

Weather pattern analysis is fundamental to meteorology, climate science, and disaster
preparedness. Satellite-derived cloud data provides a rich source of atmospheric
information, but the high dimensionality and complexity of this data makes manual
analysis challenging. Unsupervised machine learning offers a data-driven approach to
discover natural groupings in weather observations without requiring pre-labeled data.

This project investigates whether meaningful cloud-pattern clusters can be identified
from meteorological features using dimensionality reduction and clustering techniques.

## 2. Problem Statement

Given a dataset of satellite-validated weather station observations with multiple
meteorological features (temperature, humidity, pressure, wind, cloud cover, visibility,
etc.), can we:

1. Reduce dimensionality while preserving information?
2. Discover naturally occurring groups in the feature space?
3. Interpret these groups as potential weather/cloud-pattern regimes?

## 3. Dataset Description

- **Name:** Cloud Weather Observations (Representative NOAA LCD Data)
- **Records:** {len(raw_df):,}
- **Features:** {len(raw_df.columns)} total ({len(original_df.columns)} numerical)
- **Coverage:** 15 Indian weather stations, Jan 2023 – Dec 2024
- **Source Basis:** NOAA Local Climatological Data (LCD) statistical distributions

**Note:** This dataset is generated using documented statistical properties of NOAA LCD
observations. It preserves realistic meteorological distributions and inter-variable
physical constraints. The pipeline accepts real NOAA CSV or INSAT data for substitution.

### Features Used:
{chr(10).join(f'- {col}' for col in original_df.columns)}

## 4. Methodology

### 4.1 Data Preprocessing
- Duplicate removal ({raw_df.duplicated().sum()} duplicates found and removed)
- Missing value imputation using median strategy
- Feature selection ({len(original_df.columns)} numerical features)
- StandardScaler normalization (mean=0, std=1)

### 4.2 Principal Component Analysis (PCA)
- Fitted with {len(evr)} components
- First component explains {evr[0]*100:.1f}% of variance
- {pca_summary['optimal_components_95']} components needed for 95% variance retention

### 4.3 t-SNE Analysis
- Non-linear 2D projection for visualization
- Perplexity experimentation (5, 15, 30, 50)
- PCA preprocessing applied before t-SNE

### 4.4 K-Means Clustering
- Applied on PCA-reduced features (5 components)
- Optimal K selection via Elbow Method and Silhouette Score
- Cluster analysis with centroid profiling

## 5. Results

### 5.1 PCA Results
- PC1 explains {evr[0]*100:.1f}% of variance
- PC1+PC2 explain {cumvar[1]*100:.1f}% of variance
- {pca_summary['optimal_components_95']} components capture 95% of total variance
- Significant dimensionality reduction from {len(original_df.columns)} to {pca_summary['optimal_components_95']} features

### 5.2 K-Means Results (K={cluster_analysis['n_clusters']})
- Silhouette Score: {cluster_analysis['silhouette_score']:.4f}
- Inertia: {cluster_analysis['inertia']:.1f}
- Cluster sizes: {', '.join(f'{k}: {v}' for k, v in cluster_analysis['sizes'].items())}

## 6. Limitations

1. This analysis is exploratory — clusters are mathematical groupings, not confirmed cloud types.
2. The dataset represents surface weather stations, not direct satellite pixel observations.
3. K-Means assumes spherical clusters which may not match actual weather pattern geometry.
4. t-SNE distances between clusters are NOT meaningful.
5. Results may differ significantly with different dataset subsets or time periods.
6. Independent meteorological validation is required for any cloud type interpretations.

## 7. Ethical Considerations

- This application is for educational/exploratory purposes only, NOT operational forecasting.
- The dataset may underrepresent certain geographical regions or weather conditions.
- Cluster assignments should never be used for safety-critical weather decisions.

## 8. Conclusion

This project demonstrates the application of unsupervised machine learning to
satellite-derived weather data exploration. PCA effectively reduces dimensionality while
retaining most variance. K-Means discovers natural groupings that may correspond to
distinct weather regimes. t-SNE provides useful visualization of local data structure.

However, the interpretation of clusters as specific cloud types requires independent
meteorological validation. The project provides a configurable pipeline that can be
applied to actual INSAT satellite data when it becomes available.

## 9. Future Scope

1. Integration with actual INSAT-3D/3DR satellite data from MOSDAC
2. Application of DBSCAN and Gaussian Mixture Models
3. Temporal pattern evolution analysis
4. Geographic clustering for region-specific weather regimes
5. UMAP implementation as a modern t-SNE alternative
6. Real-time data ingestion from weather APIs
7. Cloud deployment for wider accessibility

## 10. References

1. Jolliffe, I. T., & Cadima, J. (2016). Principal Component Analysis: A Review.
2. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE.
3. Lloyd, S. P. (1982). Least Squares Quantization in PCM.
4. Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid.
5. NOAA NCEI. Local Climatological Data (LCD).
6. ISRO MOSDAC. INSAT-3D Satellite Data Products.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
8. Wattenberg, M., et al. (2016). How to Use t-SNE Effectively.
"""
    return report

