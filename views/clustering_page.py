"""
Page 6: K-Means Clustering
==========================
Elbow method, silhouette scores, cluster visualization, centroid analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.clustering import KMeansClusterer
from src.pca_analysis import PCAAnalyzer
from config import VIZ_CONFIG, KMEANS_CONFIG, PCA_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


@st.cache_data
def compute_elbow(pca_values, k_range):
    clusterer = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
    return clusterer.compute_elbow(pca_values, k_range=k_range)


@st.cache_data
def compute_silhouettes(pca_values, k_range):
    clusterer = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
    return clusterer.compute_silhouette_scores(pca_values, k_range=k_range)


def render_clustering(scaled_df, original_df, metadata_df):
    """Render the K-Means Clustering page."""

    st.markdown("""
    <h1>🎯 K-Means Clustering</h1>
    <p style='color: #8892A0;'>Discover natural groupings in cloud/weather data using K-Means on PCA-reduced features.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # ── Controls ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters (K):", 2, 15, KMEANS_CONFIG["default_k"])
    with col2:
        max_pca = min(len(scaled_df.columns), 10)
        n_pca_comp = st.slider("PCA Components for Clustering:", 2, max_pca, min(5, max_pca))

    # Apply PCA first
    pca_analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
    pca_data = pca_analyzer.fit_transform(scaled_df, n_components=n_pca_comp)

    # ── Tabs ───────────────────────────────────────────────
    tabs = st.tabs(["📉 Elbow & Silhouette", "🎨 Cluster Visualization", "📊 Cluster Analysis", "📖 Interpretation"])

    # TAB 1: Elbow & Silhouette
    with tabs[0]:
        st.markdown("### Optimal K Selection")

        with st.spinner("Computing Elbow Curve and Silhouette Scores..."):
            elbow_result = compute_elbow(pca_data, k_range=(2, 12))
            sil_result = compute_silhouettes(pca_data, k_range=(2, 12))

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=elbow_result["k_values"],
                y=elbow_result["inertias"],
                mode="lines+markers",
                marker=dict(size=10, color=COLORS[0]),
                line=dict(width=3, color=COLORS[0]),
                name="Inertia",
            ))
            # Mark suggested elbow
            elbow_k = elbow_result["suggested_elbow"]
            elbow_idx = elbow_result["k_values"].index(elbow_k)
            fig.add_trace(go.Scatter(
                x=[elbow_k], y=[elbow_result["inertias"][elbow_idx]],
                mode="markers", marker=dict(size=18, color="#F43F5E", symbol="star"),
                name=f"Suggested Elbow (K={elbow_k})",
            ))
            fig.update_layout(
                title="Elbow Method",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Inertia (WCSS)",
                template=TEMPLATE, height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sil_result["k_values"],
                y=sil_result["silhouette_scores"],
                mode="lines+markers",
                marker=dict(size=10, color=COLORS[1]),
                line=dict(width=3, color=COLORS[1]),
                name="Silhouette",
            ))
            best_k = sil_result["best_k"]
            best_idx = sil_result["k_values"].index(best_k)
            fig2.add_trace(go.Scatter(
                x=[best_k], y=[sil_result["silhouette_scores"][best_idx]],
                mode="markers", marker=dict(size=18, color="#F43F5E", symbol="star"),
                name=f"Best K={best_k} ({sil_result['best_score']:.3f})",
            ))
            fig2.update_layout(
                title="Silhouette Score",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score",
                template=TEMPLATE, height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info(
            f"📊 **Elbow Method** suggests K={elbow_k}. "
            f"**Silhouette Score** is highest at K={best_k} ({sil_result['best_score']:.3f})."
        )

    # TAB 2: Cluster Visualization
    with tabs[1]:
        st.markdown("### Cluster Visualization")

        clusterer = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
        labels = clusterer.fit_predict(pca_data, n_clusters=n_clusters)

        # 2D PCA scatter colored by cluster
        pca_2d_analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
        pca_2d = pca_2d_analyzer.fit_transform(scaled_df, n_components=2)
        evr = pca_2d_analyzer.pca_model.explained_variance_ratio_

        cluster_labels = [f"Cluster {l}" for l in labels]
        fig = px.scatter(
            x=pca_2d[:, 0], y=pca_2d[:, 1],
            color=cluster_labels,
            labels={
                "x": f"PC1 ({evr[0]*100:.1f}%)",
                "y": f"PC2 ({evr[1]*100:.1f}%)",
                "color": "Cluster",
            },
            template=TEMPLATE,
            title=f"K-Means Clusters (K={n_clusters}) on PCA 2D",
            color_discrete_sequence=COLORS,
            opacity=0.6,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        # 3D Visualization
        if n_pca_comp >= 3:
            pca_3d_analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
            pca_3d = pca_3d_analyzer.fit_transform(scaled_df, n_components=3)
            evr3 = pca_3d_analyzer.pca_model.explained_variance_ratio_

            fig3d = px.scatter_3d(
                x=pca_3d[:, 0], y=pca_3d[:, 1], z=pca_3d[:, 2],
                color=cluster_labels,
                labels={
                    "x": f"PC1 ({evr3[0]*100:.1f}%)",
                    "y": f"PC2 ({evr3[1]*100:.1f}%)",
                    "z": f"PC3 ({evr3[2]*100:.1f}%)",
                    "color": "Cluster",
                },
                template=TEMPLATE,
                title=f"K-Means Clusters (K={n_clusters}) — 3D PCA",
                color_discrete_sequence=COLORS,
                opacity=0.5,
            )
            fig3d.update_traces(marker=dict(size=2))
            fig3d.update_layout(height=600)
            st.plotly_chart(fig3d, use_container_width=True)

        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        size_df = pd.DataFrame({"Cluster": [f"Cluster {u}" for u in unique], "Size": counts})
        fig_size = px.bar(
            size_df, x="Cluster", y="Size",
            color="Cluster", color_discrete_sequence=COLORS,
            template=TEMPLATE, title="Cluster Size Distribution",
            text="Size",
        )
        fig_size.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)

    # TAB 3: Cluster Analysis
    with tabs[2]:
        st.markdown("### Cluster Centroid Analysis")

        clusterer2 = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
        labels2 = clusterer2.fit_predict(pca_data, n_clusters=n_clusters)

        analysis = clusterer2.get_cluster_analysis(
            pca_data, labels2,
            feature_names=list(original_df.columns),
            original_data=original_df,
        )

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", analysis["n_clusters"])
        c2.metric("Silhouette Score", f"{analysis['silhouette_score']:.4f}" if analysis["silhouette_score"] else "N/A")
        c3.metric("Inertia", f"{analysis['inertia']:.1f}" if analysis["inertia"] else "N/A")

        # Cluster sizes
        st.markdown("#### Cluster Sizes")
        size_data = pd.DataFrame({
            "Cluster": list(analysis["sizes"].keys()),
            "Count": list(analysis["sizes"].values()),
            "Percentage": [f"{v}%" for v in analysis["size_percentages"].values()],
        })
        st.dataframe(size_data, use_container_width=True)

        # Per-cluster silhouette
        if analysis["silhouette_per_cluster"]:
            st.markdown("#### Silhouette Score per Cluster")
            sil_df = pd.DataFrame({
                "Cluster": list(analysis["silhouette_per_cluster"].keys()),
                "Score": list(analysis["silhouette_per_cluster"].values()),
            })
            fig = px.bar(
                sil_df, x="Cluster", y="Score",
                color="Score", color_continuous_scale=["#FF6B6B", "#4ECDC4"],
                template=TEMPLATE, title="Silhouette Score per Cluster",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Centroid profiles (radar chart)
        if analysis["centroids"]:
            st.markdown("#### Feature Profiles by Cluster")
            # Format dataframe where index is features, columns are clusters
            raw_centroids = pd.DataFrame(analysis["centroids"]).T
            raw_centroids.columns = [f"Cluster {c}" for c in raw_centroids.columns]

            # Normalize centroids for radar chart so features on different physical scales can be compared
            if not raw_centroids.empty and len(raw_centroids.columns) > 0:
                features_to_show = list(raw_centroids.index)[:8]
                subset = raw_centroids.loc[features_to_show]
                # Min-Max normalize each feature across clusters
                norm_subset = subset.apply(
                    lambda row: (row - row.min()) / (row.max() - row.min() + 1e-6)
                    if row.max() > row.min() else row * 0 + 0.5,
                    axis=1,
                )

                fig = go.Figure()
                for col_idx, cluster_col in enumerate(norm_subset.columns):
                    values = norm_subset[cluster_col].values.tolist()
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=features_to_show + [features_to_show[0]],
                        fill="toself",
                        name=str(cluster_col),
                        line_color=COLORS[col_idx % len(COLORS)],
                        opacity=0.4,
                    ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8, color="#8892A0")),
                        bgcolor="#1A1F2E",
                    ),
                    template=TEMPLATE, height=500,
                    title="Relative Feature Profiles by Cluster (Normalized 0–1)",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 Actual Centroid Feature Values (Unscaled)"):
                    st.dataframe(raw_centroids.round(2), use_container_width=True)

    # TAB 4: Interpretation
    with tabs[3]:
        st.markdown("### 📖 Interpreting K-Means Results")
        guidelines = clusterer.get_interpretation_guidelines() if 'clusterer' in dir() else KMeansClusterer().get_interpretation_guidelines()

        st.markdown(f"""
        <div class='accent-card'>
            <h4 style='color: #00D4FF;'>What Do These Clusters Mean?</h4>
            <p style='color: #B0B8C8;'>{guidelines['cluster_meaning']}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='custom-card'>
                <h4 style='color: #4ECDC4;'>Elbow Method</h4>
                <p style='color: #B0B8C8;'>{guidelines['elbow_method']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='custom-card'>
                <h4 style='color: #FFE66D;'>Silhouette Score</h4>
                <p style='color: #B0B8C8;'>{guidelines['silhouette_score']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### ⚠️ Limitations")
        for lim in guidelines["limitations"]:
            st.markdown(f"- {lim}")

        st.warning(guidelines["validation_note"])

