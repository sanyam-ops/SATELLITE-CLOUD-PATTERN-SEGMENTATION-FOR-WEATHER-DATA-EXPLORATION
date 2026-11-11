"""
Page 7: Comparative Analysis
============================
Side-by-side comparison of PCA vs t-SNE and clustering across K values.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pca_analysis import PCAAnalyzer
from src.tsne_analysis import TSNEAnalyzer
from src.clustering import KMeansClusterer
from src.evaluation import compare_dimensionality_reduction, compare_clustering_configs
from config import VIZ_CONFIG, PCA_CONFIG, TSNE_CONFIG, KMEANS_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


@st.cache_data
def compute_comparison_clusters(pca_data, original_df, k_values=(2, 3, 4, 5, 6, 7, 8)):
    cluster_results = {}
    cluster_labels = {}
    for k in k_values:
        clusterer = KMeansClusterer(random_state=KMEANS_CONFIG["random_state"])
        labels = clusterer.fit_predict(pca_data, n_clusters=k)
        analysis = clusterer.get_cluster_analysis(
            pca_data, labels,
            feature_names=list(original_df.columns),
            original_data=original_df,
        )
        cluster_results[k] = analysis
        cluster_labels[k] = labels
    return cluster_results, cluster_labels


def render_comparison(scaled_df, original_df):
    """Render the Comparative Analysis page."""

    st.markdown("""
    <h1>📈 Comparative Analysis</h1>
    <p style='color: #8892A0;'>Compare dimensionality reduction techniques and clustering configurations.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔄 PCA vs t-SNE", "🎯 Clustering Comparison", "📊 Summary"])

    # ── TAB 1: PCA vs t-SNE ───────────────────────────────
    with tabs[0]:
        st.markdown("### PCA vs t-SNE: Side-by-Side Comparison")

        # PCA
        pca_analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
        pca_full = pca_analyzer.fit_transform(scaled_df)
        pca_summary = pca_analyzer.get_pca_summary()
        pca_2d = pca_full[:, :2]
        evr = pca_analyzer.pca_model.explained_variance_ratio_

        # t-SNE (use cached result if available, else run with sample)
        tsne_analyzer = TSNEAnalyzer(random_state=TSNE_CONFIG["random_state"])
        sample_size = min(3000, len(scaled_df))

        if "tsne_result" in st.session_state:
            tsne_result = st.session_state["tsne_result"]
        else:
            with st.spinner("Running t-SNE for comparison (sample of 3000 points)..."):
                tsne_result = tsne_analyzer.run_tsne(
                    scaled_df.values,
                    perplexity=30, learning_rate=200, n_iter=1000,
                    sample_size=sample_size if sample_size < len(scaled_df) else None,
                )

        # Side-by-side plots
        col1, col2 = st.columns(2)

        with col1:
            fig_pca = px.scatter(
                x=pca_2d[:, 0], y=pca_2d[:, 1],
                labels={"x": f"PC1 ({evr[0]*100:.1f}%)", "y": f"PC2 ({evr[1]*100:.1f}%)"},
                template=TEMPLATE,
                title="PCA 2D Projection",
                opacity=0.4,
                color_discrete_sequence=[COLORS[0]],
            )
            fig_pca.update_traces(marker=dict(size=2))
            fig_pca.update_layout(height=450)
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            emb = tsne_result["embedding"]
            fig_tsne = px.scatter(
                x=emb[:, 0], y=emb[:, 1],
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                template=TEMPLATE,
                title=f"t-SNE 2D Projection (n={tsne_result['n_samples']:,})",
                opacity=0.4,
                color_discrete_sequence=[COLORS[3]],
            )
            fig_tsne.update_traces(marker=dict(size=2))
            fig_tsne.update_layout(height=450)
            st.plotly_chart(fig_tsne, use_container_width=True)

        # Comparison table
        comparison = compare_dimensionality_reduction(pca_summary, tsne_result)

        st.markdown("### 📋 Method Comparison Table")
        comp_rows = []
        criteria = [
            ("Type", "type"),
            ("Preserves Global Structure", "preserves_global_structure"),
            ("Preserves Local Structure", "preserves_local_structure"),
            ("Information Retention", "information_retention"),
            ("Interpretable Components", "interpretable_components"),
            ("Deterministic", "deterministic"),
            ("Scalability", "scalability"),
            ("Use Case", "use_case"),
        ]
        for label, key in criteria:
            pca_val = comparison["PCA"].get(key, "—")
            tsne_val = comparison["t-SNE"].get(key, "—")
            if isinstance(pca_val, bool):
                pca_val = "✅ Yes" if pca_val else "❌ No"
                tsne_val = "✅ Yes" if tsne_val else "❌ No"
            comp_rows.append({"Criterion": label, "PCA": str(pca_val), "t-SNE": str(tsne_val)})

        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Limitations comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### PCA Limitations")
            for lim in comparison["PCA"]["limitations"]:
                st.markdown(f"- {lim}")
        with col2:
            st.markdown("#### t-SNE Limitations")
            for lim in comparison["t-SNE"]["limitations"]:
                st.markdown(f"- {lim}")

    # ── TAB 2: Clustering Comparison ──────────────────────
    with tabs[1]:
        st.markdown("### Clustering Results Across K Values")

        n_pca = st.slider("PCA components for clustering:", 2, min(10, len(scaled_df.columns)), 5, key="comp_pca")
        pca_clust = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
        pca_clust_data = pca_clust.fit_transform(scaled_df, n_components=n_pca)

        k_values = [2, 3, 4, 5, 6, 7, 8]
        with st.spinner("Computing clusters for multiple K values..."):
            cluster_results, cluster_labels = compute_comparison_clusters(pca_clust_data, original_df, tuple(k_values))

        # Comparison table
        comp_table = compare_clustering_configs(cluster_results)
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        # Visual grid: 2x2 for K=2,3,4,5
        st.markdown("### Cluster Visualizations")
        pca_2d_data = pca_clust_data[:, :2]

        for row_start in [2, 4, 6]:
            cols = st.columns(2)
            for col_idx, k in enumerate([row_start, row_start + 1]):
                if k in cluster_results:
                    with cols[col_idx]:
                        lbls = cluster_labels[k]
                        cl_labels = [f"C{l}" for l in lbls]
                        fig = px.scatter(
                            x=pca_2d_data[:, 0], y=pca_2d_data[:, 1],
                            color=cl_labels,
                            template=TEMPLATE,
                            title=f"K={k} (Sil: {cluster_results[k]['silhouette_score']:.3f})",
                            color_discrete_sequence=COLORS,
                            opacity=0.5,
                        )
                        fig.update_traces(marker=dict(size=3))
                        fig.update_layout(height=350, showlegend=True,
                                          xaxis_title="PC1", yaxis_title="PC2",
                                          legend=dict(font=dict(size=9)))
                        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Summary ────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📊 Analysis Summary")

        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #00F0FF;'>When to Use PCA</h4>
            <p style='color: #CBD5E1;'>
                PCA is the preferred choice when you need <b>interpretable</b> dimensionality reduction,
                want to understand <b>which features contribute most</b> to variance, need a
                <b>deterministic</b> result, or plan to use the reduced data for <b>downstream ML tasks</b>
                like clustering.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #A855F7;'>When to Use t-SNE</h4>
            <p style='color: #CBD5E1;'>
                t-SNE excels at <b>visualization</b> of high-dimensional data, especially when you want
                to explore <b>local neighborhood structures</b> and visually identify potential clusters.
                However, it should <b>not</b> be used for quantitative analysis or as input to other algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='accent-card'>
            <h4 style='color: #10B981;'>Recommended Workflow</h4>
            <p style='color: #CBD5E1;'>
                Use <b>PCA for dimensionality reduction</b> (retaining 90-95% variance), then apply
                <b>K-Means on PCA-reduced features</b>. Use <b>t-SNE for visualization</b> only,
                coloring points by their K-Means cluster assignments to visually assess cluster quality.
            </p>
        </div>
        """, unsafe_allow_html=True)

