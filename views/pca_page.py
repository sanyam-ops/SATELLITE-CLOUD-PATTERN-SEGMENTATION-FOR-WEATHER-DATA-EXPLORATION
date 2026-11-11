"""
Page 4: PCA Analysis
====================
Explained variance, scree plot, component loadings, 2D/3D scatter plots.
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
from config import VIZ_CONFIG, PCA_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


@st.cache_data
def run_pca_full(_scaled_df):
    """Run PCA with all components (cached)."""
    analyzer = PCAAnalyzer(random_state=PCA_CONFIG["random_state"])
    transformed = analyzer.fit_transform(_scaled_df)
    summary = analyzer.get_pca_summary()
    loadings = analyzer.get_component_loadings()
    return analyzer, transformed, summary, loadings


def render_pca(scaled_df, original_df):
    """Render the PCA Analysis page."""

    st.markdown("""
    <h1>📊 Principal Component Analysis</h1>
    <p style='color: #8892A0;'>Linear dimensionality reduction — variance analysis, component loadings, and visualization.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # Run PCA
    analyzer, transformed, summary, loadings = run_pca_full(scaled_df)
    var_info = summary["variance"]

    # ── Controls ───────────────────────────────────────────
    max_comp = min(len(scaled_df.columns), 10)
    n_components = st.slider(
        "Number of PCA components to analyze:",
        min_value=2, max_value=max_comp, value=min(5, max_comp),
    )

    # ── Key Metrics ────────────────────────────────────────
    st.markdown("### 📈 Variance Summary")
    cum_var = var_info["cumulative_variance"]
    c1, c2, c3, c4 = st.columns(4)
    var_at_n = cum_var[n_components - 1] if n_components <= len(cum_var) else cum_var[-1]
    c1.metric("Components Selected", n_components)
    c2.metric("Variance Retained", f"{var_at_n*100:.1f}%")
    c3.metric("95% Threshold", f"{summary['optimal_components_95']} PCs")
    c4.metric("Original Features", f"{summary['n_original_features']}")

    # ── Tabs ───────────────────────────────────────────────
    tabs = st.tabs(["📉 Variance Plots", "🔬 2D Scatter", "🌐 3D Scatter", "📋 Loadings", "⭐ Top Features"])

    # TAB 1: Variance Plots
    with tabs[0]:
        evr = var_info["explained_variance_ratio"]
        cumvar = var_info["cumulative_variance"]
        pc_labels = [f"PC{i+1}" for i in range(len(evr))]

        col1, col2 = st.columns(2)

        with col1:
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pc_labels, y=[v * 100 for v in evr],
                marker_color=COLORS[0], name="Individual",
                text=[f"{v*100:.1f}%" for v in evr], textposition="outside",
            ))
            fig.update_layout(
                title="Scree Plot — Explained Variance Ratio",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance (%)",
                template=TEMPLATE, height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cumulative variance
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pc_labels, y=[v * 100 for v in cumvar],
                mode="lines+markers", marker_color=COLORS[1],
                name="Cumulative", line=dict(width=3),
            ))
            # Threshold lines (70-20-10 palette)
            for thresh, color, label in [(95, "#F43F5E", "95%"), (90, "#FBBF24", "90%"), (80, "#10B981", "80%")]:
                fig2.add_hline(y=thresh, line_dash="dash", line_color=color,
                               annotation_text=label, annotation_position="bottom right")
            fig2.update_layout(
                title="Cumulative Explained Variance",
                xaxis_title="Principal Component",
                yaxis_title="Cumulative Variance (%)",
                template=TEMPLATE, height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info(
            f"💡 **{summary['optimal_components_95']}** components capture 95% of variance. "
            f"The first 2 components alone capture **{cumvar[1]*100:.1f}%**."
        )

    # TAB 2: 2D PCA Scatter
    with tabs[1]:
        st.markdown("### 2D PCA Visualization")
        pca_2d = transformed[:, :2]
        fig = px.scatter(
            x=pca_2d[:, 0], y=pca_2d[:, 1],
            labels={"x": f"PC1 ({evr[0]*100:.1f}%)", "y": f"PC2 ({evr[1]*100:.1f}%)"},
            template=TEMPLATE,
            title="PCA 2D Projection",
            opacity=0.5,
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"*PC1 and PC2 together explain **{cumvar[1]*100:.1f}%** of the total variance.*"
        )

    # TAB 3: 3D PCA Scatter
    with tabs[2]:
        if len(evr) >= 3:
            st.markdown("### 3D PCA Visualization")
            pca_3d = transformed[:, :3]
            fig = px.scatter_3d(
                x=pca_3d[:, 0], y=pca_3d[:, 1], z=pca_3d[:, 2],
                labels={
                    "x": f"PC1 ({evr[0]*100:.1f}%)",
                    "y": f"PC2 ({evr[1]*100:.1f}%)",
                    "z": f"PC3 ({evr[2]*100:.1f}%)",
                },
                template=TEMPLATE,
                title="PCA 3D Projection",
                opacity=0.4,
                color_discrete_sequence=[COLORS[2]],
            )
            fig.update_traces(marker=dict(size=2))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"*Three components explain **{cumvar[2]*100:.1f}%** of the total variance.*"
            )
        else:
            st.info("Need at least 3 components for 3D visualization.")

    # TAB 4: Component Loadings
    with tabs[3]:
        st.markdown("### Component Loadings Heatmap")
        load_display = loadings.iloc[:, :n_components]
        fig = px.imshow(
            load_display.round(3),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template=TEMPLATE,
            title="PCA Component Loadings (Feature Contributions)",
            aspect="auto",
        )
        fig.update_layout(height=max(400, len(loadings) * 30))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='accent-card'>
            <p style='color: #B0B8C8;'>
                <b>How to read:</b> Each cell shows how much an original feature contributes to a
                principal component. <b style='color: #FF6B6B;'>Red = positive loading</b>,
                <b style='color: #00D4FF;'>Blue = negative loading</b>. Larger absolute values
                indicate stronger contributions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 Raw Loadings Table"):
            st.dataframe(load_display.round(4), use_container_width=True)

    # TAB 5: Top Features
    with tabs[4]:
        st.markdown("### Top Contributing Features per Component")
        top_features = summary["top_features"]

        for pc_name, info in list(top_features.items())[:n_components]:
            st.markdown(f"#### {pc_name}")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=info["features"],
                y=info["loadings"],
                marker_color=[COLORS[i % len(COLORS)] for i in range(len(info["features"]))],
                text=[f"{v:.3f}" for v in info["loadings"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Top Features for {pc_name}",
                xaxis_title="Feature", yaxis_title="Loading",
                template=TEMPLATE, height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

