"""
Page 3: Preprocessing
=====================
Missing-value handling, feature selection, encoding, scaling, before-and-after comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import VIZ_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


def render_preprocessing(raw_df, scaled_df, original_df, metadata_df, preprocessor):
    """Render the Preprocessing page."""

    st.markdown("""
    <h1>⚙️ Data Preprocessing</h1>
    <p style='color: #8892A0;'>Cleaning, encoding, scaling, and feature selection pipeline.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    before, after = preprocessor.get_before_after_stats()

    # ── Before / After Summary ─────────────────────────────
    st.markdown("### 📊 Before vs After Preprocessing")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h4 style='color: #F43F5E; font-weight: 700;'>🔴 Raw Ingest State</h4>
        </div>
        """, unsafe_allow_html=True)
        if before:
            ba, bb, bc, bd = st.columns(4)
            ba.metric("Records", f"{before['n_records']:,}")
            bb.metric("Features", f"{before['n_features']}")
            bc.metric("Missing", f"{before['total_missing']:,}")
            bd.metric("Duplicates", f"{before['duplicates']}")

    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h4 style='color: #10B981; font-weight: 700;'>🟢 Clean Preprocessed State</h4>
        </div>
        """, unsafe_allow_html=True)
        if after:
            aa, ab, ac, ad = st.columns(4)
            aa.metric("Records", f"{after['n_records']:,}")
            ab.metric("Features", f"{after['n_features']}")
            ac.metric("Missing", f"{after['total_missing']}")
            ad.metric("Duplicates", f"{after['duplicates']}")

    # ── Tabs ───────────────────────────────────────────────
    tabs = st.tabs(["❓ Missing Values", "📐 Feature Selection", "📏 Scaling", "📋 Outliers", "📝 Pipeline Log"])

    # TAB 1: Missing Values
    with tabs[0]:
        st.markdown("### Missing Value Handling")
        st.markdown("""
        <div class='accent-card'>
            <p style='color: #B0B8C8;'>
                <b>Strategy:</b> Numerical features imputed using <b style='color: #00D4FF;'>median</b> values
                (robust to outliers). Categorical features imputed with <b style='color: #4ECDC4;'>most frequent</b> values.
                Columns with >50% missing are dropped entirely.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Before missing values
        missing_before = raw_df.isnull().sum()
        missing_before = missing_before[missing_before > 0].sort_values(ascending=False)
        if len(missing_before) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=missing_before.index.tolist(),
                y=(missing_before / len(raw_df) * 100).values,
                name="Before",
                marker_color="#FF6B6B",
                opacity=0.8,
            ))
            fig.add_trace(go.Bar(
                x=missing_before.index.tolist(),
                y=[0] * len(missing_before),
                name="After",
                marker_color="#4ECDC4",
                opacity=0.8,
            ))
            fig.update_layout(
                title="Missing Values: Before vs After (%)",
                barmode="group", template=TEMPLATE, height=400,
                yaxis_title="Missing %",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ All missing values resolved. {after['n_records']:,} clean records retained.")

    # TAB 2: Feature Selection
    with tabs[1]:
        st.markdown("### Feature Selection")
        if after and "features_used" in after:
            features_used = after["features_used"]
            st.markdown(f"**{len(features_used)}** numerical features selected for ML analysis:")

            feat_df = pd.DataFrame({
                "Feature": features_used,
                "Status": ["✅ Selected"] * len(features_used),
            })
            st.dataframe(feat_df, use_container_width=True)

            # Show excluded features
            all_cols = set(raw_df.columns)
            excluded = all_cols - set(features_used)
            if excluded:
                st.markdown("**Excluded columns** (metadata, categorical, or constant):")
                exc_df = pd.DataFrame({
                    "Feature": sorted(excluded),
                    "Reason": ["Metadata / Categorical / Not selected"] * len(excluded),
                })
                st.dataframe(exc_df, use_container_width=True)

        if not metadata_df.empty:
            st.markdown("### Preserved Metadata")
            st.info(f"Metadata columns preserved separately: {list(metadata_df.columns)}")
            st.dataframe(metadata_df.head(10), use_container_width=True)

    # TAB 3: Scaling
    with tabs[2]:
        st.markdown("### StandardScaler Normalization")
        st.markdown("""
        <div class='accent-card'>
            <p style='color: #B0B8C8;'>
                <b>StandardScaler</b> transforms each feature to have <b style='color: #00D4FF;'>mean = 0</b>
                and <b style='color: #4ECDC4;'>standard deviation = 1</b>. This ensures all features
                contribute equally to distance-based algorithms (PCA, t-SNE, K-Means).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Select features to compare
        common_cols = [c for c in original_df.columns if c in scaled_df.columns]
        sel = st.multiselect("Compare features:", common_cols, default=common_cols[:4])

        if sel:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Before Scaling", "After Scaling"])
            for idx, feat in enumerate(sel):
                color = COLORS[idx % len(COLORS)]
                fig.add_trace(
                    go.Box(y=original_df[feat].values, name=feat, marker_color=color, showlegend=True),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Box(y=scaled_df[feat].values, name=feat, marker_color=color, showlegend=False),
                    row=1, col=2,
                )
            fig.update_layout(height=450, template=TEMPLATE, title_text="Feature Scaling Comparison")
            st.plotly_chart(fig, use_container_width=True)

        # Show scaled statistics
        st.markdown("#### Scaled Feature Statistics")
        st.dataframe(scaled_df.describe().round(4), use_container_width=True)

    # TAB 4: Outliers
    with tabs[3]:
        st.markdown("### Outlier Detection (IQR Method)")
        from src.preprocessing import DataPreprocessor
        temp_pp = DataPreprocessor()
        outlier_df = temp_pp.detect_outliers(original_df)
        if len(outlier_df) > 0:
            st.dataframe(outlier_df, use_container_width=True)

            # Visualize outlier percentages
            fig = px.bar(
                outlier_df.sort_values("Outlier_Pct", ascending=False),
                x="Feature", y="Outlier_Pct",
                color="Outlier_Pct",
                color_continuous_scale=["#4ECDC4", "#FF6B6B"],
                template=TEMPLATE,
                title="Outlier Percentage by Feature",
                labels={"Outlier_Pct": "Outlier %"},
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "💡 Outliers are **not removed** in this analysis. Weather data naturally contains "
                "extreme values (e.g., very high winds during storms). Removing them could bias results."
            )

    # TAB 5: Pipeline Log
    with tabs[4]:
        st.markdown("### Preprocessing Pipeline Log")
        if preprocessor.preprocessing_log:
            for i, entry in enumerate(preprocessor.preprocessing_log, 1):
                st.markdown(f"**Step {i}:** {entry}")
        else:
            st.info("No preprocessing log available.")

        st.markdown("### Data Leakage Prevention")
        st.markdown("""
        <div class='custom-card'>
            <ul style='color: #B0B8C8;'>
                <li>StandardScaler fitted on the <b>full training set</b> (unsupervised learning — no train/test split needed)</li>
                <li>Metadata (timestamps, coordinates) preserved <b>separately</b> from features</li>
                <li>No target variable used in preprocessing decisions</li>
                <li>Feature selection based on data type and availability, not outcome</li>
                <li>Imputation strategy (median) chosen for robustness, not to optimize any metric</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

