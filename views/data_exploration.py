"""
Page 2: Data Exploration
========================
Raw data preview, feature descriptions, missing values, correlations, and distributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_analysis import (
    compute_correlation_matrix,
    compute_feature_statistics,
    get_feature_descriptions,
    identify_highly_correlated,
)
from config import VIZ_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


def render_data_exploration(df):
    """Render the Data Exploration page."""

    st.markdown("""
    <h1>🔍 Data Exploration</h1>
    <p style='color: #8892A0;'>Explore the raw dataset, feature distributions, correlations, and geographic coverage.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # ── Tab Layout ──────────────────────────────────────────
    tabs = st.tabs([
        "📋 Raw Data", "📖 Features", "❓ Missing Values",
        "🔗 Correlations", "📊 Distributions", "🌍 Geographic",
    ])

    # ── TAB 1: Raw Data ────────────────────────────────────
    with tabs[0]:
        st.markdown("### Raw Dataset Preview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Features", f"{len(df.columns)}")
        c3.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
        c4.metric("Duplicates", f"{df.duplicated().sum()}")

        st.dataframe(df.head(100), use_container_width=True, height=400)

        with st.expander("📊 Quick Statistics"):
            st.dataframe(df.describe().round(2), use_container_width=True)

    # ── TAB 2: Feature Descriptions ────────────────────────
    with tabs[1]:
        st.markdown("### Feature Descriptions")
        descs = get_feature_descriptions()
        feature_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            missing = df[col].isnull().sum()
            desc = descs.get(col, "—")
            feature_info.append({
                "Feature": col,
                "Type": dtype,
                "Unique": nunique,
                "Missing": missing,
                "Description": desc,
            })
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True, height=500)

        st.markdown("### Feature Statistics")
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) > 0:
            stats = compute_feature_statistics(df)
            st.dataframe(stats.round(3), use_container_width=True)

    # ── TAB 3: Missing Values ──────────────────────────────
    with tabs[2]:
        st.markdown("### Missing Value Analysis")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Feature": missing_counts.index,
            "Count": missing_counts.values,
            "Percentage": missing_pct.values,
        }).sort_values("Count", ascending=False)

        has_missing = missing_df[missing_df["Count"] > 0]

        if len(has_missing) > 0:
            fig = px.bar(
                has_missing, x="Feature", y="Percentage",
                color="Percentage",
                color_continuous_scale=["#4ECDC4", "#FF6B6B"],
                template=TEMPLATE,
                title="Missing Values by Feature (%)",
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(has_missing, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")

        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        st.info(f"Total missing cells: **{total_missing:,}** out of **{total_cells:,}** ({total_missing/total_cells*100:.2f}%)")

    # ── TAB 4: Correlations ────────────────────────────────
    with tabs[3]:
        st.markdown("### Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            corr = compute_correlation_matrix(df)
            fig = px.imshow(
                corr.round(2),
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                template=TEMPLATE,
                title="Pearson Correlation Matrix",
                aspect="auto",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Highly correlated pairs
            high_corr = identify_highly_correlated(corr, threshold=0.7)
            if high_corr:
                st.markdown("#### 🔗 Highly Correlated Feature Pairs (|r| ≥ 0.7)")
                hc_df = pd.DataFrame(high_corr, columns=["Feature 1", "Feature 2", "Correlation"])
                st.dataframe(hc_df, use_container_width=True)
        else:
            st.warning("Not enough numerical features for correlation analysis.")

    # ── TAB 5: Distributions ───────────────────────────────
    with tabs[4]:
        st.markdown("### Feature Distributions")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        dist_type = st.radio("Chart type:", ["Histogram", "Box Plot"], horizontal=True)
        selected_features = st.multiselect(
            "Select features:", num_cols,
            default=num_cols[:6] if len(num_cols) >= 6 else num_cols,
        )

        if selected_features:
            if dist_type == "Histogram":
                n_cols_grid = min(3, len(selected_features))
                n_rows_grid = (len(selected_features) + n_cols_grid - 1) // n_cols_grid
                fig = make_subplots(
                    rows=n_rows_grid, cols=n_cols_grid,
                    subplot_titles=selected_features,
                )
                for idx, feat in enumerate(selected_features):
                    r = idx // n_cols_grid + 1
                    c = idx % n_cols_grid + 1
                    data = df[feat].dropna()
                    fig.add_trace(
                        go.Histogram(x=data, name=feat, marker_color=COLORS[idx % len(COLORS)],
                                     opacity=0.8, showlegend=False),
                        row=r, col=c,
                    )
                fig.update_layout(height=300 * n_rows_grid, template=TEMPLATE,
                                  title_text="Feature Distributions")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                for idx, feat in enumerate(selected_features):
                    fig.add_trace(go.Box(
                        y=df[feat].dropna(), name=feat,
                        marker_color=COLORS[idx % len(COLORS)],
                    ))
                fig.update_layout(height=500, template=TEMPLATE,
                                  title="Box Plots of Selected Features")
                st.plotly_chart(fig, use_container_width=True)

        # Categorical distributions
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols:
            st.markdown("### Categorical Feature Distributions")
            sel_cat = st.selectbox("Select categorical feature:", cat_cols)
            if sel_cat:
                val_counts = df[sel_cat].value_counts().head(20)
                fig = px.bar(
                    x=val_counts.index.astype(str), y=val_counts.values,
                    labels={"x": sel_cat, "y": "Count"},
                    color=val_counts.values,
                    color_continuous_scale=["#1A1F2E", "#00D4FF"],
                    template=TEMPLATE,
                    title=f"Distribution of {sel_cat}",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6: Geographic ─────────────────────────────────
    with tabs[5]:
        st.markdown("### Geographic Coverage")
        if "latitude" in df.columns and "longitude" in df.columns:
            color_col = st.selectbox(
                "Color by:", ["temperature", "relative_humidity", "cloud_cover_fraction",
                              "visibility", "station_pressure"],
                index=0,
            )
            color_col = color_col if color_col in df.columns else None

            plot_df = df.dropna(subset=["latitude", "longitude"])
            if color_col:
                plot_df = plot_df.dropna(subset=[color_col])

            # Station summary
            station_avg = plot_df.groupby(["latitude", "longitude"]).agg(
                n_obs=("latitude", "count"),
                **({color_col: (color_col, "mean")} if color_col else {}),
            ).reset_index()

            if hasattr(px, "scatter_map"):
                fig = px.scatter_map(
                    station_avg, lat="latitude", lon="longitude",
                    size="n_obs",
                    color=color_col if color_col else None,
                    color_continuous_scale="Turbo" if color_col else None,
                    zoom=3, center={"lat": 20, "lon": 78},
                    title="Weather Station Locations",
                    hover_data=["n_obs"],
                )
            elif hasattr(px, "scatter_mapbox"):
                fig = px.scatter_mapbox(
                    station_avg, lat="latitude", lon="longitude",
                    size="n_obs",
                    color=color_col if color_col else None,
                    color_continuous_scale="Turbo" if color_col else None,
                    mapbox_style="carto-darkmatter",
                    zoom=3, center={"lat": 20, "lon": 78},
                    title="Weather Station Locations",
                    hover_data=["n_obs"],
                )
            else:
                fig = px.scatter_geo(
                    station_avg, lat="latitude", lon="longitude",
                    size="n_obs",
                    color=color_col if color_col else None,
                    color_continuous_scale="Turbo" if color_col else None,
                    title="Weather Station Locations",
                    hover_data=["n_obs"],
                )
            fig.update_layout(height=500, template=TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

            # Scatter plot alternative
            fig2 = px.scatter(
                station_avg, x="longitude", y="latitude",
                size="n_obs",
                color=color_col if color_col else "n_obs",
                color_continuous_scale="Turbo",
                template=TEMPLATE,
                title="Station Locations (Scatter)",
                labels={"latitude": "Latitude (°N)", "longitude": "Longitude (°E)"},
            )
            fig2.update_layout(height=450)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No geographic coordinates available in the dataset.")

        # Temporal analysis
        if "datetime" in df.columns:
            st.markdown("### Temporal Distribution")
            try:
                temp_df = df.copy()
                temp_df["datetime"] = pd.to_datetime(temp_df["datetime"])
                temp_df["month"] = temp_df["datetime"].dt.month
                temp_df["hour"] = temp_df["datetime"].dt.hour

                monthly = temp_df.groupby("month").size().reset_index(name="count")
                fig = px.bar(
                    monthly, x="month", y="count",
                    labels={"month": "Month", "count": "Observations"},
                    color="count", color_continuous_scale=["#1A1F2E", "#00D4FF"],
                    template=TEMPLATE, title="Observations by Month",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                hourly = temp_df.groupby("hour").size().reset_index(name="count")
                fig2 = px.line(
                    hourly, x="hour", y="count",
                    labels={"hour": "Hour of Day", "count": "Observations"},
                    template=TEMPLATE, title="Observations by Hour of Day",
                    markers=True,
                )
                fig2.update_traces(line_color="#00D4FF")
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("Could not parse datetime column for temporal analysis.")

