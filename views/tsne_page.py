"""
Page 5: t-SNE Analysis
======================
Interactive t-SNE with parameter controls, visualizations, and educational content.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tsne_analysis import TSNEAnalyzer
from config import VIZ_CONFIG, TSNE_CONFIG

COLORS = VIZ_CONFIG["color_palette"]
TEMPLATE = VIZ_CONFIG["plotly_template"]


def render_tsne(scaled_df, original_df):
    """Render the t-SNE Analysis page."""

    st.markdown("""
    <h1>🌀 t-SNE Analysis</h1>
    <p style='color: #8892A0;'>Non-linear dimensionality reduction for high-dimensional data visualization.</p>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    analyzer = TSNEAnalyzer(random_state=TSNE_CONFIG["random_state"])

    # ── Controls ───────────────────────────────────────────
    st.markdown("### ⚙️ Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        perplexity = st.slider("Perplexity", 5, 100, TSNE_CONFIG["default_perplexity"])
    with col2:
        learning_rate = st.slider("Learning Rate", 10, 1000, TSNE_CONFIG["default_learning_rate"], step=10)
    with col3:
        n_iter = st.slider("Iterations", 250, 5000, TSNE_CONFIG["default_n_iter"], step=250)
    with col4:
        max_samples = len(scaled_df)
        default_sample = min(max_samples, TSNE_CONFIG.get("default_samples_for_tsne", 2500))
        sample_size = st.slider("Sample Size", 500, max_samples, default_sample, step=500)

    sampling_note = ""
    if sample_size < len(scaled_df):
        sampling_note = f"⚠️ Sampling {sample_size:,} from {len(scaled_df):,} points for computational efficiency."

    tabs = st.tabs(["🎯 Run t-SNE", "🔬 Parameter Comparison", "📖 Understanding t-SNE"])

    # TAB 1: Run t-SNE
    with tabs[0]:
        if st.button("▶️ Run t-SNE", type="primary"):
            if sampling_note:
                st.info(sampling_note)

            with st.spinner(f"Running t-SNE (perplexity={perplexity}, lr={learning_rate}, iter={n_iter})..."):
                result = analyzer.run_tsne(
                    scaled_df.values,
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    sample_size=sample_size if sample_size < len(scaled_df) else None,
                )

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("KL Divergence", f"{result['kl_divergence']:.4f}")
            c2.metric("Computation Time", f"{result['elapsed_seconds']:.1f}s")
            c3.metric("Points Plotted", f"{result['n_samples']:,}")
            c4.metric("Sampled", "Yes" if result["sampled"] else "No")

            # 2D Scatter
            embedding = result["embedding"]
            fig = px.scatter(
                x=embedding[:, 0], y=embedding[:, 1],
                labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
                template=TEMPLATE,
                title=f"t-SNE 2D Projection (perplexity={result['parameters']['perplexity']:.0f}, lr={learning_rate})",
                opacity=0.5,
                color_discrete_sequence=[COLORS[3]],
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Store result in session state for comparison page
            st.session_state["tsne_result"] = result
            st.session_state["tsne_params"] = {
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
            }

        elif "tsne_result" in st.session_state:
            result = st.session_state["tsne_result"]
            embedding = result["embedding"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("KL Divergence", f"{result['kl_divergence']:.4f}")
            c2.metric("Computation Time", f"{result['elapsed_seconds']:.1f}s")
            c3.metric("Points Plotted", f"{result['n_samples']:,}")
            c4.metric("Sampled", "Yes" if result["sampled"] else "No")

            fig = px.scatter(
                x=embedding[:, 0], y=embedding[:, 1],
                labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
                template=TEMPLATE,
                title=f"t-SNE 2D Projection (cached result)",
                opacity=0.5,
                color_discrete_sequence=[COLORS[3]],
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Click **Run t-SNE** to generate the visualization.")

    # TAB 2: Parameter Comparison
    with tabs[1]:
        st.markdown("### 🔬 Perplexity Comparison")
        st.markdown(
            "Compare how different perplexity values affect the t-SNE visualization. "
            "Lower perplexity emphasizes local structure; higher values capture more global patterns."
        )

        perplexities = st.multiselect(
            "Select perplexity values to compare:",
            [5, 10, 15, 20, 30, 50, 75, 100],
            default=[5, 15, 30, 50],
        )

        comp_sample = st.slider("Comparison sample size:", 500, min(5000, len(scaled_df)), min(2000, len(scaled_df)), step=500, key="comp_sample")

        if st.button("▶️ Run Comparison", type="secondary"):
            if len(perplexities) < 2:
                st.warning("Select at least 2 perplexity values.")
            else:
                results = {}
                progress = st.progress(0)
                for i, p in enumerate(perplexities):
                    with st.spinner(f"Running t-SNE with perplexity={p}..."):
                        res = analyzer.run_tsne(
                            scaled_df.values, perplexity=p,
                            learning_rate=learning_rate, n_iter=n_iter,
                            sample_size=comp_sample if comp_sample < len(scaled_df) else None,
                        )
                        results[p] = res
                    progress.progress((i + 1) / len(perplexities))

                # Grid of plots
                n_plots = len(perplexities)
                cols_per_row = 2
                for i in range(0, n_plots, cols_per_row):
                    plot_cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < n_plots:
                            p = perplexities[idx]
                            emb = results[p]["embedding"]
                            with plot_cols[j]:
                                fig = px.scatter(
                                    x=emb[:, 0], y=emb[:, 1],
                                    template=TEMPLATE,
                                    title=f"Perplexity = {p} (KL: {results[p]['kl_divergence']:.3f})",
                                    opacity=0.5,
                                    color_discrete_sequence=[COLORS[idx % len(COLORS)]],
                                )
                                fig.update_traces(marker=dict(size=2))
                                fig.update_layout(height=350, showlegend=False,
                                                  xaxis_title="", yaxis_title="")
                                st.plotly_chart(fig, use_container_width=True)

                # KL divergence comparison
                kl_data = pd.DataFrame({
                    "Perplexity": perplexities,
                    "KL_Divergence": [results[p]["kl_divergence"] for p in perplexities],
                    "Time_s": [results[p]["elapsed_seconds"] for p in perplexities],
                })
                fig = px.bar(
                    kl_data, x="Perplexity", y="KL_Divergence",
                    color="Time_s", color_continuous_scale=["#4ECDC4", "#FF6B6B"],
                    template=TEMPLATE, title="KL Divergence by Perplexity",
                    text=[f"{v:.3f}" for v in kl_data["KL_Divergence"]],
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Understanding t-SNE
    with tabs[2]:
        st.markdown("### 📖 Understanding t-SNE")
        interp = analyzer.get_tsne_interpretation()

        col1, col2 = st.columns(2)
        with col1:
            for param in ["perplexity", "learning_rate", "n_iter"]:
                info = interp[param]
                st.markdown(f"""
                <div class='custom-card'>
                    <h4 style='color: #00D4FF;'>{param.replace('_', ' ').title()}</h4>
                    <p style='color: #B0B8C8;'>{info['description']}</p>
                    <p style='color: #4ECDC4; font-size: 0.85rem;'>
                        💡 {info['recommendation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='accent-card'>
                <h4 style='color: #FF6B6B;'>⚠️ Critical Limitations</h4>
            </div>
            """, unsafe_allow_html=True)
            for lim in interp["limitations"]:
                st.markdown(f"- ⛔ {lim}")

            st.warning(
                "**IMPORTANT:** t-SNE is a visualization tool. Do not interpret distances between "
                "clusters or cluster sizes as representing actual data relationships. Only local "
                "neighborhood structure is preserved."
            )

