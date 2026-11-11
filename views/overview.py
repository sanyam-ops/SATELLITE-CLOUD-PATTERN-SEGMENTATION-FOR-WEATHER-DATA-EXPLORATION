"""
Page 1: Overview
================
Project introduction, objectives, dataset summary, and ML pipeline overview.
"""

import streamlit as st


def render_overview():
    """Render the Overview page."""

    # Header
    st.markdown("""
    <div style='text-align: center; padding: 30px 0 10px 0;'>
        <h1 style='font-size: 2.6rem; margin-bottom: 6px; letter-spacing: -0.02em;'>
            🛰️ Satellite Cloud Pattern Segmentation
        </h1>
        <p style='color: #00F0FF; font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; text-shadow: 0 0 12px rgba(0, 240, 255, 0.3);'>
            Weather Data Exploration Using PCA, t-SNE & K-Means Clustering
        </p>
        <p style='color: #94A3B8; font-size: 0.95rem;'>
            An Unsupervised Machine Learning Framework for Atmospheric Regime Discovery
        </p>
    </div>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # Research Objective
    st.markdown("### 🎯 Research Objective")
    st.markdown("""
    <div class='accent-card'>
        <p style='color: #F1F5F9; line-height: 1.8; font-size: 1.05rem;'>
            This project implements an <b style='color: #00F0FF;'>unsupervised machine learning</b> architecture
            to explore naturally occurring structures in satellite-derived atmospheric observation data.
            Combining <b style='color: #38BDF8;'>Principal Component Analysis (PCA)</b> for optimal linear variance retention,
            <b style='color: #FBBF24;'>t-Distributed Stochastic Neighbor Embedding (t-SNE)</b> for non-linear manifold inspection, and
            <b style='color: #10B981;'>K-Means Clustering</b> on PCA-reduced subspaces, we identify persistent cloud-pattern
            regimes without imposing preconceived meteorological labels.
        </p>
        <p style='color: #94A3B8; font-size: 0.88rem; margin-top: 12px; border-top: 1px solid rgba(51, 65, 85, 0.4); padding-top: 10px;'>
            ⚠️ <b>Scientific Clarification:</b> All clusters represent empirical mathematical groupings in feature space rather than
            authoritative meteorological classifications. Independent atmospheric sensor verification is required for operational conclusions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Statistics
    st.markdown("### 📊 Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", "15,075", help="Total weather observations")
    c2.metric("Features", "22", help="Total columns in dataset")
    c3.metric("Numerical", "14", help="Features used for ML analysis")
    c4.metric("Stations", "15", help="Indian weather stations")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Time Span", "2 Years", help="Jan 2023 – Dec 2024")
    c6.metric("Region", "India", help="Indian subcontinent coverage")
    c7.metric("Missing", "~3%", help="Missing value percentage")
    c8.metric("File Size", "2.2 MB", help="CSV dataset size")

    st.markdown("")

    # ML Pipeline
    st.markdown("### 🔄 Machine Learning Pipeline")
    st.markdown("""
    <div class='custom-card'>
        <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 12px;'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>📥</div>
                <div style='color: #00F0FF; font-weight: 700; font-size: 0.85rem;'>1. Ingestion</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>CSV ingestion &<br>validation</div>
            </div>
            <div style='color: #334155; font-size: 1.5rem;'>→</div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>⚙️</div>
                <div style='color: #38BDF8; font-weight: 700; font-size: 0.85rem;'>2. Preprocessing</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>Impute, encode,<br>StandardScaler</div>
            </div>
            <div style='color: #334155; font-size: 1.5rem;'>→</div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>📊</div>
                <div style='color: #FBBF24; font-weight: 700; font-size: 0.85rem;'>3. PCA</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>Variance analysis<br>& loadings</div>
            </div>
            <div style='color: #334155; font-size: 1.5rem;'>→</div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🌀</div>
                <div style='color: #A855F7; font-weight: 700; font-size: 0.85rem;'>4. t-SNE</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>Non-linear<br>manifold check</div>
            </div>
            <div style='color: #334155; font-size: 1.5rem;'>→</div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🎯</div>
                <div style='color: #10B981; font-weight: 700; font-size: 0.85rem;'>5. K-Means</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>Subspace regime<br>segmentation</div>
            </div>
            <div style='color: #334155; font-size: 1.5rem;'>→</div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>📈</div>
                <div style='color: #F43F5E; font-weight: 700; font-size: 0.85rem;'>6. Evaluation</div>
                <div style='color: #94A3B8; font-size: 0.75rem;'>Silhouette, elbow<br>& ethics audit</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Information
    st.markdown("### 📁 About the Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h4 style='color: #00F0FF;'>Data Source Architecture</h4>
            <p style='color: #CBD5E1;'>
                <b>Dataset:</b> Cloud & Meteorological Observations<br>
                <b>Statistical Foundation:</b> NOAA Local Climatological Data (LCDv2)<br>
                <b>Structure:</b> Multimodal tabular observations with continuous physics constraints<br>
                <b>Licensing:</b> Open Data / Public Domain Reference
            </p>
            <p style='color: #94A3B8; font-size: 0.85rem;'>
                Synthesized following documented atmospheric distributions (lapse rates, Magnus formula,
                Weibull wind distributions). The engine is source-configurable to allow instant INSAT substitution.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h4 style='color: #38BDF8;'>ISRO MOSDAC / INSAT Compatibility</h4>
            <p style='color: #CBD5E1;'>
                Designed for straightforward plug-and-play substitution with real INSAT-3D/3DR payloads:
            </p>
            <ul style='color: #94A3B8; font-size: 0.85rem;'>
                <li>Outgoing Longwave Radiation (OLR) products</li>
                <li>Quantitative Precipitation Estimates (QPE)</li>
                <li>Cloud Motion Vectors (CMV) & Brightness Temperatures</li>
            </ul>
            <p style='color: #94A3B8; font-size: 0.85rem;'>
                Simply export or drop your converted CSV into <code>data/raw/</code> to recompute all analytics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Technologies
    st.markdown("### 🛠️ Technologies Used")
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 0;'>
        <span class='badge badge-blue'>Python 3.12</span>
        <span class='badge badge-blue'>Streamlit</span>
        <span class='badge badge-green'>Scikit-learn</span>
        <span class='badge badge-green'>Pandas</span>
        <span class='badge badge-green'>NumPy</span>
        <span class='badge badge-purple'>Plotly</span>
        <span class='badge badge-purple'>Matplotlib</span>
        <span class='badge badge-purple'>Seaborn</span>
        <span class='badge badge-orange'>SciPy</span>
    </div>
    """, unsafe_allow_html=True)

    # Project Architecture & Metadata
    st.markdown("### 📌 Project Scope & Specifications")
    st.markdown("""
    <div class='custom-card'>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;'>
            <div style='text-align: center;'>
                <div style='color: #6B7280; font-size: 0.8rem;'>DOMAIN</div>
                <div style='color: #E0E0E0; font-weight: 600;'>Satellite Meteorology</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #6B7280; font-size: 0.8rem;'>PARADIGM</div>
                <div style='color: #E0E0E0; font-weight: 600;'>Unsupervised ML</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #6B7280; font-size: 0.8rem;'>REDUCTION</div>
                <div style='color: #E0E0E0; font-weight: 600;'>PCA & t-SNE</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #6B7280; font-size: 0.8rem;'>SEGMENTATION</div>
                <div style='color: #E0E0E0; font-weight: 600;'>K-Means Clustering</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #6B7280; font-size: 0.8rem;'>EVALUATION</div>
                <div style='color: #E0E0E0; font-weight: 600;'>Silhouette & Elbow</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

