"""
Satellite Cloud Pattern Segmentation Dashboard
===============================================
Main application entry point for the Streamlit dashboard.

An interactive machine learning application that explores satellite cloud patterns
using PCA, t-SNE, and K-Means clustering on satellite-derived weather observations.
"""

import streamlit as st
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import VIZ_CONFIG, DATASET_CONFIG

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Satellite Cloud Pattern Segmentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS - Professional Dark Theme
# ============================================================
st.markdown("""
<style>
    /* =========================================================
       70-20-10 SCIENTIFIC DARK PALETTE WITH ULTRA-SMOOTH ANIMATIONS
       ========================================================= */

    /* Smooth page-load & view transitions */
    @keyframes smoothPageEnter {
        0% {
            opacity: 0;
            transform: translateY(14px);
            filter: blur(3px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }
    }

    @keyframes celestialGlow {
        0%, 100% {
            opacity: 0.7;
            filter: drop-shadow(0 0 4px rgba(0, 240, 255, 0.35));
        }
        50% {
            opacity: 1;
            filter: drop-shadow(0 0 12px rgba(0, 240, 255, 0.75));
        }
    }

    .main .block-container {
        animation: smoothPageEnter 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* 70% Canvas: Main background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #10192D 0%, #090D16 60%);
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* 20% Structural: Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0A0E1A !important;
        border-right: 1px solid rgba(51, 65, 85, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Sidebar Navigation item smooth transition */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        padding: 9px 14px !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: rgba(19, 27, 46, 0.85) !important;
        border-color: rgba(0, 240, 255, 0.28) !important;
        transform: translateX(4px);
    }

    /* 20% Structural: Cards & Metric Containers with silky hover elevation */
    .stMetric {
        background: linear-gradient(145deg, #131B2E 0%, #0E1626 100%) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 14px !important;
        padding: 18px !important;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
        transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.32s cubic-bezier(0.16, 1, 0.3, 1) !important;
        will-change: transform, box-shadow;
    }
    .stMetric:hover {
        border-color: rgba(0, 240, 255, 0.45) !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 14px 32px -4px rgba(0, 0, 0, 0.6), 0 0 18px -2px rgba(0, 240, 255, 0.18) !important;
    }

    /* 20% Structural: Custom Card */
    .custom-card {
        background: linear-gradient(145deg, #131B2E 0%, #0E1626 100%);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 28px -6px rgba(0, 0, 0, 0.5);
        transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.32s cubic-bezier(0.16, 1, 0.3, 1) !important;
        will-change: transform, box-shadow;
    }
    .custom-card:hover {
        border-color: rgba(0, 240, 255, 0.35) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.6), 0 0 18px -4px rgba(0, 240, 255, 0.1) !important;
    }

    /* 10% Accent Card (Key Highlights) */
    .accent-card {
        background: linear-gradient(135deg, #131B2E 0%, #092336 100%);
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px -5px rgba(0, 240, 255, 0.12);
        transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.32s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.32s cubic-bezier(0.16, 1, 0.3, 1) !important;
        will-change: transform, box-shadow;
    }
    .accent-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(0, 240, 255, 0.6) !important;
        box-shadow: 0 16px 40px -4px rgba(0, 240, 255, 0.24) !important;
    }

    /* Typography Hierarchy */
    h1 {
        color: #F8FAFC !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    h2 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
    h4 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    p, span, label {
        color: #94A3B8;
    }

    /* 10% Metric Accent */
    [data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.25);
        transition: color 0.3s ease;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* 20% Tabs Navigation with smooth underline glow */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #101726 !important;
        border-radius: 10px 10px 0 0 !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(51, 65, 85, 0.4) !important;
        border-bottom: none !important;
        padding: 10px 18px !important;
        font-weight: 500 !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background-color: #141E33 !important;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #16243D 0%, #0F1A2E 100%) !important;
        color: #00F0FF !important;
        border: 1px solid rgba(0, 240, 255, 0.4) !important;
        border-bottom: 2px solid #00F0FF !important;
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.15);
        font-weight: 600 !important;
    }

    /* 10% Primary Button Accent with spring micro-interaction */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00F0FF 0%, #00B4D8 100%) !important;
        color: #090D16 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(0, 240, 255, 0.35) !important;
        transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(0, 240, 255, 0.55) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) scale(0.98) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #131B2E !important;
        color: #CBD5E1 !important;
        border: 1px solid rgba(51, 65, 85, 0.6) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #00F0FF !important;
        color: #00F0FF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(0, 240, 255, 0.2) !important;
    }

    /* Sliders and Selects with smooth focus transitions */
    .stSlider [data-baseweb="slider"] {
        color: #00F0FF !important;
        transition: all 0.2s ease;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
        transition: border-color 0.25s ease;
    }

    /* Dataframe rounding */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(51, 65, 85, 0.4);
        transition: border-color 0.3s ease;
    }
    .stDataFrame:hover {
        border-color: rgba(51, 65, 85, 0.7);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #131B2E !important;
        border: 1px solid rgba(51, 65, 85, 0.4) !important;
        border-radius: 10px !important;
        color: #CBD5E1 !important;
        transition: all 0.25s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(0, 240, 255, 0.3) !important;
        background-color: #162035 !important;
    }

    /* Glowing Divider with subtle animation */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00F0FF, transparent);
        margin: 22px 0;
        border: none;
        animation: celestialGlow 4s ease-in-out infinite;
    }

    /* Glowing Text */
    .glow-text {
        color: #00F0FF;
        text-shadow: 0 0 14px rgba(0, 240, 255, 0.45);
    }

    /* 10% Accent Badges with gentle hover */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin: 3px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge:hover {
        transform: translateY(-1px);
    }
    .badge-blue { background: rgba(0, 240, 255, 0.12); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.35); }
    .badge-blue:hover { box-shadow: 0 0 10px rgba(0, 240, 255, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.12); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-green:hover { box-shadow: 0 0 10px rgba(16, 185, 129, 0.3); }
    .badge-orange { background: rgba(251, 146, 60, 0.12); color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.35); }
    .badge-orange:hover { box-shadow: 0 0 10px rgba(251, 146, 60, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.12); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-purple:hover { box-shadow: 0 0 10px rgba(168, 85, 247, 0.3); }

    /* Hide Streamlit default branding and native multi-page navigation */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    div[data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING WITH CACHING
# ============================================================
@st.cache_data(ttl=3600)
def load_raw_data():
    """Load raw dataset with caching."""
    from src.data_loader import load_dataset
    filepath = os.path.join(PROJECT_ROOT, "data", "raw", DATASET_CONFIG["raw_filename"])
    if not os.path.exists(filepath):
        st.error(f"Dataset not found at: {filepath}")
        st.info("Run `python prepare_dataset.py` to generate the dataset.")
        st.stop()
    return load_dataset(filepath)


@st.cache_data(ttl=3600)
def preprocess_data(_df):
    """Run preprocessing pipeline with caching."""
    from src.preprocessing import DataPreprocessor
    from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, METADATA_FEATURES

    preprocessor = DataPreprocessor(random_state=42)

    # Use all available numerical features
    available_numerical = [f for f in NUMERICAL_FEATURES if f in _df.columns]
    # Add any additional numerical columns
    extra_num = [c for c in _df.select_dtypes(include=['number']).columns
                 if c not in available_numerical and c not in METADATA_FEATURES]
    available_numerical.extend(extra_num)

    scaled_df, original_df, metadata_df = preprocessor.preprocess(
        _df.copy(),
        numerical_features=available_numerical,
        metadata_features=[f for f in METADATA_FEATURES if f in _df.columns],
        categorical_features=[f for f in CATEGORICAL_FEATURES if f in _df.columns],
        imputation_strategy="median",
    )

    return scaled_df, original_df, metadata_df, preprocessor


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2.5rem; margin: 0;'>🛰️</h1>
        <h3 style='color: #00D4FF; margin: 5px 0; font-size: 1.1rem;'>Cloud Pattern</h3>
        <h3 style='color: #4ECDC4; margin: 0; font-size: 1.1rem;'>Segmentation</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "📍 Navigation",
        [
            "🏠 Overview",
            "🔍 Data Exploration",
            "⚙️ Preprocessing",
            "📊 PCA Analysis",
            "🌀 t-SNE Analysis",
            "🎯 K-Means Clustering",
            "📈 Comparative Analysis",
            "📋 Conclusions & Report",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <p style='color: #6B7280; font-size: 0.75rem; margin: 0;'>
            Unsupervised Machine Learning<br>
            Satellite Data Exploration<br>
            PCA • t-SNE • K-Means
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE ROUTING
# ============================================================
if page == "🏠 Overview":
    from views.overview import render_overview
    render_overview()

elif page == "🔍 Data Exploration":
    from views.data_exploration import render_data_exploration
    df = load_raw_data()
    render_data_exploration(df)

elif page == "⚙️ Preprocessing":
    from views.preprocessing_page import render_preprocessing
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_preprocessing(df, scaled_df, original_df, metadata_df, preprocessor)

elif page == "📊 PCA Analysis":
    from views.pca_page import render_pca
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_pca(scaled_df, original_df)

elif page == "🌀 t-SNE Analysis":
    from views.tsne_page import render_tsne
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_tsne(scaled_df, original_df)

elif page == "🎯 K-Means Clustering":
    from views.clustering_page import render_clustering
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_clustering(scaled_df, original_df, metadata_df)

elif page == "📈 Comparative Analysis":
    from views.comparison import render_comparison
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_comparison(scaled_df, original_df)

elif page == "📋 Conclusions & Report":
    from views.conclusions import render_conclusions
    df = load_raw_data()
    scaled_df, original_df, metadata_df, preprocessor = preprocess_data(df)
    render_conclusions(scaled_df, original_df, df)

