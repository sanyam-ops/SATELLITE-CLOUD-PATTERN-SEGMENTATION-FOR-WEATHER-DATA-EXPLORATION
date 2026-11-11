"""
Configuration file for Satellite Cloud Pattern Segmentation project.
Contains all configurable parameters, paths, and constants.
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Primary dataset: GLOBE Observer Clouds Matched Satellite Data (NASA)
# Alternative: NOAA Local Climatological Data (LCD)
# The data source is configurable so INSAT data can be substituted later.

DATASET_CONFIG = {
    "name": "NOAA Local Climatological Data (LCD) – Hourly Cloud & Weather Observations",
    "source": "NOAA National Centers for Environmental Information (NCEI)",
    "url": "https://www.ncei.noaa.gov/data/local-climatological-data/",
    "license": "Public Domain (U.S. Government Work)",
    "description": (
        "Hourly surface weather observations from NOAA automated weather stations "
        "including satellite-validated cloud cover, ceiling height, temperature, "
        "humidity, pressure, wind speed/direction, visibility, and precipitation. "
        "These stations use satellite data products (GOES geostationary satellites) "
        "for cloud analysis validation."
    ),
    "alternative_for_insat": (
        "INSAT-3D/3DR data from ISRO MOSDAC (mosdac.gov.in) is available in HDF5/NetCDF "
        "format but requires registration and conversion. This NOAA LCD dataset serves as "
        "a scientifically appropriate alternative containing genuine satellite-validated "
        "cloud and meteorological observations. The data pipeline is designed to accept "
        "any CSV with numerical weather features, making INSAT data substitution straightforward."
    ),
    "raw_filename": "cloud_weather_observations.csv",
    "processed_filename": "processed_cloud_data.csv",
}

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
# Features used for ML analysis (actual column names will be mapped during loading)
NUMERICAL_FEATURES = [
    "temperature",
    "dew_point",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "station_pressure",
    "sea_level_pressure",
    "visibility",
    "cloud_ceiling",
    "precipitation",
    "wind_gust",
    "altimeter_setting",
]

CATEGORICAL_FEATURES = [
    "sky_condition",
    "weather_type",
    "wind_direction_cardinal",
]

METADATA_FEATURES = [
    "station_id",
    "datetime",
    "latitude",
    "longitude",
    "elevation",
]

# ============================================================
# PCA CONFIGURATION
# ============================================================
PCA_CONFIG = {
    "default_n_components": 5,
    "max_components": 10,
    "variance_threshold": 0.95,  # Retain 95% variance
    "random_state": 42,
}

# ============================================================
# t-SNE CONFIGURATION
# ============================================================
TSNE_CONFIG = {
    "default_perplexity": 30,
    "perplexity_range": (5, 100),
    "default_learning_rate": 200,
    "learning_rate_range": (10, 1000),
    "default_n_iter": 1000,
    "n_iter_range": (250, 5000),
    "random_state": 42,
    "default_samples_for_tsne": 2500,
    "max_samples_for_tsne": 10000,  # t-SNE sampling limit for performance
    "n_components": 2,
}

# ============================================================
# K-MEANS CONFIGURATION
# ============================================================
KMEANS_CONFIG = {
    "default_k": 5,
    "k_range": (2, 15),
    "max_iter": 300,
    "n_init": 10,
    "random_state": 42,
}

# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================
VIZ_CONFIG = {
    "color_palette": [
        "#00F0FF", "#F43F5E", "#10B981", "#FBBF24", "#A855F7",
        "#38BDF8", "#FB923C", "#EC4899", "#6366F1", "#14B8A6",
        "#E11D48", "#8B5CF6", "#06B6D4", "#F59E0B", "#84CC16",
    ],
    "background_color": "#090D16",
    "card_background": "#131B2E",
    "text_color": "#F1F5F9",
    "accent_color": "#00F0FF",
    "secondary_accent": "#10B981",
    "plotly_template": "plotly_dark",
    "figure_height": 500,
    "figure_width": 800,
}

# ============================================================
# RANDOM SEED (Global)
# ============================================================
RANDOM_STATE = 42

