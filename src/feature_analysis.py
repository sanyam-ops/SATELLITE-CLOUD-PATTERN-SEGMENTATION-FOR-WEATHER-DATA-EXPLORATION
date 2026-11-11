"""
Feature Analysis Module
=======================
Provides statistical analysis, correlation analysis, and feature distribution analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numerical features."""
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr()


def compute_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive statistics for all numerical features."""
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats["skewness"] = num_df.skew()
    stats["kurtosis"] = num_df.kurtosis()
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = (stats["std"] / stats["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    stats["iqr"] = num_df.quantile(0.75) - num_df.quantile(0.25)
    return stats.round(4)


def identify_highly_correlated(
    corr_matrix: pd.DataFrame, threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """Identify pairs of highly correlated features."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                pairs.append((cols[i], cols[j], round(corr_matrix.iloc[i, j], 4)))
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def get_feature_descriptions() -> Dict[str, str]:
    """Return descriptions for standard cloud/weather features."""
    return {
        "temperature": "Air temperature at station level (°C)",
        "dew_point": "Dew point temperature (°C) — indicator of atmospheric moisture content",
        "relative_humidity": "Relative humidity (%) — ratio of actual to saturation vapor pressure",
        "wind_speed": "Sustained wind speed (m/s or km/h)",
        "wind_direction": "Wind direction in degrees (0-360°, meteorological convention)",
        "station_pressure": "Atmospheric pressure at station level (hPa/mb)",
        "sea_level_pressure": "Pressure adjusted to mean sea level (hPa/mb)",
        "visibility": "Horizontal visibility distance (km)",
        "cloud_ceiling": "Height of cloud base above ground (meters or feet)",
        "precipitation": "Precipitation amount in observation period (mm)",
        "wind_gust": "Maximum wind gust speed (m/s or km/h)",
        "altimeter_setting": "Altimeter pressure setting (inHg or hPa)",
        "cloud_cover_fraction": "Fraction of sky covered by clouds (0-1 or oktas)",
        "wet_bulb_temperature": "Wet bulb temperature (°C)",
        "heat_index": "Apparent temperature combining heat and humidity (°C)",
        "wind_chill": "Wind chill factor (°C)",
        "sky_condition": "Categorical sky coverage (CLR, FEW, SCT, BKN, OVC)",
        "weather_type": "Present weather phenomena (rain, fog, snow, etc.)",
    }

