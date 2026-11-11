"""
Data Loader Module
==================
Handles loading, initial inspection, and validation of satellite cloud/weather datasets.
Supports NOAA LCD data and is configurable for INSAT data substitution.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file with basic validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict:
    """Perform initial dataset inspection and return summary statistics."""
    inspection = {
        "shape": df.shape,
        "n_records": len(df),
        "n_features": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "total_missing_pct": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
    }

    # Numerical feature statistics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        inspection["numerical_stats"] = df[num_cols].describe().to_dict()
        inspection["numerical_columns"] = num_cols

    # Categorical feature statistics
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        inspection["categorical_columns"] = cat_cols
        inspection["categorical_unique"] = {col: int(df[col].nunique()) for col in cat_cols}

    return inspection


def validate_dataset(df: pd.DataFrame, min_records: int = 100, min_features: int = 5) -> Tuple[bool, list]:
    """Validate dataset meets minimum requirements for ML analysis."""
    issues = []

    if len(df) < min_records:
        issues.append(f"Insufficient records: {len(df)} (minimum: {min_records})")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < min_features:
        issues.append(f"Insufficient numerical features: {len(num_cols)} (minimum: {min_features})")

    # Check for columns that are entirely missing
    fully_missing = df.columns[df.isnull().all()].tolist()
    if fully_missing:
        issues.append(f"Fully missing columns: {fully_missing}")

    # Check for constant columns
    constant_cols = [col for col in num_cols if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant columns (no variation): {constant_cols}")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_dataset_info(df: pd.DataFrame, config: dict) -> Dict:
    """Generate comprehensive dataset information for display."""
    inspection = inspect_dataset(df)

    info = {
        "dataset_name": config.get("name", "Unknown"),
        "source": config.get("source", "Unknown"),
        "url": config.get("url", ""),
        "license": config.get("license", "Unknown"),
        "description": config.get("description", ""),
        "n_records": inspection["n_records"],
        "n_features": inspection["n_features"],
        "columns": inspection["columns"],
        "dtypes": {k: str(v) for k, v in inspection["dtypes"].items()},
        "missing_values": inspection["missing_values"],
        "missing_percentages": inspection["missing_percentages"],
        "total_missing_pct": inspection["total_missing_pct"],
        "duplicates": inspection["duplicates"],
        "memory_mb": inspection["memory_usage_mb"],
    }

    if "numerical_columns" in inspection:
        info["numerical_columns"] = inspection["numerical_columns"]
        info["numerical_stats"] = inspection["numerical_stats"]
    if "categorical_columns" in inspection:
        info["categorical_columns"] = inspection["categorical_columns"]
        info["categorical_unique"] = inspection["categorical_unique"]

    return info

