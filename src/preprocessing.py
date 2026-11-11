"""
Preprocessing Module
====================
Complete data preprocessing pipeline for satellite cloud/weather data.
Handles missing values, duplicates, encoding, scaling, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Optional
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Complete preprocessing pipeline for cloud/weather observation data."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = None
        self.selected_features = []
        self.metadata_columns = []
        self.preprocessing_log = []
        self._before_stats = None
        self._after_stats = None

    def _log(self, message: str):
        """Add entry to preprocessing log."""
        self.preprocessing_log.append(message)
        logger.info(message)

    def get_before_after_stats(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return before and after preprocessing statistics."""
        return self._before_stats, self._after_stats

    def detect_and_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove duplicate records."""
        n_duplicates = df.duplicated().sum()
        self._log(f"Duplicate records found: {n_duplicates}")

        if n_duplicates > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            self._log(f"Removed {n_duplicates} duplicates. Remaining records: {len(df)}")

        return df

    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing values and return summary DataFrame."""
        missing = pd.DataFrame({
            "Feature": df.columns,
            "Missing_Count": df.isnull().sum().values,
            "Missing_Percentage": (df.isnull().sum() / len(df) * 100).values,
            "Data_Type": df.dtypes.values.astype(str),
        })
        missing = missing.sort_values("Missing_Percentage", ascending=False)
        missing = missing[missing["Missing_Count"] > 0]
        return missing

    def handle_missing_values(
        self, df: pd.DataFrame,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        max_missing_pct: float = 50.0,
    ) -> pd.DataFrame:
        """
        Handle missing values with configurable strategies.

        Parameters
        ----------
        numerical_strategy : Strategy for numerical imputation ('mean', 'median', 'most_frequent')
        categorical_strategy : Strategy for categorical imputation ('most_frequent', 'constant')
        max_missing_pct : Drop columns with more than this percentage missing
        """
        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        cols_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
        if cols_to_drop:
            self._log(f"Dropping columns with >{max_missing_pct}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Impute numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols and df[num_cols].isnull().any().any():
            self.imputer = SimpleImputer(strategy=numerical_strategy)
            df[num_cols] = self.imputer.fit_transform(df[num_cols])
            self._log(f"Imputed {len(num_cols)} numerical columns using '{numerical_strategy}' strategy")

        # Impute categorical columns
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols and df[cat_cols].isnull().any().any():
            cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value="Unknown")
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            self._log(f"Imputed {len(cat_cols)} categorical columns using '{categorical_strategy}' strategy")

        return df

    def encode_categorical(
        self, df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding."""
        if columns is None:
            columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        # Filter to only columns that exist in df
        columns = [c for c in columns if c in df.columns]

        for col in columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            self._log(f"Encoded '{col}': {len(le.classes_)} unique categories")

        return df

    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Detect outliers using IQR method. Returns DataFrame with outlier statistics."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_stats = []
        for col in columns:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            outlier_stats.append({
                "Feature": col,
                "Q1": round(Q1, 4),
                "Q3": round(Q3, 4),
                "IQR": round(IQR, 4),
                "Lower_Bound": round(lower, 4),
                "Upper_Bound": round(upper, 4),
                "N_Outliers": n_outliers,
                "Outlier_Pct": round(n_outliers / len(df) * 100, 2),
            })

        return pd.DataFrame(outlier_stats)

    def select_features(
        self, df: pd.DataFrame,
        numerical_features: List[str],
        metadata_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features for ML analysis. Separates metadata from analytical features.

        Returns
        -------
        features_df : DataFrame with selected numerical features
        metadata_df : DataFrame with metadata (timestamps, coordinates, etc.)
        """
        # Filter to existing columns
        available_numerical = [f for f in numerical_features if f in df.columns]
        self.selected_features = available_numerical

        if metadata_features:
            available_metadata = [f for f in metadata_features if f in df.columns]
            self.metadata_columns = available_metadata
            metadata_df = df[available_metadata].copy() if available_metadata else pd.DataFrame()
        else:
            metadata_df = pd.DataFrame()

        features_df = df[available_numerical].copy()
        self._log(f"Selected {len(available_numerical)} numerical features for analysis")
        self._log(f"Preserved {len(self.metadata_columns)} metadata columns")

        return features_df, metadata_df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize numerical features using StandardScaler."""
        scaled_data = self.scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        self._log(f"Scaled {len(df.columns)} features using StandardScaler (mean=0, std=1)")
        return scaled_df

    def preprocess(
        self, df: pd.DataFrame,
        numerical_features: List[str],
        metadata_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        imputation_strategy: str = "median",
        max_missing_pct: float = 50.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.

        Returns
        -------
        scaled_df : Scaled numerical features ready for ML
        original_features_df : Unscaled numerical features (for interpretation)
        metadata_df : Metadata columns preserved separately
        """
        self.preprocessing_log = []
        self._log(f"Starting preprocessing pipeline on {len(df)} records, {len(df.columns)} columns")

        # Capture before stats
        self._before_stats = {
            "n_records": len(df),
            "n_features": len(df.columns),
            "total_missing": int(df.isnull().sum().sum()),
            "total_missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "numerical_stats": df.select_dtypes(include=[np.number]).describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        }

        # Step 1: Remove duplicates
        df = self.detect_and_remove_duplicates(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(
            df, numerical_strategy=imputation_strategy, max_missing_pct=max_missing_pct,
        )

        # Step 3: Encode categoricals if needed
        if categorical_features:
            available_cats = [c for c in categorical_features if c in df.columns]
            if available_cats:
                df = self.encode_categorical(df, columns=available_cats)

        # Step 4: Select features
        features_df, metadata_df = self.select_features(df, numerical_features, metadata_features)

        # Step 5: Drop rows with any remaining NaN in selected features
        n_before = len(features_df)
        valid_mask = features_df.notna().all(axis=1)
        features_df = features_df[valid_mask].reset_index(drop=True)
        if not metadata_df.empty:
            metadata_df = metadata_df[valid_mask].reset_index(drop=True)
        n_dropped = n_before - len(features_df)
        if n_dropped > 0:
            self._log(f"Dropped {n_dropped} rows with remaining NaN values in selected features")

        # Step 6: Remove constant columns
        constant_cols = [c for c in features_df.columns if features_df[c].nunique() <= 1]
        if constant_cols:
            features_df = features_df.drop(columns=constant_cols)
            self._log(f"Removed constant columns: {constant_cols}")

        # Step 7: Scale features
        original_features_df = features_df.copy()
        scaled_df = self.scale_features(features_df)

        # Capture after stats
        self._after_stats = {
            "n_records": len(scaled_df),
            "n_features": len(scaled_df.columns),
            "total_missing": 0,
            "total_missing_pct": 0.0,
            "duplicates": 0,
            "features_used": list(scaled_df.columns),
            "numerical_stats": original_features_df.describe().to_dict(),
        }

        self._log(f"Preprocessing complete: {len(scaled_df)} records, {len(scaled_df.columns)} features")

        return scaled_df, original_features_df, metadata_df

