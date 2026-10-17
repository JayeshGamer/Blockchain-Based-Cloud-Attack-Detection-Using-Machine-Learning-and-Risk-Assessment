"""
Zero-Leakage Preprocessing Pipeline for CICIDS2017 Network Telemetry.
Handles imputation, outlier-resilient scaling, constant column pruning,
and label mapping. Imputers and scalers are strictly fitted on training splits.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import RobustScaler, LabelEncoder

from src.data.audit import normalize_label, ATTACK_FAMILY_MAPPING


# Identified constant columns with zero variance across all 8 files
DEFAULT_CONSTANT_COLUMNS = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
]


class NetworkDataPreprocessor:
    """
    Stateful preprocessor for CICIDS2017 network flow data.
    Ensures zero data leakage by computing medians and scaling parameters
    exclusively on the training split.
    """

    def __init__(
        self,
        drop_constant: bool = True,
        constant_columns: Optional[List[str]] = None,
        scaler_type: str = "robust",
    ):
        self.drop_constant = drop_constant
        self.constant_columns = constant_columns or DEFAULT_CONSTANT_COLUMNS
        self.scaler_type = scaler_type

        self.feature_names_: List[str] = []
        self.impute_values_: Dict[str, float] = {}
        self.scaler_: Optional[RobustScaler] = None
        self.multiclass_encoder_: LabelEncoder = LabelEncoder()
        self.family_encoder_: LabelEncoder = LabelEncoder()
        self.is_fitted_: bool = False

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from column names."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        return df

    def extract_and_clean_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Extract and sanitize labels, stripping all non-feature metadata columns.
        Returns:
            features_df: Clean DataFrame containing strictly numeric flow features
            y_binary: pd.Series (0 = BENIGN, 1 = ATTACK)
            y_multiclass: pd.Series (Normalized specific attack names)
            y_family: pd.Series (High-level attack family)
        """
        df = self.clean_column_names(df)
        label_candidates = [c for c in df.columns if "label" in c.lower()]
        if not label_candidates:
            raise KeyError("No label column found in DataFrame.")
        label_col = label_candidates[0]
        raw_labels = df[label_col].astype(str)
        cleaned_labels = raw_labels.apply(normalize_label)

        y_binary = (cleaned_labels != "BENIGN").astype(int)
        y_multiclass = cleaned_labels
        y_family = cleaned_labels.map(lambda lbl: ATTACK_FAMILY_MAPPING.get(lbl, "Other Attack"))

        # Strip all label variants and metadata columns so they never enter feature space
        metadata_to_drop = [
            label_col,
            "source_file",
            "source_day",
            "original_row_index",
            "row_hash",
            "AttackFamily",
            "NormalizedLabel",
            "Label",
        ]
        cols_to_drop = [c for c in metadata_to_drop if c in df.columns]
        features_df = df.drop(columns=cols_to_drop)
        return features_df, y_binary, y_multiclass, y_family

    def fit(self, X: pd.DataFrame, y_multiclass: Optional[pd.Series] = None) -> "NetworkDataPreprocessor":
        """
        Fit preprocessing parameters on the training DataFrame:
        - Prune constant features and duplicate columns
        - Compute median statistics for numerical imputation
        - Fit RobustScaler
        - Fit multiclass label encoders
        """
        X = self.clean_column_names(X)

        # Remove duplicate column 'Fwd Header Length.1' if present
        cols_to_drop = list(self.constant_columns) if self.drop_constant else []
        if "Fwd Header Length.1" in X.columns and "Fwd Header Length" in X.columns:
            cols_to_drop.append("Fwd Header Length.1")

        remaining_cols = [c for c in X.columns if c not in cols_to_drop and c.lower() != "label"]
        self.feature_names_ = remaining_cols

        X_num = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        # Replace infinities with NaN
        X_num = X_num.replace([np.inf, -np.inf], np.nan)

        # Compute median imputation values strictly on training data
        self.impute_values_ = {}
        for col in self.feature_names_:
            med = X_num[col].median()
            # If all are NaN, fallback to 0.0
            self.impute_values_[col] = 0.0 if np.isnan(med) else float(med)

        X_imputed = X_num.fillna(self.impute_values_)

        # Fit scaler
        if self.scaler_type == "robust":
            self.scaler_ = RobustScaler(quantile_range=(5.0, 95.0))
        else:
            self.scaler_ = RobustScaler()

        self.scaler_.fit(X_imputed.values)

        # Fit label encoders if provided
        if y_multiclass is not None:
            self.multiclass_encoder_.fit(y_multiclass)
            families = y_multiclass.map(lambda lbl: ATTACK_FAMILY_MAPPING.get(lbl, "Other Attack"))
            self.family_encoder_.fit(families)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using parameters strictly learned during fit:
        - Prune identical columns
        - Impute using training medians
        - Scale using training scaler
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")

        X = self.clean_column_names(X)
        missing_cols = [c for c in self.feature_names_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

        X_subset = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        X_subset = X_subset.replace([np.inf, -np.inf], np.nan)
        X_imputed = X_subset.fillna(self.impute_values_)

        X_scaled = self.scaler_.transform(X_imputed.values)
        # Clip extreme outliers to [-50, 50] to protect online models from catastrophic numeric overflow
        X_scaled = np.clip(X_scaled, -50.0, 50.0)
        return X_scaled

    def fit_transform(self, X: pd.DataFrame, y_multiclass: Optional[pd.Series] = None) -> np.ndarray:
        """Fit on X and return transformed matrix."""
        return self.fit(X, y_multiclass).transform(X)

    def save(self, filepath: str) -> None:
        """Serialize preprocessor state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "NetworkDataPreprocessor":
        """Load serialized preprocessor state."""
        return joblib.load(filepath)
