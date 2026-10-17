"""
Production Saved-Model Inference Module for Adaptive ML-DST (V6).
Loads exported joblib model bundles, enforces strict feature schema validation,
handles extra/missing columns, and computes attack scores without modifying model state.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class OnlineSGDPredictor:
    """
    Inference engine for saved Online SGD model bundles.
    Guarantees:
    1. Zero weight mutation during inference (frozen evaluation).
    2. Strict feature schema verification.
    3. Graceful handling of extra metadata columns.
    4. Uncalibrated attack scores clearly demarcated from calibrated posterior probabilities.
    """

    def __init__(
        self,
        model_path: str = "models/online_sgd_final_online.joblib",
        default_threshold: Optional[float] = None,
    ):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint bundle not found at '{self.model_path}'. "
                f"Please run 'python scripts/train_export.py' to generate trained model artifacts."
            )

        logger.info(f"Loading saved model bundle from: {self.model_path}")
        self.bundle: Dict[str, Any] = joblib.load(self.model_path)

        # Validate bundle integrity
        required_keys = ["model", "preprocessor", "feature_names", "decision_threshold"]
        for k in required_keys:
            if k not in self.bundle:
                raise KeyError(f"Invalid model bundle: missing required key '{k}'")

        self.model = self.bundle["model"]
        self.preprocessor = self.bundle["preprocessor"]
        self.feature_names: List[str] = list(self.bundle["feature_names"])
        self.class_mapping: Dict[int, str] = self.bundle.get("class_mapping", {0: "BENIGN", 1: "ATTACK"})
        self.threshold: float = float(default_threshold if default_threshold is not None else self.bundle["decision_threshold"])
        self.checkpoint_stage: str = self.bundle.get("checkpoint_stage", "unknown")
        self.dataset_fingerprint: str = self.bundle.get("dataset_fingerprint", "unknown")
        self.calibrated_thresholds: Dict[Any, float] = self.bundle.get("calibrated_thresholds", {})

        logger.info(
            f"Successfully loaded {self.checkpoint_stage} model with {len(self.feature_names)} features. "
            f"Active decision threshold: {self.threshold:.4f}"
        )

    def predict_df(
        self,
        df: pd.DataFrame,
        flow_id_col: Optional[str] = None,
        threshold_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Executes frozen inference on a DataFrame of raw network flow features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing raw network flow measurements (CICIDS2017 schema).
        flow_id_col : Optional[str]
            Column name containing unique flow identifiers. If None, detects or generates IDs.
        threshold_override : Optional[float]
            Optional threshold override for this inference batch.
            
        Returns:
        --------
        pd.DataFrame with columns: ['flow_id', 'attack_score', 'threshold', 'prediction', 'predicted_label']
        """
        if df.empty:
            return pd.DataFrame(columns=["flow_id", "attack_score", "threshold", "prediction", "predicted_label"])

        # Strip whitespace from incoming column names
        df_clean = df.copy()
        df_clean.columns = [str(c).strip() for c in df_clean.columns]

        # Extract flow IDs
        if flow_id_col and flow_id_col in df_clean.columns:
            flow_ids = df_clean[flow_id_col].astype(str).tolist()
        elif "flow_id" in df_clean.columns:
            flow_ids = df_clean["flow_id"].astype(str).tolist()
        elif "Flow ID" in df_clean.columns:
            flow_ids = df_clean["Flow ID"].astype(str).tolist()
        elif "row_hash" in df_clean.columns:
            flow_ids = df_clean["row_hash"].astype(str).tolist()
        else:
            flow_ids = [f"flow_{i}" for i in range(len(df_clean))]

        # Validate required feature columns
        missing = [f for f in self.feature_names if f not in df_clean.columns]
        if missing:
            raise ValueError(
                f"Input data is missing {len(missing)} required feature column(s). "
                f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Expected {len(self.feature_names)} precomputed network flow features according to CICIDS2017 schema."
            )

        # Reorder and isolate strictly expected feature columns (safely dropping extraneous columns)
        feature_df = df_clean[self.feature_names].copy()

        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(feature_df)

        # Predict continuous attack score (note: linear decision score or sigmoid score, uncalibrated)
        scores = self.model.predict_score(X_trans)

        thresh = float(threshold_override if threshold_override is not None else self.threshold)
        preds = (scores >= thresh).astype(int)
        labels = [self.class_mapping.get(p, "ATTACK" if p == 1 else "BENIGN") for p in preds]

        result_df = pd.DataFrame({
            "flow_id": flow_ids,
            "attack_score": np.round(scores, 6),
            "threshold": thresh,
            "prediction": preds,
            "predicted_label": labels,
        })
        return result_df

    def predict_single(
        self,
        features: Dict[str, Any],
        flow_id: str = "flow_0",
        threshold_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Inference endpoint for a single flow feature dictionary (used by REST API).
        """
        df_single = pd.DataFrame([features])
        df_single["flow_id"] = flow_id
        res_df = self.predict_df(df_single, flow_id_col="flow_id", threshold_override=threshold_override)
        row = res_df.iloc[0]
        return {
            "flow_id": str(row["flow_id"]),
            "attack_score": float(row["attack_score"]),
            "threshold": float(row["threshold"]),
            "prediction": int(row["prediction"]),
            "predicted_label": str(row["predicted_label"]),
            "is_attack": bool(row["prediction"] == 1),
            "model_stage": self.checkpoint_stage,
            "note": "Attack score is an uncalibrated linear streaming score, not a posterior probability.",
        }
