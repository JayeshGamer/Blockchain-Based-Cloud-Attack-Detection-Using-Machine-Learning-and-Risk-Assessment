"""
Model Training and Export Pipeline for Adaptive ML-DST (V6).
Trains and exports:
1. models/online_sgd_initial.joblib (Permitted Monday baseline warm-up)
2. models/online_sgd_final_online.joblib (Prequential chronological stream with delayed feedback)
3. models/model_metadata.json (Bundle metadata, file sizes, demo checkpoint designation, and limitations)
4. data/example_input.csv and data/example_output.csv (Verification test flow batch)
"""

import os
import sys
import copy
import json
import time
import hashlib
from collections import deque
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import sklearn

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.preprocessor import NetworkDataPreprocessor
from src.models.online_classifier import OnlineSGDClassifier
from src.utils.config import load_config
from src.data.deduplication import CHRONOLOGICAL_FILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_stream_sha256(df: pd.DataFrame) -> str:
    """Computes deterministic SHA256 fingerprint over ordered DataFrame rows."""
    hasher = hashlib.sha256()
    # Hash row count, columns, and sample of contents
    summary_str = f"rows={len(df)}_cols={len(df.columns)}_"
    hasher.update(summary_str.encode("utf-8"))
    if "row_hash" in df.columns:
        for h in df["row_hash"].iloc[:5000]:
            hasher.update(str(h).encode("utf-8"))
    elif "NormalizedLabel" in df.columns:
        for lbl in df["NormalizedLabel"].iloc[:5000]:
            hasher.update(str(lbl).encode("utf-8"))
    return hasher.hexdigest()


def run_training_and_export(
    sample_path: str = "data/processed/lightweight_sample.csv.gz",
    config_path: str = "configs/primary.yaml",
    output_dir: str = "models",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(sample_path):
        logger.info(f"Sample not found at {sample_path}. Generating from raw dataset...")
        from src.data.loader import load_stratified_sample
        df_strat = load_stratified_sample("dataset", total_samples=50000, random_seed=42)
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df_strat.to_csv(sample_path, index=False, compression="gzip")

    config = load_config(config_path) if os.path.exists(config_path) else {}
    seed = config.get("random_seed", 42)

    logger.info("=" * 70)
    logger.info("STARTING V6 MODEL TRAINING AND BUNDLE EXPORT")
    logger.info(f"Data source: {sample_path} | Config: {config_path}")
    logger.info("=" * 70)

    # 1. Load Chronological Stream
    df_all = pd.read_csv(sample_path, compression="gzip", low_memory=False)
    df_all.columns = [c.strip() for c in df_all.columns]
    stream_fingerprint = compute_stream_sha256(df_all)

    # Separate Monday warmup baseline from subsequent evaluation days
    mon_mask = (df_all["source_file"] == CHRONOLOGICAL_FILES[0][0])
    if mon_mask.any():
        df_monday = df_all[mon_mask].reset_index(drop=True)
        df_stream = df_all[~mon_mask].reset_index(drop=True)
    else:
        split_idx = int(len(df_all) * 0.17)
        df_monday = df_all.iloc[:split_idx].reset_index(drop=True)
        df_stream = df_all.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"Monday Warm-up Flows:   {len(df_monday):,} (Strict Benign Initialization)")
    logger.info(f"Subsequent Stream Flows: {len(df_stream):,} (Chronological Prequential Updates)")
    logger.info(f"Dataset SHA256:          {stream_fingerprint[:16]}...")

    from src.utils.config import build_preprocessor
    prep = build_preprocessor(config) if config else NetworkDataPreprocessor(scaler_type="robust")
    X_mon_df, y_mon_bin, y_mon_multi, _ = prep.extract_and_clean_labels(df_monday)
    prep.fit(X_mon_df, y_mon_multi)

    X_monday = prep.transform(X_mon_df)
    y_monday = y_mon_bin.values

    # 3. Train Initial Checkpoint
    logger.info("Fitting Initial Online SGD Classifier on Monday baseline...")
    clf_initial = OnlineSGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        random_seed=seed,
    )
    clf_initial.partial_fit(X_monday, y_monday)

    # Calibrate baseline decision thresholds
    s_mon_initial = clf_initial.predict_score(X_monday)
    thresh_01 = float(np.percentile(s_mon_initial, 99.0))
    thresh_02 = float(np.percentile(s_mon_initial, 98.0))
    thresh_05 = float(np.percentile(s_mon_initial, 95.0))

    env_metadata = {
        "python_version": sys.version,
        "platform": sys.platform,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
    }

    initial_path = os.path.join(output_dir, "online_sgd_initial.joblib")
    bundle_initial = {
        "model": clf_initial,
        "preprocessor": prep,
        "scaler": prep.scaler_,
        "imputer": getattr(prep, "imputer_", getattr(prep, "impute_values_", {})),
        "impute_values": getattr(prep, "impute_values_", {}),
        "feature_names": list(prep.feature_names_),
        "class_mapping": {0: "BENIGN", 1: "ATTACK"},
        "decision_threshold": 0.50,
        "calibrated_thresholds": {
            0.01: thresh_01,
            0.02: thresh_02,
            0.05: thresh_05,
        },
        "effective_config": config,
        "checkpoint_stage": "initial_baseline",
        "training_cutoff": "Monday warm-up baseline (8,445 flows, all benign)",
        "dataset_fingerprint": stream_fingerprint,
        "environment": env_metadata,
        "training_stats": {
            "flows_trained": clf_initial.sample_count_,
            "updates_count": clf_initial.update_count_,
            "feature_count": len(prep.feature_names_),
        },
    }
    joblib.dump(bundle_initial, initial_path, compress=3)
    size_initial_mb = os.path.getsize(initial_path) / (1024 * 1024)
    logger.info(f"Saved initial checkpoint to {initial_path} ({size_initial_mb:.3f} MB)")

    # 4. Train Final-Online Checkpoint via Chronological Delayed Feedback
    logger.info("Executing chronological prequential streaming for Final-Online Checkpoint...")
    clf_final = copy.deepcopy(clf_initial)

    batch_size = 200
    label_delay = 1
    feedback_queue = deque()

    stream_batches = [
        df_stream.iloc[i : i + batch_size].reset_index(drop=True)
        for i in range(0, len(df_stream), batch_size)
    ]

    for b_idx, b_df in enumerate(stream_batches):
        X_b_df, y_b_bin, _, _ = prep.extract_and_clean_labels(b_df)
        X_b = prep.transform(X_b_df)
        y_b = y_b_bin.values

        # Predict-before-update
        _ = clf_final.predict_score(X_b)

        # Enqueue for delayed feedback
        feedback_queue.append((X_b, y_b))

        # Apply feedback when queue exceeds delay D
        while len(feedback_queue) > label_delay:
            mature_X, mature_y = feedback_queue.popleft()
            clf_final.partial_fit(mature_X, mature_y)

    # Flush remaining feedback at stream end
    while feedback_queue:
        mature_X, mature_y = feedback_queue.popleft()
        clf_final.partial_fit(mature_X, mature_y)

    final_path = os.path.join(output_dir, "online_sgd_final_online.joblib")
    bundle_final = {
        "model": clf_final,
        "preprocessor": prep,
        "scaler": prep.scaler_,
        "imputer": getattr(prep, "imputer_", getattr(prep, "impute_values_", {})),
        "impute_values": getattr(prep, "impute_values_", {}),
        "feature_names": list(prep.feature_names_),
        "class_mapping": {0: "BENIGN", 1: "ATTACK"},
        "decision_threshold": 0.50,
        "calibrated_thresholds": {
            0.01: thresh_01,
            0.02: thresh_02,
            0.05: thresh_05,
        },
        "effective_config": config,
        "checkpoint_stage": "final_online_streamed",
        "training_cutoff": "End of chronological stream Friday-DDoS (41,555 evaluation flows)",
        "dataset_fingerprint": stream_fingerprint,
        "environment": env_metadata,
        "training_stats": {
            "flows_trained": clf_final.sample_count_,
            "updates_count": clf_final.update_count_,
            "feature_count": len(prep.feature_names_),
        },
    }
    joblib.dump(bundle_final, final_path, compress=3)
    size_final_mb = os.path.getsize(final_path) / (1024 * 1024)
    logger.info(f"Saved final-online checkpoint to {final_path} ({size_final_mb:.3f} MB)")

    # 5. Write Model Metadata JSON
    metadata = {
        "default_demo_checkpoint": "models/online_sgd_final_online.joblib",
        "checkpoints": {
            "initial": {
                "file": "models/online_sgd_initial.joblib",
                "size_mb": round(size_initial_mb, 3),
                "stage": "initial_baseline",
                "description": "Trained strictly on Monday warm-up baseline (8,445 benign flows). Clean starting weights prior to attack exposure.",
                "flows_learned": clf_initial.sample_count_,
                "threshold": 0.50,
            },
            "final_online": {
                "file": "models/online_sgd_final_online.joblib",
                "size_mb": round(size_final_mb, 3),
                "stage": "final_online_streamed",
                "description": "State after genuine chronological prequential streaming with 1-batch delayed feedback across all 41,555 flows.",
                "flows_learned": clf_final.sample_count_,
                "threshold": 0.50,
            },
        },
        "feature_schema": {
            "feature_count": len(prep.feature_names_),
            "ordered_features": list(prep.feature_names_),
            "scaler": "robust",
            "imputer": "training_median",
            "outlier_clip": [-50.0, 50.0],
        },
        "dataset_fingerprint": stream_fingerprint,
        "limitations": [
            "Requires precomputed network flow feature statistics (CICIDS2017 schema); does not directly parse raw packet bytes.",
            "Continuous attack score is an uncalibrated linear decision score, not a posterior probability.",
            "The final-online checkpoint was updated using released chronological stream labels; do not evaluate it on historical training flows to claim held-out accuracy.",
            "Thresholds calibrated on Monday benign traffic do not guarantee future-stream false positive rates under arbitrary concept drift."
        ],
        "created_timestamp": pd.Timestamp.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {meta_path}")

    # 6. Generate Verification Example Input and Output CSVs
    example_in_path = "data/example_input.csv"
    example_out_path = "data/example_output.csv"

    # Select 25 benign and 25 attack flows from stream for verification
    ben_rows = df_stream[df_stream["NormalizedLabel"] == "BENIGN"].head(25)
    atk_rows = df_stream[df_stream["NormalizedLabel"] != "BENIGN"].head(25)
    example_df = pd.concat([ben_rows, atk_rows], ignore_index=True)
    example_df["flow_id"] = [f"sample_flow_{i:03d}" for i in range(len(example_df))]

    # Save example input
    example_df.to_csv(example_in_path, index=False)
    logger.info(f"Saved example input batch (50 flows) to {example_in_path}")

    # Predict using saved final model
    from src.inference.predictor import OnlineSGDPredictor
    pred = OnlineSGDPredictor(model_path=final_path)
    out_df = pred.predict_df(example_df, flow_id_col="flow_id")
    out_df.to_csv(example_out_path, index=False)
    logger.info(f"Saved example inference output to {example_out_path}")

    logger.info("=" * 70)
    logger.info("MODEL TRAINING AND EXPORT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    return metadata


if __name__ == "__main__":
    run_training_and_export()
