"""
Protocol A: Duplicate-Safe Conventional Evaluation Engine.
Evaluates conventional classification capabilities (not temporal adaptation).
Requirements:
1. Global deduplication via SHA-256 before splitting.
2. Hash-disjoint Stratified Train / Validation / Test partitions.
3. Preprocessing fitted strictly on training partition only.
4. Operational decision thresholds calibrated strictly on validation partition.
5. Test partition used exactly once for unbiased evaluation.
6. Models:
   - Dummy Majority
   - Logistic Regression
   - Static SGD
   - Random Forest
   - HistGradientBoosting
   - Benign-trained Isolation Forest
   - Static DST (Base Heuristic)
   - Proposed Adaptive ML-DST (Batch Non-Streaming Mode)
7. Independent latency and memory profiling per model.
"""

import os
import time
import json
import gzip
import psutil
import tracemalloc
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.features.preprocessor import NetworkDataPreprocessor
from src.features.temporal import TemporalDeviationTracker
from src.models.baselines import (
    DummyMajorityBaseline,
    LogisticRegressionBaseline,
    StaticSGDBaseline,
    RandomForestBaseline,
    HistGradientBoostingBaseline,
    IsolationForestBaseline,
    StaticDSTBaseline,
)
from src.models.online_classifier import OnlineSGDClassifier
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.dst.evidence_fusion import DSTEvidenceFusion
from src.risk.adaptive_threshold import optimize_threshold_on_validation
from src.utils.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_model_inference(model: Any, X_eval: np.ndarray) -> Dict[str, float]:
    """
    Profiles isolated median, 95th percentile, and 99th percentile inference latency
    and throughput (flows per second) on individual samples.
    """
    sample_indices = np.random.choice(len(X_eval), size=min(500, len(X_eval)), replace=False)
    latencies_ms = []

    for idx in sample_indices:
        x_single = X_eval[idx : idx + 1]
        t0 = time.perf_counter()
        if hasattr(model, "predict_score"):
            _ = model.predict_score(x_single)
        elif hasattr(model, "predict_proba"):
            _ = model.predict_proba(x_single)
        else:
            _ = model.predict(x_single)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    med_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    fps = 1000.0 / max(1e-4, med_lat)

    return {
        "median_latency_ms": round(med_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "p99_latency_ms": round(p99_lat, 4),
        "throughput_fps": round(fps, 1),
    }


def run_protocol_a(
    df: pd.DataFrame,
    random_seed: int = 42,
    output_dir: str = "reports/experiments",
) -> Dict[str, Any]:
    """
    Executes Protocol A across all static and batch baselines plus proposed ML-DST.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting Protocol A: Duplicate-Safe Conventional Evaluation ===")

    from src.data.pipeline import create_hash_disjoint_splits
    train_df, val_df, test_df = create_hash_disjoint_splits(df, test_size=0.20, val_size=0.15, random_seed=random_seed)

    prep = NetworkDataPreprocessor(scaler_type="robust")
    X_tr_df, y_tr_bin, y_tr_multi, _ = prep.extract_and_clean_labels(train_df)
    X_val_df, y_val_bin, _, _ = prep.extract_and_clean_labels(val_df)
    X_te_df, y_te_bin, _, _ = prep.extract_and_clean_labels(test_df)

    # Strictly fit preprocessor on training data only
    prep.fit(X_tr_df, y_tr_multi)

    X_train = prep.transform(X_tr_df)
    y_train = y_tr_bin.values

    X_val = prep.transform(X_val_df)
    y_val = y_val_bin.values

    X_test = prep.transform(X_te_df)
    y_test = y_te_bin.values

    logger.info(f"Partitions: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    models = {
        "Dummy Majority": DummyMajorityBaseline(random_seed=random_seed),
        "Logistic Regression": LogisticRegressionBaseline(random_seed=random_seed),
        "Static SGD": StaticSGDBaseline(random_seed=random_seed),
        "Random Forest": RandomForestBaseline(n_estimators=50, random_seed=random_seed),
        "HistGradientBoosting": HistGradientBoostingBaseline(max_iter=100, random_seed=random_seed),
        "Isolation Forest (Unsupervised)": IsolationForestBaseline(n_estimators=50, random_seed=random_seed),
        "Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)": StaticDSTBaseline(),
    }

    results: Dict[str, Any] = {}
    stored_predictions: Dict[str, List[int]] = {"y_true": [int(y) for y in y_test]}
    stored_scores: Dict[str, List[float]] = {}

    for name, model in models.items():
        logger.info(f"Training and evaluating baseline: {name}...")
        tracemalloc.start()
        t0 = time.time()

        if "Isolation Forest" in name:
            # Train strictly on benign flows
            model.fit(X_train[y_train == 0])
        else:
            model.fit(X_train, y_train)

        fit_time = time.time() - t0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calibrate decision threshold on validation set
        val_scores = model.predict_proba(X_val)[:, 1]
        calib = optimize_threshold_on_validation(y_val, val_scores, mode="recall_constrained", min_recall=0.90)
        best_t = calib["optimal_threshold"]

        # Final evaluation on test partition
        test_scores = model.predict_proba(X_test)[:, 1]
        test_preds = (test_scores >= best_t).astype(int)

        perf = profile_model_inference(model, X_test)
        metrics = compute_binary_metrics(y_test, test_preds, test_scores)
        metrics["validated_threshold"] = best_t
        metrics["training_time_sec"] = round(fit_time, 3)
        metrics["peak_memory_mb"] = round(peak_mem_bytes / (1024 * 1024), 2)
        metrics.update(perf)

        results[name] = metrics
        stored_predictions[name] = [int(p) for p in test_preds]
        stored_scores[name] = [round(float(s), 4) for s in test_scores]

    # Proposed Adaptive ML-DST in Non-Streaming Batch Mode
    logger.info("Training and evaluating: Adaptive ML-DST (Batch Mode)...")
    tracemalloc.start()
    t0 = time.time()

    online_clf = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_train, y_train)
    online_clf.calibrate(X_val, y_val)

    anom_det = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=50).fit(X_train[y_train == 0])

    temp_tracker = TemporalDeviationTracker()
    temp_tracker.resolve_feature_indices(prep.feature_names_)

    # Validation-based calibration of DST source reliabilities
    p_ml_val = online_clf.predict_score(X_val)
    p_anom_val = anom_det.score(X_val)
    p_temp_val = temp_tracker.compute_batch_deviations(X_val, update_window=False)

    from src.dst.evidence_fusion import calibrate_dst_reliabilities
    dst_calib = calibrate_dst_reliabilities(y_val, p_ml_val, p_anom_val, p_temp_val, uncertainty_budget=0.15)
    logger.info(f"DST Validation Reliabilities: {dst_calib['best_reliabilities']} (Status: {dst_calib['status']})")

    dst_engine = DSTEvidenceFusion(
        uncertainty_budget=0.15,
        source_reliabilities=dst_calib["best_reliabilities"],
    )

    fit_time = time.time() - t0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Obtain validation fusion scores for threshold optimization
    fused_val = dst_engine.fuse_batch(p_ml_val, p_anom_val, p_temp_val)
    val_risks = fused_val["risk_scores"]

    calib_dst = optimize_threshold_on_validation(y_val, val_risks, mode="recall_constrained", min_recall=0.90)
    thresh_dst = calib_dst["optimal_threshold"]

    # Evaluate on test partition
    p_ml_te = online_clf.predict_score(X_test)
    p_anom_te = anom_det.score(X_test)
    p_temp_te = temp_tracker.compute_batch_deviations(X_test, update_window=False)
    fused_te = dst_engine.fuse_batch(p_ml_te, p_anom_te, p_temp_te)
    test_risks = fused_te["risk_scores"]
    test_preds_dst = (test_risks >= thresh_dst).astype(int)

    metrics_dst = compute_binary_metrics(y_test, test_preds_dst, test_risks)
    metrics_dst["validated_threshold"] = thresh_dst
    metrics_dst["training_time_sec"] = round(fit_time, 3)
    metrics_dst["peak_memory_mb"] = round(peak_mem_bytes / (1024 * 1024), 2)
    metrics_dst["mean_uncertainty"] = round(float(np.mean(fused_te["uncertainties"])), 4)
    metrics_dst["mean_conflict"] = round(float(np.mean(fused_te["conflicts"])), 4)

    # Isolated profiling
    class DSTBatchPredictor:
        def predict_score(self, x):
            pm = online_clf.predict_score(x)
            pa = anom_det.score(x)
            pt = temp_tracker.compute_batch_deviations(x, update_window=False)
            return dst_engine.fuse_batch(pm, pa, pt)["risk_scores"]

    metrics_dst.update(profile_model_inference(DSTBatchPredictor(), X_test))
    results["Adaptive ML-DST (Proposed Batch)"] = metrics_dst

    stored_predictions["Adaptive ML-DST (Proposed Batch)"] = [int(p) for p in test_preds_dst]
    stored_scores["Adaptive ML-DST (Proposed Batch)"] = [round(float(s), 4) for s in test_risks]

    # Save outputs
    res_path = os.path.join(output_dir, "protocol_a_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "Protocol A: Duplicate-Safe Conventional Evaluation",
            "timestamp": pd.Timestamp.now().isoformat(),
            "test_size": len(y_test),
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved Protocol A metrics to {res_path}")

    pred_path = os.path.join(output_dir, "protocol_a_predictions.json.gz")
    with gzip.open(pred_path, "wt", encoding="utf-8") as f:
        json.dump({
            "predictions": stored_predictions,
            "scores": stored_scores,
        }, f)
    logger.info(f"Saved compressed Protocol A prediction arrays to {pred_path}")

    return results
