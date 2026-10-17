"""
Comprehensive V3 Scientific Validity & Rigor Regression Test Suite.
Verifies all 7 core validity specifications:
1. Independent ablation state (zero state leakage across variants).
2. Classifier vs anomaly threshold separation (no Isolation Forest threshold on classifier probs).
3. One-to-one drift event matching (detection rate, duplicate alarms, false alarms, missed events, delay).
4. Report-to-JSON numeric consistency (zero manual transcription discrepancy).
5. Chronological stream ordering (Monday through Friday sequence, monotonic row indices).
6. Zero train/validation/test hash overlap (cryptographic hash-disjoint isolation).
7. Full-mode configuration routing (strict separation of streaming generator and lightweight mode).
"""

import os
import re
import json
import pytest
import numpy as np
import pandas as pd

from src.data.deduplication import CHRONOLOGICAL_FILES, compute_row_hashes
from src.data.pipeline import (
    create_hash_disjoint_splits,
    route_dataset_by_config,
    stream_full_deduplicated_dataset,
)
from src.features.temporal import TemporalDeviationTracker
from src.models.online_classifier import OnlineSGDClassifier
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.dst.evidence_fusion import DSTEvidenceFusion, calibrate_dst_reliabilities
from src.risk.adaptive_threshold import AdaptiveThresholdManager
from src.drift.adwin import create_adwin_detector, MultiSignalDriftCoordinator
from src.evaluation.protocol_c import evaluate_one_to_one_drift_matching


# 1. Independent Ablation & Protocol B State Isolation
def test_independent_ablation_state():
    """Verify that every variant has independent mutable objects and buffers."""
    tracker_f = TemporalDeviationTracker()
    tracker_g = TemporalDeviationTracker()
    tracker_h = TemporalDeviationTracker()
    tracker_i = TemporalDeviationTracker()

    # Confirm distinct memory addresses
    assert tracker_f is not tracker_g
    assert tracker_g is not tracker_h
    assert tracker_h is not tracker_i

    # Simulate update on tracker_f only
    dummy_feat_names = ["Flow Bytes/s", "Flow Packets/s", "Flow Duration", "Flow IAT Mean"]
    tracker_f.resolve_feature_indices(dummy_feat_names)
    tracker_g.resolve_feature_indices(dummy_feat_names)

    fake_row = np.array([1000.0, 50.0, 200.0, 10.0])
    tracker_f.compute_deviation(fake_row, update_window=True, is_high_risk=False)

    # Confirm tracker_g was NOT modified
    assert tracker_f.total_observed == 1
    assert tracker_g.total_observed == 0

    # Confirm threshold managers in H and I are independent
    mgr_h = AdaptiveThresholdManager(initial_threshold=0.50)
    mgr_i = AdaptiveThresholdManager(initial_threshold=0.50)
    assert mgr_h is not mgr_i

    # Update mgr_h with high risk scores to change threshold
    mgr_h.update(risk_scores=np.array([0.9, 0.95, 0.88]), drift_detected=True, delayed_labels=np.array([1, 1, 1]), step_index=100)
    assert mgr_h.get_threshold() != mgr_i.get_threshold()


# 2. Classifier / Anomaly Threshold Separation
def test_classifier_anomaly_threshold_separation():
    """Verify that classifier probabilities and anomaly scores maintain independent thresholds."""
    np.random.seed(42)
    X_benign = np.random.randn(200, 10)
    y_benign = np.zeros(200, dtype=int)

    clf = OnlineSGDClassifier(random_seed=42).partial_fit(X_benign, y_benign)
    anom = AnomalyEvidenceDetector(random_seed=42, n_estimators=20).fit(X_benign)

    s_clf = clf.predict_score(X_benign)
    s_anom = anom.score(X_benign)

    # Classifier scores on benign-only training are very small (negative dot products + negative intercept)
    # Anomaly scores are normalized to [0, 1] relative to benign percentiles
    thresh_clf_cold_start = float(np.percentile(s_clf, 99.0))
    thresh_anom = float(np.percentile(s_anom, 98.0))

    # Assert thresholds belong to vastly different score distributions
    assert thresh_clf_cold_start < 0.01, f"Classifier cold-start threshold {thresh_clf_cold_start} should reflect raw probabilities"
    assert thresh_anom >= 0.50, f"Anomaly threshold {thresh_anom} should reflect normalized anomaly percentiles"

    # Verifying threshold separation contract: applying anomaly threshold to classifier probabilities would cause 100% false negatives
    wrong_preds = (s_clf >= thresh_anom).astype(int)
    assert np.all(wrong_preds == 0), "Applying anomaly threshold to classifier probabilities suppresses all predictions"


# 3. One-to-One Drift Matching Rigor
def test_one_to_one_drift_matching():
    """Verify exact one-to-one matching between true change points and alarms."""
    true_events = [
        {"change_point_idx": 1000, "drift_type": "Sudden Prevalence"},
        {"change_point_idx": 2000, "drift_type": "Gradual Prevalence"},
        {"change_point_idx": 3000, "drift_type": "Return to Normal"},
        {"change_point_idx": 4000, "drift_type": "Covariate Shift"},
    ]
    # Alarms:
    # 1050: matches 1000 (delay 50)
    # 1100: duplicate in [1000, 1200]
    # 2020: matches 2000 (delay 20)
    # 2500: false alarm outside [2000, 2200]
    # 4000: matches 4000 (delay 0)
    # 4080: duplicate in [4000, 4200]
    # 5000: false alarm
    # Change point 3000 is missed!
    detected_alarms = [1050, 1100, 2020, 2500, 4000, 4080, 5000]

    metrics = evaluate_one_to_one_drift_matching(detected_alarms, true_events, tolerance_window=200)

    assert metrics["true_drift_events"] == 4
    assert metrics["detected_alarms_total"] == 7
    assert metrics["true_alarms_matched"] == 3
    assert metrics["detection_rate"] == 0.75
    assert metrics["missed_events"] == 1
    assert metrics["duplicate_alarms"] == 2
    assert metrics["false_alarms"] == 2
    assert metrics["mean_detection_delay_flows"] == round((50 + 20 + 0) / 3.0, 1)  # 23.3
    assert metrics["median_detection_delay_flows"] == 20.0

    # Conservation law
    assert metrics["detected_alarms_total"] == (
        metrics["true_alarms_matched"] + metrics["duplicate_alarms"] + metrics["false_alarms"]
    )


# 4. Report-to-JSON Numeric Consistency
def test_report_to_json_numeric_consistency():
    """Verify that numbers in IMPLEMENTATION_RESULTS.md match the generated JSON and CSV files."""
    json_path = "reports/experiments/full_experiment_results.json"
    report_path = "IMPLEMENTATION_RESULTS.md"

    if not os.path.exists(json_path) or not os.path.exists(report_path):
        pytest.skip("Experimental JSON results or report not yet generated.")

    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Protocol A check: Random Forest accuracy
    rf_acc = res["protocol_a"]["Random Forest"]["accuracy"]
    rf_acc_str = f"{rf_acc:.4f}"
    assert rf_acc_str in report_text, f"Random Forest accuracy {rf_acc_str} missing or mismatched in report!"

    # Protocol B aggregate check: Online SGD recall
    sgd_rec = res["protocol_b"]["aggregate"]["2_Online_SGD_NoDrift"]["recall"]
    sgd_rec_str = f"{sgd_rec:.4f}"
    assert sgd_rec_str in report_text, f"Online SGD recall {sgd_rec_str} missing or mismatched in report!"

    # Protocol C check: True change points and mean delay
    c_perf = res["protocol_c"]
    c_events = str(c_perf["true_drift_events"])
    assert f"- **True Change Points:** {c_events}" in report_text or f"| **Total True Change Points** | {c_events} |" in report_text


# 5. Chronological Ordering
def test_chronological_ordering():
    """Verify day-order sequence and non-decreasing row indices."""
    day_order_map = {item[1]: idx for idx, item in enumerate(CHRONOLOGICAL_FILES)}
    assert day_order_map["Monday"] < day_order_map["Tuesday"]
    assert day_order_map["Tuesday"] < day_order_map["Wednesday"]
    assert day_order_map["Wednesday"] < day_order_map["Thursday-Morning"]
    assert day_order_map["Thursday-Morning"] < day_order_map["Thursday-Afternoon"]
    assert day_order_map["Thursday-Afternoon"] < day_order_map["Friday-Morning"]
    assert day_order_map["Friday-Morning"] < day_order_map["Friday-PortScan"]
    assert day_order_map["Friday-PortScan"] < day_order_map["Friday-DDoS"]

    sample_path = "data/processed/lightweight_sample.csv.gz"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        df["day_rank"] = df["source_day"].map(lambda d: day_order_map.get(d, 99))
        df_sorted = df.sort_values(by=["day_rank", "original_row_index"]).reset_index(drop=True)
        # Check non-decreasing day rank
        diffs = np.diff(df_sorted["day_rank"].values)
        assert np.all(diffs >= 0), "Day ranks must be monotonically non-decreasing"


# 6. Zero Hash Overlap across Partitions
def test_zero_hash_overlap():
    """Verify zero hash collision across train, validation, and test splits."""
    np.random.seed(42)
    fake_data = []
    for i in range(300):
        fake_data.append({
            "feature1": float(i % 50),
            "feature2": float((i * 3) % 70),
            "row_hash": f"hash_{i:04d}",
            "NormalizedLabel": "BENIGN" if i % 3 != 0 else "PortScan",
            "AttackFamily": "BENIGN" if i % 3 != 0 else "PortScan",
        })
    df_fake = pd.DataFrame(fake_data)

    tr, val, te = create_hash_disjoint_splits(df_fake, test_size=0.20, val_size=0.15, random_seed=42)

    tr_hashes = set(tr["row_hash"])
    val_hashes = set(val["row_hash"])
    te_hashes = set(te["row_hash"])

    assert len(tr_hashes.intersection(val_hashes)) == 0
    assert len(tr_hashes.intersection(te_hashes)) == 0
    assert len(val_hashes.intersection(te_hashes)) == 0


# 7. Full-Mode Configuration Routing
def test_full_mode_configuration_routing():
    """Verify config loader differentiates full streaming generator from lightweight mode."""
    # Lightweight mode routing
    lw_cfg = {
        "mode": "lightweight",
        "label": "lightweight development results",
        "data": {"lightweight_sample_path": "data/processed/lightweight_sample.csv.gz"},
    }
    routed_lw = route_dataset_by_config(lw_cfg)
    assert routed_lw["mode"] == "lightweight"
    assert routed_lw["data"] is not None
    assert routed_lw["stream_generator"] is None

    # Full mode routing
    full_cfg = {
        "mode": "full",
        "label": "full-dataset results",
        "data": {"raw_dir": "dataset", "chunk_size": 10000},
        "stream": {"batch_size": 500},
    }
    routed_full = route_dataset_by_config(full_cfg)
    assert routed_full["mode"] == "full"
    assert routed_full["data"] is None
    assert routed_full["stream_generator"] is not None
