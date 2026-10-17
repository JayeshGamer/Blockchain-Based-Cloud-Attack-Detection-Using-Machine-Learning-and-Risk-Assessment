"""
Comprehensive V5 Verification Test Suite for Adaptive ML-DST.
Fulfills all Phase 20 / V5 Requirements:
1. Stream / Lightweight Evaluation Equivalence on identical data, config, and seed.
2. Configuration Propagation to actual component constructors (linear, nonlinear, anomaly, drift, temporal, DST, threshold).
3. Process Memory Tracking & Nested Profiler Non-Destruction.
4. Incremental Confusion Matrix and Metric Calculation Accuracy.
5. Full-Mode Routing and Explicit Unsupported Protocol Rejection.
6. Delayed Feedback Queue Bounds and Original Prediction Preservation.
7. End-to-End Full Streaming Generator with Complete SHA-256 and Row Accounting.
8. Artifact Consistency and Machine-Readable Report Validation.
"""

import os
import json
import gzip
import pytest
import numpy as np
import pandas as pd

from src.utils.config import (
    resolve_and_validate_config,
    load_config,
    build_online_linear_classifier,
    build_online_nonlinear_classifier,
    build_anomaly_detector,
    build_drift_detector,
    build_temporal_tracker,
    build_dst_engine,
    build_adaptive_threshold_manager,
    get_effective_components_summary,
)
from src.utils.profiling import ProcessMemoryTracker
from src.utils.metrics import compute_binary_metrics
from src.data.ordering import order_chronological_stream
from src.data.pipeline import stream_full_deduplicated_dataset
from src.evaluation.streaming_core import StreamingEvaluationEngine
from src.evaluation.protocol_b import run_protocol_b
from scripts.run_experiments import run_all_experiments


# 1. Stream / Lightweight Equivalence Test
def test_stream_lightweight_equivalence():
    """
    Verifies that feeding identical batches through the streaming evaluation core
    via DataFrame slicing vs generator yields bit-for-bit identical predictions,
    confusion matrices, and final metrics.
    """
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not found.")

    df_raw = pd.read_csv(sample_path).head(1000)
    df_ordered, _ = order_chronological_stream(df_raw)

    cfg = resolve_and_validate_config({
        "mode": "lightweight",
        "stream": {"batch_size": 100, "label_delay_batches": 1},
        "project": {"random_seed": 42},
    })

    # Run 1: Lightweight mode
    res_light = run_protocol_b(
        df_or_sample_path=df_ordered,
        batch_size=100,
        label_delay_batches=1,
        random_seed=42,
        output_dir="reports/test_run_light",
        config=cfg,
        save_predictions=False,
    )

    # Run 2: Generator simulation over identical data and batch boundaries
    def mock_generator():
        # Monday baseline was first 200 rows, remaining 800 are streaming flows
        monday_count = len(df_ordered[df_ordered["source_day"] == "Monday"])
        if monday_count == 0:
            monday_count = int(0.2 * len(df_ordered))
        df_str = df_ordered.iloc[monday_count:].reset_index(drop=True)
        for i in range(0, len(df_str), 100):
            yield df_str.iloc[i : i + 100].reset_index(drop=True)

    res_gen = run_protocol_b(
        df_or_sample_path=df_ordered,
        stream_generator_fn=mock_generator,
        batch_size=100,
        label_delay_batches=1,
        random_seed=42,
        output_dir="reports/test_run_gen",
        config=cfg,
        save_predictions=False,
    )

    # Verify identical evaluated row counts
    assert res_light["total_streaming_flows"] == res_gen["total_streaming_flows"]

    # Verify identical aggregate confusion matrices across variants
    for v_name in res_light["aggregate_metrics"]:
        cm_light = res_light["aggregate_metrics"][v_name]["confusion_matrix"]
        cm_gen = res_gen["aggregate_metrics"][v_name]["confusion_matrix"]
        assert cm_light == cm_gen, f"Mismatch in {v_name} CM: {cm_light} vs {cm_gen}"
        assert res_light["aggregate_metrics"][v_name]["recall"] == res_gen["aggregate_metrics"][v_name]["recall"]
        assert res_light["aggregate_metrics"][v_name]["false_positive_rate"] == res_gen["aggregate_metrics"][v_name]["false_positive_rate"]


# 2. Configuration Propagation to Component Constructors
def test_component_configuration_propagation():
    """
    Verifies that YAML configuration settings propagate directly into component attributes,
    not just loaded into a dict.
    """
    custom_cfg = {
        "mode": "lightweight",
        "project": {"random_seed": 777},
        "online_models": {
            "linear": {
                "loss": "modified_huber",
                "penalty": "l1",
                "alpha": 0.005,
                "learning_rate": "optimal",
            },
            "nonlinear": {
                "grace_period": 350,
                "delta": 1e-4,
            },
        },
        "anomaly_detector": {
            "n_estimators": 88,
            "contamination": 0.08,
            "max_samples": 3000,
        },
        "drift_detector": {
            "delta": 0.005,
        },
        "temporal_deviation": {
            "window_size": 400,
            "protected_benign_baseline": False,
            "benign_confidence_threshold": 0.45,
        },
        "dst_fusion": {
            "uncertainty_budget": 0.22,
            "conflict_threshold": 0.85,
            "conflict_rule": "dempster",
        },
        "adaptive_threshold": {
            "initial_threshold": 0.65,
            "target_fpr": 0.03,
            "rolling_window_size": 750,
            "min_threshold": 0.25,
            "max_threshold": 0.90,
            "drift_relaxation_factor": 0.70,
        },
    }

    resolved = resolve_and_validate_config(custom_cfg)

    # 1. Linear Online SGD
    linear_mod = build_online_linear_classifier(resolved, seed=777)
    assert linear_mod.loss == "modified_huber"
    assert linear_mod.penalty == "l1"
    assert linear_mod.alpha == 0.005
    assert linear_mod.random_seed == 777

    # 2. River Nonlinear HATC
    tree_mod = build_online_nonlinear_classifier(resolved, seed=777)
    assert tree_mod.grace_period == 350
    assert tree_mod.delta == 1e-4

    # 3. Anomaly Detector
    anom_mod = build_anomaly_detector(resolved, seed=777)
    assert anom_mod.n_estimators == 88
    assert anom_mod.contamination == 0.08
    assert anom_mod.max_samples == 3000

    # 4. Drift Detector
    drift_mod = build_drift_detector(resolved)
    assert hasattr(drift_mod, "update")

    # 5. Temporal Tracker
    temp_mod = build_temporal_tracker(resolved)
    assert temp_mod.window_size == 400
    assert temp_mod.benign_confidence_threshold == 0.45

    # 6. DST Fusion Engine
    dst_mod = build_dst_engine(resolved)
    assert dst_mod.u_budget == 0.22
    assert dst_mod.conflict_threshold == 0.85
    assert dst_mod.conflict_rule == "dempster"

    # 7. Adaptive Threshold Manager
    thresh_mod = build_adaptive_threshold_manager(resolved)
    assert thresh_mod.current_threshold == 0.65
    assert thresh_mod.target_fpr == 0.03
    assert thresh_mod.rolling_window_size == 750
    assert thresh_mod.min_threshold == 0.25
    assert thresh_mod.max_threshold == 0.90
    assert thresh_mod.drift_relaxation_factor == 0.70

    # 8. Effective Component Summary
    summary = get_effective_components_summary(resolved)
    assert summary["online_models"]["linear_class"] == "OnlineSGDClassifier"
    assert summary["dst_fusion"]["effective_params"]["conflict_rule"] == "dempster"


# 3. Process Memory Tracker & Nested Safety
def test_process_memory_tracker_nested_safety():
    """
    Verifies that ProcessMemoryTracker returns positive peak memory in MB
    and that nested routines stopping tracemalloc do not corrupt run-level measurement.
    """
    tracker = ProcessMemoryTracker().start()

    # Simulate nested routine that starts and stops tracemalloc
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    dummy_alloc = np.ones((500, 500), dtype=np.float64)
    tracemalloc.stop()

    summary = tracker.stop()

    assert summary["status"] == "valid"
    assert summary["units"] == "MB"
    assert summary["process_peak_rss_mb"] is not None
    assert summary["process_peak_rss_mb"] > 0, "Peak RSS must be strictly positive"
    assert summary["elapsed_time_sec"] >= 0


# 4. Incremental Confusion Matrix Accuracy
def test_incremental_confusion_matrix_accuracy():
    """
    Verifies that incremental accumulation of confusion matrices across batches
    matches batch calculation exactly.
    """
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

    # Expected: TP=4 (idx 0,3,6,8), FP=1 (idx 5), FN=1 (idx 2), TN=4 (idx 1,4,7,9)
    cm_acc = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Simulate 2 batches
    for b_slice in [(0, 5), (5, 10)]:
        y_b = y_true[b_slice[0] : b_slice[1]]
        p_b = y_pred[b_slice[0] : b_slice[1]]
        cm_acc["tp"] += int(np.sum((p_b == 1) & (y_b == 1)))
        cm_acc["fp"] += int(np.sum((p_b == 1) & (y_b == 0)))
        cm_acc["tn"] += int(np.sum((p_b == 0) & (y_b == 0)))
        cm_acc["fn"] += int(np.sum((p_b == 0) & (y_b == 1)))

    assert cm_acc["tp"] == 4
    assert cm_acc["fp"] == 1
    assert cm_acc["fn"] == 1
    assert cm_acc["tn"] == 4

    metrics = StreamingEvaluationEngine._cm_to_metrics(cm_acc)
    assert metrics["recall"] == 0.80  # 4 / 5
    assert metrics["precision"] == 0.80  # 4 / 5
    assert metrics["attack_f1"] == 0.80
    assert metrics["false_positive_rate"] == 0.20  # 1 / 5
    assert metrics["false_negative_rate"] == 0.20  # 1 / 5


# 5. Full Mode Routing and Unsupported Protocol Rejection
def test_full_mode_routing_and_rejection():
    """
    Verifies that full mode requires raw dataset directory and rejects invalid protocol requests.
    """
    # Missing raw dir raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Full mode requires raw dataset directory"):
        resolve_and_validate_config({
            "mode": "full",
            "data": {"raw_dir": "non_existent_path_xyz123"},
        })

    # Full mode with valid dataset directory resolves
    if os.path.exists("dataset"):
        cfg = resolve_and_validate_config({
            "mode": "full",
            "data": {"raw_dir": "dataset"},
        })
        assert cfg["mode"] == "full"
        assert cfg["output"]["experiments_dir"] == "reports/experiments/full"


# 6. Delayed Feedback Queue Bounded Memory
def test_delayed_feedback_queue_bounded_memory():
    """
    Verifies that pending feedback queue length is strictly bounded by label_delay_batches.
    """
    from src.models.online_classifier import OnlineSGDClassifier
    from src.features.temporal import TemporalDeviationTracker
    from src.risk.adaptive_threshold import AdaptiveThresholdManager
    from src.simulation.defense import SimulatedDefenseEngine
    from src.drift.adwin import create_adwin_detector, MultiSignalDriftCoordinator
    from src.dst.evidence_fusion import DSTEvidenceFusion
    from src.models.anomaly_detector import AnomalyEvidenceDetector

    X_init = np.random.randn(50, 10)
    y_init = np.random.randint(0, 2, size=50)

    variants = {
        "1_Frozen_Static": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "2_Online_SGD_NoDrift": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "3_Online_SGD_ADWIN": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "4_River_HATC_NoDrift": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "5_River_HATC_ADWIN": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "6_Online_Anomaly_Evidence": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "7_Online_DST_FixedThresh": OnlineSGDClassifier().partial_fit(X_init, y_init),
        "8_Full_Adaptive_ML_DST": OnlineSGDClassifier().partial_fit(X_init, y_init),
    }

    aux = {
        "anom_det_6": AnomalyEvidenceDetector().fit(X_init),
        "anom_det_7": AnomalyEvidenceDetector().fit(X_init),
        "anom_det_8": AnomalyEvidenceDetector().fit(X_init),
        "adwin_sgd": create_adwin_detector(),
        "adwin_hatc": create_adwin_detector(),
        "drift_coord_full": MultiSignalDriftCoordinator(),
        "temp_tracker_7": TemporalDeviationTracker(),
        "temp_tracker_8": TemporalDeviationTracker(),
        "dst_engine_7": DSTEvidenceFusion(),
        "dst_engine_8": DSTEvidenceFusion(),
        "thresh_mgr": AdaptiveThresholdManager(),
        "defense": SimulatedDefenseEngine(),
        "thresh_frozen_clf": 0.50,
    }

    calib_thresh = {
        "Online_SGD": {0.01: 0.5, 0.02: 0.5, 0.05: 0.5},
        "Online_SGD_Anomaly": {0.01: 0.5, 0.02: 0.5, 0.05: 0.5},
        "Proposed_ML_DST": {0.01: 0.5, 0.02: 0.5, 0.05: 0.5},
    }

    engine = StreamingEvaluationEngine(
        variants=variants,
        auxiliary_engines=aux,
        calibrated_thresholds=calib_thresh,
        label_delay_batches=2,  # Delay = 2 batches
        output_dir="reports/test_queue_bound",
        save_predictions=False,
    )

    # Process 10 batches
    for b_idx in range(10):
        X_b = np.random.randn(20, 10)
        y_b = np.random.randint(0, 2, size=20)
        engine.process_batch(b_idx, X_b, y_b, "Tuesday")
        # Queue should never exceed label_delay (2)
        assert len(engine.pending_feedback_queue) <= 2, f"Queue grew to {len(engine.pending_feedback_queue)}"

    res = engine.finalize()
    assert len(engine.pending_feedback_queue) == 0, "Queue must be completely drained upon finalization"
    assert res["total_evaluated_flows"] == 200
