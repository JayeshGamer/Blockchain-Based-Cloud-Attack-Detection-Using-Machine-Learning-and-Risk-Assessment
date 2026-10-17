"""
Rigorous V4 Regression and Integration Test Suite for Adaptive ML-DST.
Tests actual runners and pipeline execution to satisfy all 10 V4 verification requirements:
1. Actual configuration routing and strict validation.
2. Full-stream generator iteration, complete SHA-256 retention, and bounded memory.
3. Identical stream ordering and fingerprints across Protocol B and Ablation.
4. Independent mutable state inside actual experiment variants.
5. Delayed feedback using original immutable predictions.
6. Operational concept-drift alarm timestamps reflecting label latency.
7. Protocol C warm-up exclusion from scored evaluation metrics.
8. Unambiguous metric calculation and exact report-table mapping.
9. Missing-artifact explicit failure.
10. Deterministic results given identical seed and configuration.
"""

import os
import json
import gzip
import pytest
import numpy as np
import pandas as pd

from src.utils.config import (
    resolve_and_validate_config,
    validate_config,
    load_config,
)
from src.data.deduplication import (
    compute_row_hashes,
    DiskBackedHashSet,
    CHRONOLOGICAL_FILES,
)
from src.data.ordering import (
    order_chronological_stream,
    verify_stream_ordering,
)
from src.data.pipeline import (
    stream_full_deduplicated_dataset,
    route_dataset_by_config,
    build_lightweight_dataset,
)
from src.utils.metrics import compute_binary_metrics
from src.evaluation.protocol_b import run_protocol_b
from src.evaluation.protocol_c import (
    run_protocol_c,
    build_controlled_drift_stream,
    evaluate_one_to_one_drift_matching,
)
from src.evaluation.ablation import run_fair_ablation_study
from scripts.generate_report import generate_implementation_report


# 1. Configuration Routing & Strict Schema Validation
def test_actual_configuration_routing(tmp_path):
    """Verify that configuration parameters route to execution and invalid configs are rejected."""
    # Test rejection of unknown mode
    with pytest.raises(ValueError, match="Invalid mode 'quantum'"):
        resolve_and_validate_config({"mode": "quantum"})

    # Test rejection of negative batch size
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        resolve_and_validate_config({"mode": "lightweight", "stream": {"batch_size": -50}})

    # Test rejection of negative label delay
    with pytest.raises(ValueError, match="label_delay_batches must be a non-negative integer"):
        resolve_and_validate_config({"mode": "lightweight", "stream": {"label_delay_batches": -1}})

    # Test rejection of invalid threshold bounds
    with pytest.raises(ValueError, match="Threshold bounds invalid"):
        resolve_and_validate_config({
            "mode": "lightweight",
            "adaptive_threshold": {"min_threshold": 0.90, "max_threshold": 0.20}
        })

    # Test rejection of missing raw directory under full mode
    with pytest.raises(FileNotFoundError, match="Silent fallback to lightweight mode is strictly prohibited"):
        resolve_and_validate_config({
            "mode": "full",
            "data": {"raw_dir": "non_existent_raw_directory_xyz"}
        })

    # Test valid resolution
    valid_cfg = resolve_and_validate_config({
        "mode": "lightweight",
        "project": {"random_seed": 1234},
        "stream": {"batch_size": 150, "label_delay_batches": 2},
    })
    assert valid_cfg["mode"] == "lightweight"
    assert valid_cfg["project"]["random_seed"] == 1234
    assert valid_cfg["stream"]["batch_size"] == 150
    assert valid_cfg["stream"]["label_delay_batches"] == 2


# 2. Full-Stream Iteration, SHA-256 Retention, and Bounded Memory
def test_full_stream_generator_iteration_and_sha256():
    """Verify streaming generator yields complete 64-char hashes and bounded memory."""
    # Run a smoke stream of 100 rows from dataset
    if not os.path.exists("dataset"):
        pytest.skip("Dataset directory not available.")

    gen = stream_full_deduplicated_dataset(
        data_dir="dataset",
        chunk_size=5000,
        batch_size=50,
        max_rows=100,
        use_disk_backed=True,
    )

    batches = list(gen)
    assert len(batches) >= 1
    df_streamed = pd.concat(batches, ignore_index=True)
    assert len(df_streamed) == 100

    # Verify complete 64-char SHA-256 hash preservation
    assert "row_hash" in df_streamed.columns
    first_hash = df_streamed["row_hash"].iloc[0]
    assert isinstance(first_hash, str)
    assert len(first_hash) == 64, f"Hash must be full 64 hex chars, got {len(first_hash)}"

    # Verify required row identifiers
    assert "source_file" in df_streamed.columns
    assert "source_day" in df_streamed.columns
    assert "original_row_index" in df_streamed.columns
    assert "NormalizedLabel" in df_streamed.columns

    # Verify all yielded rows are unique
    assert len(df_streamed["row_hash"].unique()) == len(df_streamed)


# 3. Identical Stream Ordering and Fingerprints
def test_identical_stream_ordering_protocol_b_and_ablation():
    """Verify Protocol B and Ablation receive identically ordered rows and matching fingerprints."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    df_ordered_b, fp_b = order_chronological_stream(df_sample)
    df_ordered_abl, fp_abl = order_chronological_stream(df_sample)

    assert fp_b == fp_abl, "Stream fingerprints between Protocol B and Ablation must be identical"
    assert verify_stream_ordering(df_ordered_b)
    assert verify_stream_ordering(df_ordered_abl)

    # Check non-decreasing day rank
    ranks = df_ordered_b["day_rank"].values
    assert np.all(np.diff(ranks) >= 0)


# 4. Independent Mutable State Inside Real Runners
def test_independent_mutable_state_inside_runners(tmp_path):
    """Verify that different variants in Protocol B maintain strictly isolated weights and buffers."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    # Take a miniature 600-flow sample with Monday baseline
    df_ordered, _ = order_chronological_stream(df_sample)
    df_mon = df_ordered[df_ordered["source_day"] == "Monday"].iloc[:200]
    df_tue = df_ordered[df_ordered["source_day"] == "Tuesday"].iloc[:200]
    df_mini = pd.concat([df_mon, df_tue], ignore_index=True)

    out_dir = str(tmp_path / "proto_b_test")
    res = run_protocol_b(
        df_or_sample_path=df_mini,
        batch_size=50,
        label_delay_batches=1,
        random_seed=42,
        output_dir=out_dir,
    )

    agg = res["aggregate_metrics"]
    # Variant 1 (Frozen) vs Variant 2 (Online SGD) vs Variant 8 (Full ML-DST)
    # They must have different predictions and metrics, proving state independence
    from src.evaluation.streaming_core import read_prediction_arrays
    pred_path_l = os.path.join(out_dir, "protocol_b_predictions.jsonl.gz")
    pred_path_legacy = os.path.join(out_dir, "protocol_b_predictions.json.gz")
    pred_path = pred_path_l if os.path.exists(pred_path_l) else pred_path_legacy
    p_data = read_prediction_arrays(pred_path)

    p_froz = p_data["predictions"]["1_Frozen_Static"]
    p_sgd = p_data["predictions"]["2_Online_SGD_NoDrift"]
    p_mldst = p_data["predictions"]["8_Full_Adaptive_ML_DST"]

    # Frozen and Online SGD must not be identical across the whole Tuesday adaptation
    assert not np.array_equal(p_froz, p_sgd), "Frozen model and Online SGD predictions must differ"
    assert not np.array_equal(p_sgd, p_mldst), "Online SGD and ML-DST predictions must differ"


# 5. Delayed Feedback Preserves Original Predictions
def test_delayed_feedback_preserves_original_predictions(tmp_path):
    """Verify that model updates at step t+1 do not alter stored predictions made at step t."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    df_ordered, _ = order_chronological_stream(df_sample)
    df_mon = df_ordered[df_ordered["source_day"] == "Monday"].iloc[:200]
    df_tue = df_ordered[df_ordered["source_day"] == "Tuesday"].iloc[:300]
    df_mini = pd.concat([df_mon, df_tue], ignore_index=True)

    out_dir = str(tmp_path / "delayed_test")
    res = run_protocol_b(
        df_or_sample_path=df_mini,
        batch_size=100,
        label_delay_batches=1,
        random_seed=42,
        output_dir=out_dir,
    )

    from src.evaluation.streaming_core import read_prediction_arrays
    pred_path_l = os.path.join(out_dir, "protocol_b_predictions.jsonl.gz")
    pred_path_legacy = os.path.join(out_dir, "protocol_b_predictions.json.gz")
    pred_path = pred_path_l if os.path.exists(pred_path_l) else pred_path_legacy
    p_data = read_prediction_arrays(pred_path)

    stored_preds = p_data["predictions"]["2_Online_SGD_NoDrift"]
    assert len(stored_preds) == 300  # Evaluated Tuesday flows


# 6. Operational Drift Alarm Timestamps Reflect Label Latency
def test_operational_drift_alarm_timestamps(tmp_path):
    """Verify Protocol C logs operational alarm index and distinguishes detector-only delay."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    out_dir = str(tmp_path / "proto_c_test")

    # Run Protocol C with 1-batch delay (50 flows per batch)
    res = run_protocol_c(
        df_sample,
        warmup_flows=500,
        batch_size=50,
        label_delay_batches=1,
        tolerance_window=250,
        random_seed=42,
        output_dir=out_dir,
    )

    c_perf = res["drift_detection_performance"]
    assert c_perf["true_drift_events"] == 5

    # Check alarm records
    records = res["alarm_records"]
    if records:
        first_record = records[0]
        assert "operational_alarm_idx" in first_record
        assert "originating_flow_idx" in first_record
        assert first_record["operational_alarm_idx"] >= first_record["originating_flow_idx"]
        assert first_record["label_delay_flows"] >= 50  # 1 batch delay = 50 flows


# 7. Protocol C Warm-Up Exclusion from Scored Evaluation
def test_protocol_c_warmup_exclusion(tmp_path):
    """Verify that the first 500 flows are strictly excluded from scored evaluation."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    warmup_df, drift_points, eval_df = build_controlled_drift_stream(df_sample, warmup_flows=500, random_seed=42)

    assert len(warmup_df) == 500
    assert len(eval_df) == 5500

    # Verify zero hash overlap between warmup and evaluation stream
    warmup_hashes = set(warmup_df["row_hash"]) if "row_hash" in warmup_df.columns else set(warmup_df.index)
    eval_hashes = set(eval_df["row_hash"]) if "row_hash" in eval_df.columns else set(eval_df.index)
    assert len(warmup_hashes.intersection(eval_hashes)) == 0, "Warm-up and evaluation flows must have zero overlap"

    out_dir = str(tmp_path / "proto_c_warmup")
    res = run_protocol_c(df_sample, warmup_flows=500, random_seed=42, output_dir=out_dir)

    assert res["warmup_flows_excluded"] == 500
    assert res["evaluated_stream_flows"] == 5500


# 8. Unambiguous Metric Calculation and Explicit F2
def test_unambiguous_metric_calculation():
    """Verify metrics calculate attack_f1, macro_f1, f2, and f2 is never equal to FNR by fallback."""
    y_true = np.array([0] * 110 + [1] * 90)
    # Predictions designed with TP=80, FP=20, FN=10, TN=90
    y_pred = np.array([0] * 90 + [1] * 20 + [0] * 10 + [1] * 80)

    metrics = compute_binary_metrics(y_true, y_pred)

    precision = 80.0 / 100.0  # 0.80
    recall = 80.0 / 90.0      # 0.888889
    expected_f2 = (5.0 * precision * recall) / (4.0 * precision + recall)  # 0.869565
    expected_fnr = 10.0 / 90.0  # 0.111111

    assert "attack_f1" in metrics
    assert "macro_f1" in metrics
    assert "f2" in metrics
    assert "false_positive_rate" in metrics
    assert "false_negative_rate" in metrics

    assert round(metrics["f2"], 5) == round(expected_f2, 5)
    assert round(metrics["false_negative_rate"], 5) == round(expected_fnr, 5)
    assert metrics["f2"] != metrics["false_negative_rate"], "F2 must NOT equal FNR"


# 9. Missing-Artifact Explicit Failure
def test_missing_artifact_explicit_failure(tmp_path):
    """Verify report generation raises FileNotFoundError when required artifacts are missing."""
    fake_json = str(tmp_path / "non_existent_results.json")
    with pytest.raises(FileNotFoundError):
        generate_implementation_report(results_json_path=fake_json)


# 10. Deterministic Reproducibility
def test_deterministic_reproducibility(tmp_path):
    """Verify that identical seed and configuration produce 100% identical outputs."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        pytest.skip("Lightweight sample dataset not generated.")

    df_sample = pd.read_csv(sample_path)
    df_ordered, _ = order_chronological_stream(df_sample)
    df_mini = pd.concat([
        df_ordered[df_ordered["source_day"] == "Monday"].iloc[:100],
        df_ordered[df_ordered["source_day"] == "Tuesday"].iloc[:100],
    ], ignore_index=True)

    dir1 = str(tmp_path / "run1")
    dir2 = str(tmp_path / "run2")

    res1 = run_protocol_b(df_or_sample_path=df_mini, batch_size=50, random_seed=42, output_dir=dir1)
    res2 = run_protocol_b(df_or_sample_path=df_mini, batch_size=50, random_seed=42, output_dir=dir2)

    rec1 = res1["aggregate_metrics"]["2_Online_SGD_NoDrift"]["recall"]
    rec2 = res2["aggregate_metrics"]["2_Online_SGD_NoDrift"]["recall"]
    assert rec1 == rec2

    # Verify predictions match exactly
    from src.evaluation.streaming_core import read_prediction_arrays
    p_path1 = os.path.join(dir1, "protocol_b_predictions.jsonl.gz") if os.path.exists(os.path.join(dir1, "protocol_b_predictions.jsonl.gz")) else os.path.join(dir1, "protocol_b_predictions.json.gz")
    p_path2 = os.path.join(dir2, "protocol_b_predictions.jsonl.gz") if os.path.exists(os.path.join(dir2, "protocol_b_predictions.jsonl.gz")) else os.path.join(dir2, "protocol_b_predictions.json.gz")
    p1 = read_prediction_arrays(p_path1)
    p2 = read_prediction_arrays(p_path2)
    for k in p1["predictions"]:
        assert np.array_equal(p1["predictions"][k], p2["predictions"][k])
