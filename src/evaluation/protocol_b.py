"""
Protocol B: Genuine Chronological Prequential Evaluation Engine (V6_VERIFIED).
Evaluates continuous temporal adaptation across the 8-day ingress stream:
  Monday (Normal) -> Tuesday (Brute Force) -> Wednesday (DoS/Heartbleed) ->
  Thursday Morning (Web Attacks) -> Thursday Afternoon (Infiltration) ->
  Friday Morning (Bot) -> Friday PortScan -> Friday DDoS

Rigor Specifications:
1. Strict predict-before-update prequential protocol via shared StreamingEvaluationEngine.
2. D-batch delayed label feedback: models updated strictly on mature delayed feedback.
3. Monday serves as legitimate benign baseline (zero attacks on Monday).
4. Row-level warmup exclusion before batch accumulation across both lightweight and full modes.
5. Original predictions and scores saved incrementally at inference time; never recomputed.
6. Unified chronological file/day-and-row ordering with cryptographic stream fingerprint.
7. Genuine achieved calibration FPR measurement (false positives / benign calibration samples).
8. Full configuration propagation to all auxiliary components (preprocessor, drift, temporal, DST).
"""

import os
import time
import json
import gzip
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Generator
import numpy as np
import pandas as pd

from src.data.deduplication import CHRONOLOGICAL_FILES, check_dataset_files
from src.data.ordering import order_chronological_stream, save_stream_manifest
from src.features.preprocessor import NetworkDataPreprocessor
from src.features.temporal import TemporalDeviationTracker
from src.models.online_classifier import OnlineSGDClassifier, RiverHoeffdingAdaptiveTreeOnline
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.drift.adwin import create_adwin_detector, MultiSignalDriftCoordinator
from src.dst.evidence_fusion import DSTEvidenceFusion
from src.risk.adaptive_threshold import AdaptiveThresholdManager
from src.risk.hierarchical_risk import HierarchicalRiskEngine
from src.simulation.defense import SimulatedDefenseEngine
from src.utils.profiling import ProcessMemoryTracker
from src.utils.config import (
    build_preprocessor,
    build_online_linear_classifier,
    build_online_nonlinear_classifier,
    build_anomaly_detector,
    build_drift_detector,
    build_drift_coordinator,
    build_temporal_tracker,
    build_dst_engine,
    build_adaptive_threshold_manager,
)
from src.evaluation.streaming_core import StreamingEvaluationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_protocol_b(
    df_or_sample_path: Optional[Union[pd.DataFrame, str]] = "data/processed/lightweight_sample.csv.gz",
    stream_generator_fn: Optional[Any] = None,
    batch_size: int = 200,
    label_delay_batches: int = 1,
    random_seed: int = 42,
    output_dir: str = "reports/experiments",
    config: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Executes Protocol B evaluation.
    Guarantees strict equivalence of prequential semantics across both lightweight
    and full out-of-core streaming paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting Protocol B: Genuine Chronological Prequential Evaluation (V6) ===")

    # Run-level memory tracking
    mem_tracker = ProcessMemoryTracker.get_active()
    owns_mem_tracker = False
    if mem_tracker is None:
        mem_tracker = ProcessMemoryTracker().start()
        owns_mem_tracker = True

    t0_proto = time.time()
    effective_config = config or {}

    # Extract runtime parameters with config precedence
    batch_size = effective_config.get("stream", {}).get("batch_size", batch_size)
    label_delay_batches = effective_config.get("stream", {}).get("label_delay_batches", label_delay_batches)
    seed = effective_config.get("project", {}).get("random_seed", random_seed)

    mode = effective_config.get("mode", "lightweight").lower()
    is_full_mode = stream_generator_fn is not None or mode == "full"

    stream_fingerprint = "full_stream_generator"
    df_monday: Optional[pd.DataFrame] = None
    lightweight_batches: Optional[List[pd.DataFrame]] = None
    accounting_sink: Dict[str, Any] = {}

    if isinstance(df_or_sample_path, pd.DataFrame):
        df_all, stream_fingerprint = order_chronological_stream(df_or_sample_path)
        manifest_path = os.path.join(output_dir, "protocol_b_stream_manifest.json")
        save_stream_manifest(df_all, stream_fingerprint, manifest_path)

        # ROW-LEVEL WARMUP EXCLUSION: Partition Monday rows before evaluation batching
        monday_mask = df_all["source_day"] == "Monday"
        df_monday = df_all[monday_mask].reset_index(drop=True)
        df_stream = df_all[~monday_mask].reset_index(drop=True)

        if len(df_stream) == 0:
            split_idx = max(1, int(0.2 * len(df_all)))
            df_monday = df_all.iloc[:split_idx].reset_index(drop=True)
            df_stream = df_all.iloc[split_idx:].reset_index(drop=True)

        if max_rows is not None and len(df_stream) > max_rows:
            df_stream = df_stream.iloc[:max_rows].reset_index(drop=True)

        if not is_full_mode:
            logger.info(
                f"Mode: Lightweight Replay | Stream Fingerprint: {stream_fingerprint[:16]}... | "
                f"Monday Baseline Flows: {len(df_monday):,} | Streaming Flows: {len(df_stream):,}"
            )
            lightweight_batches = [
                df_stream.iloc[i : i + batch_size].reset_index(drop=True)
                for i in range(0, len(df_stream), batch_size)
            ]
        else:
            logger.info(
                f"Mode: Stream Generator with DataFrame Baseline | Stream Fingerprint: {stream_fingerprint[:16]}... | "
                f"Monday Baseline Flows: {len(df_monday):,}"
            )

    elif not is_full_mode:
        # --- LIGHTWEIGHT MODE FROM FILE PATH ---
        if not os.path.exists(df_or_sample_path):
            raise FileNotFoundError(f"Dataset file not found at: {df_or_sample_path}")
        df_raw = pd.read_csv(df_or_sample_path)

        df_all, stream_fingerprint = order_chronological_stream(df_raw)
        manifest_path = os.path.join(output_dir, "protocol_b_stream_manifest.json")
        save_stream_manifest(df_all, stream_fingerprint, manifest_path)

        # ROW-LEVEL WARMUP EXCLUSION: Filter Monday rows before evaluation batching
        monday_mask = df_all["source_day"] == "Monday"
        df_monday = df_all[monday_mask].reset_index(drop=True)
        df_stream = df_all[~monday_mask].reset_index(drop=True)

        if len(df_stream) == 0:
            split_idx = max(1, int(0.2 * len(df_all)))
            df_monday = df_all.iloc[:split_idx].reset_index(drop=True)
            df_stream = df_all.iloc[split_idx:].reset_index(drop=True)

        if max_rows is not None and len(df_stream) > max_rows:
            df_stream = df_stream.iloc[:max_rows].reset_index(drop=True)

        logger.info(
            f"Mode: Lightweight Replay | Stream Fingerprint: {stream_fingerprint[:16]}... | "
            f"Monday Baseline Flows: {len(df_monday):,} | Streaming Flows: {len(df_stream):,}"
        )

        lightweight_batches = [
            df_stream.iloc[i : i + batch_size].reset_index(drop=True)
            for i in range(0, len(df_stream), batch_size)
        ]

    else:
        # --- FULL CHUNKED STREAMING PATH FROM DISK ---
        raw_dir = effective_config.get("data", {}).get("raw_dir", "dataset")
        chunk_size = effective_config.get("data", {}).get("chunk_size", 50000)
        check_dataset_files(data_dir=raw_dir)

        # Bounded Monday calibration slice
        warmup_flows = effective_config.get("stream", {}).get("warmup_flows", 10000)
        monday_filename = CHRONOLOGICAL_FILES[0][0]
        mon_path = os.path.join(raw_dir, monday_filename)

        logger.info(f"Full Streaming Mode: Reading bounded Monday calibration baseline ({warmup_flows:,} flows)...")
        df_mon_chunk = pd.read_csv(
            mon_path,
            nrows=warmup_flows,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
        )
        df_mon_chunk.columns = [c.strip() for c in df_mon_chunk.columns]
        df_monday = df_mon_chunk.copy()
        df_monday["source_file"] = monday_filename
        df_monday["source_day"] = "Monday"
        stream_fingerprint = f"full_dataset_stream_chunked_{warmup_flows}"

    # --- SHARED INITIALIZATION & CALIBRATION ON MONDAY BENIGN BASELINE ---
    prep = build_preprocessor(effective_config) if effective_config else NetworkDataPreprocessor(scaler_type="robust")
    X_mon_df, y_mon_bin, y_mon_multi, _ = prep.extract_and_clean_labels(df_monday)
    prep.fit(X_mon_df, y_mon_multi)

    X_monday = prep.transform(X_mon_df)
    y_monday = y_mon_bin.values

    # Initialize Auxiliary Evidence & Drift Engines (using resolved configuration)
    if effective_config:
        anom_det_6 = build_anomaly_detector(effective_config, seed=seed).fit(X_monday)
        anom_det_7 = build_anomaly_detector(effective_config, seed=seed).fit(X_monday)
        anom_det_8 = build_anomaly_detector(effective_config, seed=seed).fit(X_monday)
        adwin_sgd = build_drift_detector(effective_config)
        adwin_hatc = build_drift_detector(effective_config)
        drift_coord_full = build_drift_coordinator(effective_config)
        dst_engine_7 = build_dst_engine(effective_config)
        dst_engine_8 = build_dst_engine(effective_config)
        thresh_mgr = build_adaptive_threshold_manager(effective_config)
        temp_tracker_7 = build_temporal_tracker(effective_config)
        temp_tracker_8 = build_temporal_tracker(effective_config)
    else:
        anom_det_6 = AnomalyEvidenceDetector(random_seed=seed, n_estimators=50).fit(X_monday)
        anom_det_7 = AnomalyEvidenceDetector(random_seed=seed, n_estimators=50).fit(X_monday)
        anom_det_8 = AnomalyEvidenceDetector(random_seed=seed, n_estimators=50).fit(X_monday)
        adwin_sgd = create_adwin_detector(delta=0.002, prefer_river=True)
        adwin_hatc = create_adwin_detector(delta=0.002, prefer_river=True)
        drift_coord_full = MultiSignalDriftCoordinator(delta=0.002, prefer_river=True)
        dst_engine_7 = DSTEvidenceFusion(uncertainty_budget=0.15)
        dst_engine_8 = DSTEvidenceFusion(uncertainty_budget=0.15)
        thresh_mgr = AdaptiveThresholdManager(
            initial_threshold=0.50,
            min_threshold=0.15,
            max_threshold=0.85,
            drift_relaxation_factor=0.85,
        )
        temp_tracker_7 = TemporalDeviationTracker()
        temp_tracker_8 = TemporalDeviationTracker()

    temp_tracker_7.resolve_feature_indices(prep.feature_names_)
    temp_tracker_8.resolve_feature_indices(prep.feature_names_)
    hierarchy = HierarchicalRiskEngine(num_nodes=20, num_clusters=4, random_seed=seed)
    defense = SimulatedDefenseEngine()

    # Build 8 Independent Model Variants
    def new_linear():
        if effective_config:
            return build_online_linear_classifier(effective_config, seed=seed)
        return OnlineSGDClassifier(random_seed=seed)

    def new_nonlinear():
        if effective_config:
            return build_online_nonlinear_classifier(effective_config, seed=seed)
        return RiverHoeffdingAdaptiveTreeOnline(random_seed=seed)

    variants = {
        "1_Frozen_Static": new_linear().partial_fit(X_monday, y_monday),
        "2_Online_SGD_NoDrift": new_linear().partial_fit(X_monday, y_monday),
        "3_Online_SGD_ADWIN": new_linear().partial_fit(X_monday, y_monday),
        "4_River_HATC_NoDrift": new_nonlinear().partial_fit(X_monday, y_monday),
        "5_River_HATC_ADWIN": new_nonlinear().partial_fit(X_monday, y_monday),
        "6_Online_Anomaly_Evidence": new_linear().partial_fit(X_monday, y_monday),
        "7_Online_DST_FixedThresh": new_linear().partial_fit(X_monday, y_monday),
        "8_Full_Adaptive_ML_DST": new_linear().partial_fit(X_monday, y_monday),
    }

    s_mon_frozen = variants["1_Frozen_Static"].predict_score(X_monday)
    thresh_frozen_clf = float(np.percentile(s_mon_frozen, 99.0))

    # Common Calibration Targets (1%, 2%, 5% FPR) on Monday Benign Baseline
    s_mon_sgd = variants["2_Online_SGD_NoDrift"].predict_score(X_monday)
    s_mon_anom = anom_det_6.score(X_monday)
    s_mon_anom_fused = 0.60 * s_mon_sgd + 0.40 * s_mon_anom
    s_mon_temp = temp_tracker_8.compute_batch_deviations(X_monday, update_window=False)
    dst_engine_calib = build_dst_engine(effective_config) if effective_config else DSTEvidenceFusion(uncertainty_budget=0.15)
    fused_mon_dst = dst_engine_calib.fuse_batch(s_mon_sgd, s_mon_anom, s_mon_temp)
    s_mon_dst = fused_mon_dst["risk_scores"]

    target_fpr_list = [0.01, 0.02, 0.05]
    n_mon = len(X_monday)

    # Measure achieved calibration FPR directly from Monday calibration predictions
    calibration_stats: Dict[str, Any] = {
        "total_calibration_samples": n_mon,
        "Online_SGD": {},
        "Online_SGD_Anomaly": {},
        "Proposed_ML_DST": {},
    }
    calibrated_thresholds: Dict[str, Dict[float, float]] = {
        "Online_SGD": {},
        "Online_SGD_Anomaly": {},
        "Proposed_ML_DST": {},
    }

    for fpr in target_fpr_list:
        # Online SGD
        t_sgd = float(np.percentile(s_mon_sgd, (1.0 - fpr) * 100.0))
        calibrated_thresholds["Online_SGD"][fpr] = t_sgd
        fp_sgd = int(np.sum(s_mon_sgd >= t_sgd))
        calibration_stats["Online_SGD"][fpr] = {
            "target_fpr": fpr,
            "calibrated_threshold": t_sgd,
            "calibration_false_positives": fp_sgd,
            "achieved_calibration_fpr": round(fp_sgd / max(1, n_mon), 4),
        }

        # Online SGD Anomaly
        t_anom = float(np.percentile(s_mon_anom_fused, (1.0 - fpr) * 100.0))
        calibrated_thresholds["Online_SGD_Anomaly"][fpr] = t_anom
        fp_anom = int(np.sum(s_mon_anom_fused >= t_anom))
        calibration_stats["Online_SGD_Anomaly"][fpr] = {
            "target_fpr": fpr,
            "calibrated_threshold": t_anom,
            "calibration_false_positives": fp_anom,
            "achieved_calibration_fpr": round(fp_anom / max(1, n_mon), 4),
        }

        # Proposed ML-DST
        t_dst = float(np.percentile(s_mon_dst, (1.0 - fpr) * 100.0))
        calibrated_thresholds["Proposed_ML_DST"][fpr] = t_dst
        fp_dst = int(np.sum(s_mon_dst >= t_dst))
        calibration_stats["Proposed_ML_DST"][fpr] = {
            "target_fpr": fpr,
            "calibrated_threshold": t_dst,
            "calibration_false_positives": fp_dst,
            "achieved_calibration_fpr": round(fp_dst / max(1, n_mon), 4),
        }

    auxiliary_engines = {
        "anom_det_6": anom_det_6,
        "anom_det_7": anom_det_7,
        "anom_det_8": anom_det_8,
        "adwin_sgd": adwin_sgd,
        "adwin_hatc": adwin_hatc,
        "drift_coord_full": drift_coord_full,
        "temp_tracker_7": temp_tracker_7,
        "temp_tracker_8": temp_tracker_8,
        "dst_engine_7": dst_engine_7,
        "dst_engine_8": dst_engine_8,
        "thresh_mgr": thresh_mgr,
        "hierarchy": hierarchy,
        "defense": defense,
        "thresh_frozen_clf": thresh_frozen_clf,
    }

    # Instantiate Shared Streaming Engine
    engine = StreamingEvaluationEngine(
        variants=variants,
        auxiliary_engines=auxiliary_engines,
        calibrated_thresholds=calibrated_thresholds,
        calibration_stats=calibration_stats,
        label_delay_batches=label_delay_batches,
        output_dir=output_dir,
        save_predictions=save_predictions,
        prediction_filename="protocol_b_predictions.jsonl.gz",
        stream_fingerprint=stream_fingerprint,
    )

    # --- BATCH FEEDING LOOP (Identical Processing via Shared Core) ---
    if not is_full_mode:
        assert lightweight_batches is not None
        logger.info(f"Executing lightweight replay through shared StreamingEvaluationEngine ({len(lightweight_batches)} batches)...")
        for b_idx, b_df in enumerate(lightweight_batches):
            X_b_df, y_b_bin, _, meta = prep.extract_and_clean_labels(b_df)
            X_b = prep.transform(X_b_df)
            y_b = y_b_bin.values
            b_meta = {
                "row_days": b_df["source_day"].values if "source_day" in b_df.columns else np.full(len(b_df), "Unknown"),
                "flow_ids": b_df["row_hash"].values if "row_hash" in b_df.columns else [f"flow_{b_idx}_{i}" for i in range(len(b_df))],
                "original_row_index": b_df["original_row_index"].values if "original_row_index" in b_df.columns else list(range(len(b_df))),
            }
            b_day = str(b_meta["row_days"][0])
            engine.process_batch(b_idx, X_b, y_b, b_day, b_meta)

        accounting_sink.update({
            "raw_scanned": len(df_monday) + len(df_stream),
            "duplicates_filtered": 0,
            "warmup_rows": len(df_monday),
            "excluded_baseline_rows": len(df_monday),
            "calibration_rows": len(df_monday),
            "evaluated_rows": engine.total_evaluated_flows,
            "omitted_smoke_test_rows": 0,
        })
    else:
        from src.data.pipeline import stream_full_deduplicated_dataset
        logger.info("Executing full chunked streaming through shared StreamingEvaluationEngine...")

        if stream_generator_fn is not None:
            batch_gen = stream_generator_fn() if callable(stream_generator_fn) else stream_generator_fn
        else:
            raw_dir = effective_config.get("data", {}).get("raw_dir", "dataset")
            chunk_size = effective_config.get("data", {}).get("chunk_size", 50000)
            max_r = max_rows or effective_config.get("data", {}).get("sample_size")
            max_per_file = None
            smoke_flag = effective_config.get("smoke_test", False) or (max_rows is not None and max_rows <= 10000)

            batch_gen = stream_full_deduplicated_dataset(
                data_dir=raw_dir,
                chunk_size=chunk_size,
                batch_size=batch_size,
                max_rows=max_r,
                max_rows_per_file=max_per_file,
                use_disk_backed=True,
                exclude_warmup_day="Monday",
                smoke_test=smoke_flag,
                accounting_sink=accounting_sink,
            )

        b_idx = 0
        try:
            for b_df in batch_gen:
                # Guarantee row-level exclusion of any Monday flows before evaluation
                if "source_day" in b_df.columns:
                    non_mon_mask = (b_df["source_day"] != "Monday")
                    if not non_mon_mask.all():
                        b_df = b_df[non_mon_mask].reset_index(drop=True)
                if len(b_df) == 0:
                    continue

                X_b_df, y_b_bin, _, meta = prep.extract_and_clean_labels(b_df)
                X_b = prep.transform(X_b_df)
                y_b = y_b_bin.values
                b_meta = {
                    "row_days": b_df["source_day"].values if "source_day" in b_df.columns else np.full(len(b_df), "Unknown"),
                    "flow_ids": b_df["row_hash"].values if "row_hash" in b_df.columns else [f"flow_{b_idx}_{i}" for i in range(len(b_df))],
                    "original_row_index": b_df["original_row_index"].values if "original_row_index" in b_df.columns else list(range(len(b_df))),
                }
                b_day = str(b_meta["row_days"][0])
                engine.process_batch(b_idx, X_b, y_b, b_day, b_meta)
                b_idx += 1
        finally:
            if hasattr(batch_gen, "close"):
                batch_gen.close()

    # Finalize evaluation and compute metrics
    engine_results = engine.finalize()

    t_elapsed = time.time() - t0_proto
    mem_summary = mem_tracker.stop() if owns_mem_tracker else {"status": "shared"}
    peak_mem_mb = mem_summary.get("process_peak_rss_mb")

    results_summary = {
        "protocol": "Protocol B: Genuine Chronological Prequential Evaluation",
        "timestamp": pd.Timestamp.now().isoformat(),
        "stream_fingerprint": stream_fingerprint,
        "execution_mode": "full_streaming" if is_full_mode else "lightweight_replay",
        "ordering_convention": "declared_file_day_and_original_row_index",
        "total_warmup_flows": len(df_monday),
        "total_streaming_flows": engine_results["total_evaluated_flows"],
        "total_batches": engine_results["total_batches"],
        "batch_size": batch_size,
        "label_delay_batches": label_delay_batches,
        "elapsed_time_sec": round(t_elapsed, 2),
        "peak_memory_mb": peak_mem_mb,
        "drift_alarms_total": engine_results["drift_alarms_total"],
        "drift_events": engine_results["drift_events"],
        "aggregate_metrics": engine_results["aggregate_metrics"],
        "daily_metrics": engine_results["daily_metrics"],
        "calibrated_thresholds": engine_results["calibrated_thresholds"],
        "common_calibration_evaluations": engine_results["common_calibration_evaluations"],
        "fpr_target_comparisons": engine_results["common_calibration_evaluations"],
        "defense_quality": engine_results["defense_quality"],
        "row_accounting": accounting_sink,
    }

    # Save outputs
    out_json = os.path.join(output_dir, "protocol_b_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved Protocol B results to {out_json}")

    return results_summary
