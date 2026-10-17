"""
Unified Experiment Runner for Adaptive ML-DST (V5_REVIEW).
Connects YAML configurations to actual component factories and streaming execution.
Supports:
1. Lightweight Development Mode (Protocols A, B, C, and Ablation Study).
2. Genuine Full Streaming Mode (Protocol B streaming via chunked deduplication).
3. Full-Path Streaming Smoke Test (--smoke-test).
4. Run-level process memory profiling via ProcessMemoryTracker.
5. Export of machine-readable JSON artifacts and verified CSV tables.
"""

import os
import sys
import time
import json
import gzip
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (
    resolve_and_validate_config,
    save_resolved_config,
    get_effective_components_summary,
)
from src.utils.profiling import ProcessMemoryTracker
from src.data.ordering import order_chronological_stream
from src.data.pipeline import build_lightweight_dataset, stream_full_deduplicated_dataset
from src.data.deduplication import check_dataset_files
from src.evaluation.protocol_a import run_protocol_a
from src.evaluation.protocol_b import run_protocol_b
from src.evaluation.protocol_c import run_protocol_c
from src.evaluation.ablation import run_fair_ablation_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    """Computes McNemar's test for paired binary classifier comparison."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if (b + c) == 0:
        return {"chi2_statistic": 0.0, "p_value": 1.0, "statistically_significant": False}

    stat = float((abs(b - c) - 1) ** 2 / (b + c))
    from scipy.stats import chi2
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "contingency_b": b,
        "contingency_c": c,
        "chi2_statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "statistically_significant": bool(p_val < 0.05),
    }


def run_all_experiments(
    config_path: str = "configs/lightweight.yaml",
    random_seed: Optional[int] = None,
    output_dir_override: Optional[str] = None,
    smoke_test: bool = False,
    max_rows: Optional[int] = None,
    protocols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes the evaluation suite driven strictly by resolved configuration.
    """
    # 1. Start Run-Level Memory Profiling
    mem_tracker = ProcessMemoryTracker().start()
    t_start_total = time.time()

    # 2. Resolve and validate configuration
    logger.info(f"Loading and resolving configuration from: {config_path}")
    config = resolve_and_validate_config(config_path)

    if random_seed is not None:
        config.setdefault("project", {})["random_seed"] = random_seed
    seed = config.get("project", {}).get("random_seed", 42)

    mode = config.get("mode", "lightweight").lower()
    label = config.get("label", "lightweight development results" if mode == "lightweight" else "full-dataset results")

    if smoke_test:
        label = "full-streaming smoke test"
        if max_rows is None:
            max_rows = 5000

    # Determine isolated output directory
    if output_dir_override:
        output_dir = output_dir_override
    elif smoke_test:
        output_dir = "reports/experiments/smoke_test"
    elif mode == "full":
        output_dir = "reports/experiments/full_run"
    else:
        output_dir = "reports/experiments/lightweight"

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save resolved configuration and effective component summary
    save_resolved_config(config, output_dir)
    eff_components = get_effective_components_summary(config)
    with open(os.path.join(output_dir, "effective_components.json"), "w", encoding="utf-8") as f:
        json.dump(eff_components, f, indent=2)

    batch_size = config.get("stream", {}).get("batch_size", 200)
    label_delay_batches = config.get("stream", {}).get("label_delay_batches", 1)

    logger.info(f"=== Execution Mode: {mode.upper()} ({label}) | Seed: {seed} | Batch Size: {batch_size} | Delay: {label_delay_batches} ===")
    logger.info(f"Artifacts output directory: {output_dir}")

    # --- EXECUTION PATHS ---
    proto_a_results = None
    proto_b_results = None
    proto_c_results = None
    ablation_results = None
    statistical_report = None
    stream_fingerprint = ""

    if mode == "full":
        # Check that raw dataset files exist
        raw_dir = config.get("data", {}).get("raw_dir", "dataset")
        check_dataset_files(data_dir=raw_dir)

        # Full mode executes Protocol B streaming
        logger.info("Executing Full-Mode Streaming Protocol B runner...")
        proto_b_results = run_protocol_b(
            df_or_sample_path=None,
            batch_size=batch_size,
            label_delay_batches=label_delay_batches,
            random_seed=seed,
            output_dir=output_dir,
            config=config,
            max_rows=max_rows,
        )
        stream_fingerprint = proto_b_results.get("stream_fingerprint", "full_stream")

    else:
        # Lightweight execution across all 4 protocols
        lightweight_path = config.get("data", {}).get("lightweight_sample_path", "data/processed/lightweight_sample.csv.gz")
        if not os.path.exists(lightweight_path):
            logger.info(f"Lightweight dataset not found at {lightweight_path}. Generating...")
            df_sample = build_lightweight_dataset(output_path=lightweight_path, random_seed=seed)
        else:
            df_sample = pd.read_csv(lightweight_path)

        df_ordered, stream_fingerprint = order_chronological_stream(df_sample)
        logger.info(f"Loaded and ordered {len(df_ordered):,} flows | Fingerprint: {stream_fingerprint[:16]}...")

        # 1. Execute Protocol A: Conventional Hash-Disjoint Evaluation
        proto_a_results = run_protocol_a(df_ordered, random_seed=seed, output_dir=output_dir)

        # 2. Execute Protocol B: Chronological Prequential Evaluation
        proto_b_results = run_protocol_b(
            df_or_sample_path=df_ordered,
            batch_size=batch_size,
            label_delay_batches=label_delay_batches,
            random_seed=seed,
            output_dir=output_dir,
            config=config,
            max_rows=max_rows,
        )

        # 3. Execute Protocol C: Controlled Concept-Drift Benchmark
        proto_c_results = run_protocol_c(
            df_ordered,
            warmup_flows=500,
            batch_size=50,
            label_delay_batches=label_delay_batches,
            tolerance_window=250,
            random_seed=seed,
            output_dir=output_dir,
        )

        # 4. Execute Phase 17 Fair Ablation Study
        ablation_results = run_fair_ablation_study(
            df_ordered,
            batch_size=batch_size,
            label_delay_batches=label_delay_batches,
            random_seed=seed,
            output_dir=output_dir,
        )

        # 5. Statistical Significance Tests (McNemar)
        with gzip.open(os.path.join(output_dir, "protocol_a_predictions.json.gz"), "rt", encoding="utf-8") as f:
            pred_data_a = json.load(f)["predictions"]
        y_true_a = np.array(pred_data_a["y_true"])
        pred_prop_a = np.array(pred_data_a["Adaptive ML-DST (Proposed Batch)"])
        pred_rf_a = np.array(pred_data_a["Random Forest"])
        dst_key_a = "Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)"
        if dst_key_a not in pred_data_a:
            dst_key_a = "Static DST (Base Heuristic)"
        pred_dst_a = np.array(pred_data_a[dst_key_a])

        mcnemar_vs_rf = compute_mcnemar_test(y_true_a, pred_prop_a, pred_rf_a)
        mcnemar_vs_dst = compute_mcnemar_test(y_true_a, pred_prop_a, pred_dst_a)

        # Protocol B predictions
        from src.evaluation.streaming_core import read_prediction_arrays
        pred_path_l = os.path.join(output_dir, "protocol_b_predictions.jsonl.gz")
        pred_path_legacy = os.path.join(output_dir, "protocol_b_predictions.json.gz")
        target_pred_path = pred_path_l if os.path.exists(pred_path_l) else pred_path_legacy

        b_data = read_prediction_arrays(target_pred_path)
        y_true_b = b_data["ground_truth"]
        pred_prop_b = b_data["predictions"]["8_Full_Adaptive_ML_DST"]
        pred_froz_b = b_data["predictions"]["1_Frozen_Static"]
        pred_sgd_b = b_data["predictions"]["2_Online_SGD_NoDrift"]

        mcnemar_b_vs_frozen = compute_mcnemar_test(y_true_b, pred_prop_b, pred_froz_b)
        mcnemar_b_vs_sgd = compute_mcnemar_test(y_true_b, pred_prop_b, pred_sgd_b)

        statistical_report = {
            "protocol_a_mcnemar": {
                "proposed_vs_random_forest": mcnemar_vs_rf,
                "proposed_vs_static_dst": mcnemar_vs_dst,
            },
            "protocol_b_mcnemar": {
                "proposed_vs_frozen_model": mcnemar_b_vs_frozen,
                "proposed_vs_online_sgd_no_drift": mcnemar_b_vs_sgd,
            },
        }

    # Stop run-level memory profiling
    mem_summary = mem_tracker.stop()
    t_total = time.time() - t_start_total

    # Package Consolidated Results
    full_summary = {
        "experiment_id": f"EXP_ADAPTIVE_ML_DST_V5_{mode.upper()}",
        "dataset_mode": label,
        "execution_mode": mode,
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": seed,
        "stream_fingerprint": stream_fingerprint,
        "batch_size": batch_size,
        "label_delay_batches": label_delay_batches,
        "elapsed_time_sec": round(t_total, 2),
        "memory_profiling": mem_summary,
        "peak_memory_mb": mem_summary.get("process_peak_rss_mb"),
        "resolved_config": config,
        "effective_components": eff_components,
        "protocol_a": proto_a_results,
        "protocol_b": {
            "total_streaming_flows": proto_b_results.get("total_streaming_flows", 0),
            "total_warmup_flows": proto_b_results.get("total_warmup_flows", 0),
            "aggregate": proto_b_results["aggregate_metrics"],
            "daily": proto_b_results["daily_metrics"],
            "common_calibration_evaluations": proto_b_results.get("common_calibration_evaluations", proto_b_results.get("fpr_target_comparisons")),
            "defense_quality": proto_b_results["defense_quality"],
            "drift_alarms_count": proto_b_results["drift_alarms_total"],
            "row_accounting": proto_b_results.get("row_accounting", {}),
        },
        "protocol_c": proto_c_results.get("drift_detection_performance") if proto_c_results else None,
        "ablation_study": ablation_results.get("variants") if ablation_results else None,
        "statistical_tests": statistical_report,
    }

    out_full_json = os.path.join(output_dir, "full_experiment_results.json")
    with open(out_full_json, "w", encoding="utf-8") as f:
        json.dump(
            full_summary,
            f,
            indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else o.tolist() if hasattr(o, "tolist") else str(o),
        )
    logger.info(f"Saved complete experiment results to {out_full_json}")

    # --- EXPORT CSV TABLES ---
    # Table 1: Protocol A (if run)
    if proto_a_results:
        t1_rows = []
        for m_name, m_dict in proto_a_results.items():
            t1_rows.append({
                "Model": m_name,
                "Accuracy": m_dict.get("accuracy"),
                "Balanced_Acc": m_dict.get("balanced_accuracy"),
                "Recall (ADR)": m_dict.get("recall"),
                "Precision": m_dict.get("precision"),
                "Attack_F1": m_dict.get("attack_f1", m_dict.get("f1")),
                "Macro_F1": m_dict.get("macro_f1", m_dict.get("f1_score")),
                "F2_Score": m_dict.get("f2", m_dict.get("f2_score")),
                "FPR": m_dict.get("false_positive_rate"),
                "FNR": m_dict.get("false_negative_rate"),
                "ROC_AUC": m_dict.get("roc_auc"),
                "PR_AUC": m_dict.get("pr_auc"),
                "Latency_ms": m_dict.get("latency_ms"),
                "Peak_Mem_MB": m_dict.get("peak_memory_mb"),
                "Throughput_FPS": m_dict.get("throughput_fps"),
            })
        pd.DataFrame(t1_rows).to_csv(os.path.join(tables_dir, "table1_comparative_models.csv"), index=False)

    # Table 1b: Common Calibration-Target Evaluations (Table 1b)
    calib_evals = proto_b_results.get("common_calibration_evaluations", proto_b_results.get("fpr_target_comparisons", []))
    if calib_evals:
        t1b_rows = []
        for comp in calib_evals:
            t1b_rows.append({
                "Model": comp.get("model"),
                "Target_Calibration_FPR": comp.get("target_calibration_fpr", comp.get("target_fpr")),
                "Calibrated_Threshold": comp.get("calibrated_threshold"),
                "Achieved_Calibration_FPR": comp.get("achieved_calibration_fpr", comp.get("target_fpr")),
                "Achieved_Evaluation_FPR": comp.get("achieved_evaluation_fpr", comp.get("achieved_fpr")),
                "Recall (ADR)": comp.get("recall"),
                "Precision": comp.get("precision"),
                "Attack_F1": comp.get("attack_f1"),
                "Macro_F1": comp.get("macro_f1"),
                "F2_Score": comp.get("f2"),
                "False_Alarms": comp.get("false_positives"),
                "Missed_Attacks": comp.get("false_negatives"),
                "PR_AUC": comp.get("pr_auc"),
                "ROC_AUC": comp.get("roc_auc"),
            })
        pd.DataFrame(t1b_rows).to_csv(os.path.join(tables_dir, "table1b_tradeoff_comparisons.csv"), index=False)

    # Table 2: Ablation Study (if run)
    if ablation_results:
        t2_rows = []
        for var_name, m_dict in ablation_results["variants"].items():
            t2_rows.append({
                "Variant": var_name,
                "Accuracy": m_dict.get("accuracy"),
                "Balanced_Acc": m_dict.get("balanced_accuracy"),
                "Recall (ADR)": m_dict.get("recall"),
                "Precision": m_dict.get("precision"),
                "Attack_F1": m_dict.get("attack_f1", m_dict.get("f1")),
                "Macro_F1": m_dict.get("macro_f1", m_dict.get("f1_score")),
                "F2_Score": m_dict.get("f2", m_dict.get("f2_score")),
                "FPR": m_dict.get("false_positive_rate"),
                "FNR": m_dict.get("false_negative_rate"),
                "ROC_AUC": m_dict.get("roc_auc"),
                "PR_AUC": m_dict.get("pr_auc"),
                "Brier_Score": m_dict.get("brier_score"),
            })
        pd.DataFrame(t2_rows).to_csv(os.path.join(tables_dir, "table2_ablation_study.csv"), index=False)

    # Table 3: Protocol B Daily Chronological Performance
    t3_rows = []
    for day_name, models_dict in proto_b_results["daily_metrics"].items():
        for m_name in ["1_Frozen_Static", "2_Online_SGD_NoDrift", "8_Full_Adaptive_ML_DST"]:
            if m_name in models_dict:
                m_metrics = models_dict[m_name]
                t3_rows.append({
                    "Day": day_name,
                    "Model": m_name,
                    "Recall (ADR)": m_metrics.get("recall"),
                    "FNR": m_metrics.get("false_negative_rate"),
                    "FPR": m_metrics.get("false_positive_rate"),
                    "Attack_F1": m_metrics.get("attack_f1", m_metrics.get("f1")),
                    "Macro_F1": m_metrics.get("macro_f1", m_metrics.get("f1_score")),
                    "F2_Score": m_metrics.get("f2", m_metrics.get("f2_score")),
                    "PR_AUC": m_metrics.get("pr_auc"),
                })
    if t3_rows:
        pd.DataFrame(t3_rows).to_csv(os.path.join(tables_dir, "table3_chronological_daily.csv"), index=False)

    # Table 4: Protocol C Controlled Drift Benchmark (if run)
    if proto_c_results:
        c_perf = proto_c_results["drift_detection_performance"]
        df_t4 = pd.DataFrame([{
            "Total_True_Change_Points": c_perf.get("true_drift_events"),
            "Total_Detected_Alarms": c_perf.get("detected_alarms_total"),
            "True_Alarms_Matched": c_perf.get("true_alarms_matched"),
            "Detection_Rate": c_perf.get("detection_rate"),
            "Missed_Events": c_perf.get("missed_events"),
            "False_Alarms": c_perf.get("false_alarms"),
            "Duplicate_Alarms": c_perf.get("duplicate_alarms", 0),
            "Tolerance_Window_Flows": c_perf.get("tolerance_window_flows", 250),
            "Mean_Operational_Delay_Flows": c_perf.get("mean_detection_delay_flows"),
            "Median_Operational_Delay_Flows": c_perf.get("median_detection_delay_flows"),
            "Mean_Detector_Only_Delay_Flows": c_perf.get("mean_detector_only_delay_flows"),
            "Median_Detector_Only_Delay_Flows": c_perf.get("median_detector_only_delay_flows"),
        }])
        df_t4.to_csv(os.path.join(tables_dir, "table4_drift_benchmark.csv"), index=False)

    logger.info("All experimental tables successfully generated!")

    # Auto-generate implementation report if in lightweight or smoke mode
    try:
        from scripts.generate_report import generate_implementation_report
        report_out = os.path.join(output_dir, "IMPLEMENTATION_RESULTS.md")
        generate_implementation_report(
            results_json_path=out_full_json,
            tables_dir=tables_dir,
            output_report_path=report_out,
        )
        if not smoke_test and mode == "lightweight":
            # Also copy to root IMPLEMENTATION_RESULTS.md
            generate_implementation_report(
                results_json_path=out_full_json,
                tables_dir=tables_dir,
                output_report_path="IMPLEMENTATION_RESULTS.md",
            )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

    return full_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Adaptive ML-DST Experiments")
    parser.add_argument("--config", type=str, default="configs/lightweight.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--smoke-test", action="store_true", help="Execute full streaming pipeline smoke test")
    parser.add_argument("--max-rows", type=int, default=None, help="Capped maximum flows for test run")
    args = parser.parse_args()

    run_all_experiments(
        config_path=args.config,
        random_seed=args.seed,
        output_dir_override=args.output_dir,
        smoke_test=args.smoke_test,
        max_rows=args.max_rows,
    )
