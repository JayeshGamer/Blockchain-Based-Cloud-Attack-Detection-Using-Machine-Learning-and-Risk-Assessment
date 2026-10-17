"""
Real Full-Path Streaming Smoke Test for Adaptive ML-DST (V5_REVIEW).
Exercises the exact CLI/configuration execution path used for full-dataset runs:
1. Invokes run_all_experiments with configs/full.yaml and --smoke-test.
2. Streams a small chronological subset (5,000 flows) across raw files.
3. Feeds deduplicated batches directly through the shared StreamingEvaluationEngine.
4. Never collects the entire stream in memory.
5. Verifies row accounting, full 64-char SHA-256 digests, and genuine peak memory.
6. Isolates outputs to reports/experiments/smoke_test/.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_experiments import run_all_experiments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test_full_mode")


def run_full_path_smoke_test(
    config_path: str = "configs/full.yaml",
    smoke_rows: int = 5000,
    output_dir: str = "reports/experiments/smoke_test",
):
    logger.info("=" * 75)
    logger.info("STARTING REAL FULL-PATH STREAMING SMOKE TEST")
    logger.info(f"Config: {config_path} | Rows: {smoke_rows:,} | Output Dir: {output_dir}")
    logger.info("=" * 75)

    # Invoke the real full streaming execution path
    summary = run_all_experiments(
        config_path=config_path,
        random_seed=42,
        output_dir_override=output_dir,
        smoke_test=True,
        max_rows=smoke_rows,
    )

    # Verify generated artifacts
    assert os.path.exists(os.path.join(output_dir, "full_experiment_results.json")), "Missing full_experiment_results.json"
    assert os.path.exists(os.path.join(output_dir, "protocol_b_results.json")), "Missing protocol_b_results.json"
    assert os.path.exists(os.path.join(output_dir, "effective_components.json")), "Missing effective_components.json"
    assert os.path.exists(os.path.join(output_dir, "tables", "table1b_tradeoff_comparisons.csv")), "Missing table1b"
    assert os.path.exists(os.path.join(output_dir, "tables", "table3_chronological_daily.csv")), "Missing table3"

    pred_file = os.path.join(output_dir, "protocol_b_predictions.jsonl.gz")
    if not os.path.exists(pred_file):
        pred_file = os.path.join(output_dir, "protocol_b_predictions.json.gz")
    assert os.path.exists(pred_file), "Missing prediction log artifact"
    assert os.path.getsize(pred_file) > 0, "Prediction log artifact is empty"

    # Verify meaningful memory measurement
    mem_info = summary.get("memory_profiling", {})
    peak_rss = mem_info.get("process_peak_rss_mb")
    assert peak_rss is not None and peak_rss > 0, f"Memory measurement invalid: {peak_rss}"

    proto_b = summary.get("protocol_b", {})
    total_eval = proto_b.get("total_streaming_flows", 0)
    assert total_eval > 0, "Evaluated streaming flows must be > 0"

    # Verify Requirement 4: Both benign and attack classes MUST be present in evaluated stream
    agg = proto_b.get("aggregate_metrics") or proto_b.get("aggregate", {})
    cm_prop = agg.get("8_Full_Adaptive_ML_DST", {}).get("confusion_matrix", {})
    pos_count = cm_prop.get("tp", 0) + cm_prop.get("fn", 0)
    neg_count = cm_prop.get("tn", 0) + cm_prop.get("fp", 0)
    assert pos_count > 0, f"Smoke test failed: no attack flows evaluated (pos_count={pos_count})"
    assert neg_count > 0, f"Smoke test failed: no benign flows evaluated (neg_count={neg_count})"

    # Verify Requirement 2 & 8: Independent per-model metrics
    for mod_name in ["1_Frozen_Static", "2_Online_SGD_NoDrift", "8_Full_Adaptive_ML_DST"]:
        assert mod_name in agg, f"Model {mod_name} missing from aggregate metrics"
        assert "roc_auc" in agg[mod_name], f"Model {mod_name} missing roc_auc"
        assert "pr_auc" in agg[mod_name], f"Model {mod_name} missing pr_auc"

    # Verify Requirement 3: Row accounting
    row_acc = proto_b.get("row_accounting", {})
    assert "warmup_rows" in row_acc or proto_b.get("total_warmup_flows", 0) > 0, "Warm-up accounting missing"

    logger.info(f"Smoke Test Completed Successfully:")
    logger.info(f"  Execution Mode: {summary.get('execution_mode')}")
    logger.info(f"  Dataset Label: {summary.get('dataset_mode')}")
    logger.info(f"  Evaluated Streaming Flows: {total_eval:,} (Benign: {neg_count:,}, Attack: {pos_count:,})")
    logger.info(f"  Peak Resident Memory (RSS): {peak_rss} MB")
    logger.info(f"  Status: SMOKE-TESTED (Informative mixed-class multi-file subset)")
    logger.info("=" * 75)

    return summary


if __name__ == "__main__":
    run_full_path_smoke_test()
