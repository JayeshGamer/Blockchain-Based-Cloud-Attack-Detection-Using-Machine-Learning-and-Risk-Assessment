"""
Independent Prediction Integrity and Metric Recomputation Verifier.
Reads compressed prediction logs incrementally, recomputes confusion matrices
and score metrics (ROC-AUC, Average Precision), and verifies exact alignment
with protocol_b_results.json.
"""
import gzip
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def verify_run_integrity(run_dir: str):
    print(f"\n==================================================================")
    print(f"VERIFYING INTEGRITY FOR: {run_dir}")
    print(f"==================================================================")

    results_file = os.path.join(run_dir, "protocol_b_results.json")
    pred_file = os.path.join(run_dir, "protocol_b_predictions.jsonl.gz")

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Missing {results_file}")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Missing {pred_file}")

    with open(results_file, "r") as f:
        results = json.load(f)

    agg = results.get("aggregate_metrics") or results.get("aggregate", {})
    models = list(agg.keys())

    # Read predictions incrementally
    record_count = 0
    y_true_list = []
    y_pred_dict = {m: [] for m in models}
    score_dict = {m: [] for m in models}
    source_days = set()

    with gzip.open(pred_file, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            record_count += 1
            y_true = int(rec["y_true"])
            y_true_list.append(y_true)
            source_days.add(rec["source_day"])

            # Verify schema
            assert "flow_id" in rec
            assert "source_day" in rec
            assert "batch_idx" in rec
            assert "threshold_used" in rec

            for m in models:
                y_pred_dict[m].append(int(rec["predictions"][m]))
                score_dict[m].append(float(rec["scores"][m]))

    print(f"Records read: {record_count:,}")
    print(f"Source days observed in evaluated stream: {sorted(list(source_days))}")
    assert "Monday" not in source_days, "CRITICAL ERROR: Monday warm-up row found in evaluated predictions!"
    print("Zero Monday warm-up rows confirmed in evaluated stream.")

    y_true_arr = np.array(y_true_list)
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    print(f"Evaluated flow breakdown: Benign={n_neg:,}, Attack={n_pos:,}")

    # Recompute confusion matrices and metrics
    all_matched = True
    for m in models:
        y_p = np.array(y_pred_dict[m])
        s_m = np.array(score_dict[m])

        # CM: tn, fp, fn, tp
        tn = int(np.sum((y_true_arr == 0) & (y_p == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_p == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_p == 0)))
        tp = int(np.sum((y_true_arr == 1) & (y_p == 1)))

        expected_cm = agg[m]["confusion_matrix"]
        cm_match = (
            expected_cm["tn"] == tn and
            expected_cm["fp"] == fp and
            expected_cm["fn"] == fn and
            expected_cm["tp"] == tp
        )

        # ROC-AUC / PR-AUC
        if n_pos > 0 and n_neg > 0:
            rec_roc = roc_auc_score(y_true_arr, s_m)
            rec_pr = average_precision_score(y_true_arr, s_m)
            exp_roc = agg[m].get("roc_auc")
            exp_pr = agg[m].get("pr_auc")

            roc_match = (exp_roc is not None and abs(exp_roc - rec_roc) < 1e-4)
            pr_match = (exp_pr is not None and abs(exp_pr - rec_pr) < 1e-4)
        else:
            rec_roc = None
            rec_pr = None
            roc_match = agg[m].get("roc_auc") is None
            pr_match = agg[m].get("pr_auc") is None

        status = "OK" if (cm_match and roc_match and pr_match) else "MISMATCH"
        if status != "OK":
            all_matched = False
        print(f"Model: {m:<26} | CM match: {cm_match} | ROC-AUC: {rec_roc} (exp {agg[m].get('roc_auc')}) | PR-AUC: {rec_pr} (exp {agg[m].get('pr_auc')}) [{status}]")

    assert all_matched, f"Integrity check failed for {run_dir}!"
    print(f"SUCCESS: All metrics for {run_dir} independently recomputed and exactly verified!")

if __name__ == "__main__":
    verify_run_integrity("reports/experiments/lightweight")
    verify_run_integrity("reports/experiments/smoke_test")
    print("\nALL RUN INTEGRITY AUDITS PASSED PERFECTLY!")
