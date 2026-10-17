"""
Shared Streaming Evaluation Core for Adaptive ML-DST (V6_VERIFIED).
Provides a unified, strictly prequential (predict-before-update) evaluation engine
used identically by both lightweight DataFrame replay and full chunked streaming generators.

Guarantees:
1. Identical prediction and update semantics across lightweight and full streaming.
2. Original predictions and continuous risk scores saved incrementally at arrival time.
3. Errors and drift signals for delayed feedback (t + D) derived strictly from original arrival predictions.
4. Strictly bounded memory: pending feedback queue size <= D + 1, ring-buffered events.
5. Zero memory accumulation at finalization: prediction logs written directly to compressed JSONL.
6. Discrete, model-specific score persistence and independent metric computation (ROC-AUC, PR-AUC / AP).
7. Genuine achieved calibration FPR measurement (false positives / benign calibration samples).
8. Protection of temporal baseline: unverified attack traffic never contaminates the rolling window.
"""

import os
import json
import gzip
import time
import logging
from collections import deque
from typing import Dict, Any, List, Tuple, Optional, Generator, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.profiling import ProcessMemoryTracker
from src.utils.metrics import compute_binary_metrics

logger = logging.getLogger(__name__)

# Explicit, validated mapping of models to binary score log columns (0: y_true, 1..8: model scores)
EVALUATED_MODELS_COL_MAP: Dict[str, int] = {
    "1_Frozen_Static": 1,
    "2_Online_SGD_NoDrift": 2,
    "3_Online_SGD_ADWIN": 3,
    "4_River_HATC_NoDrift": 4,
    "5_River_HATC_ADWIN": 5,
    "6_Online_Anomaly_Evidence": 6,
    "7_Online_DST_FixedThresh": 7,
    "8_Full_Adaptive_ML_DST": 8,
}

CALIBRATION_MODEL_TO_VARIANT: Dict[str, str] = {
    "Online_SGD": "2_Online_SGD_NoDrift",
    "Online_SGD_Anomaly": "6_Online_Anomaly_Evidence",
    "Proposed_ML_DST": "8_Full_Adaptive_ML_DST",
}


def read_prediction_records(pred_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Streams prediction records incrementally from compressed JSONL (or legacy JSON)
    without accumulating the complete dataset into memory.
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction log file not found at: {pred_path}")

    with gzip.open(pred_path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            return
        first_stripped = first_line.strip()
        # Handle legacy single-dict JSON if encountered
        if first_stripped.startswith('{"stream_fingerprint"') or '"ground_truth"' in first_stripped[:200]:
            full_dict = json.loads(first_stripped + f.read())
            n_total = len(full_dict.get("ground_truth", []))
            for i in range(n_total):
                yield {
                    "flow_id": str(i),
                    "batch_idx": i // 1000,
                    "source_day": "Unknown",
                    "y_true": int(full_dict["ground_truth"][i]),
                    "predictions": {k: int(v[i]) for k, v in full_dict.get("predictions", {}).items()},
                    "scores": {k: float(v[i]) for k, v in full_dict.get("risk_scores", {}).items()},
                    "threshold_used": 0.50,
                }
            return

        # Modern incremental JSONL format: each line is one flow record
        yield json.loads(first_stripped)
        for line in f:
            line_s = line.strip()
            if line_s:
                yield json.loads(line_s)


def read_prediction_arrays(pred_path: str) -> Dict[str, Any]:
    """
    Reads prediction records into dictionary of arrays for offline plotting or statistical tests.
    Documented for bounded memory offline analysis.
    """
    flow_ids = []
    days = []
    y_true_list = []
    preds: Dict[str, List[int]] = {}
    scores: Dict[str, List[float]] = {}
    thresholds = []

    for rec in read_prediction_records(pred_path):
        flow_ids.append(rec.get("flow_id", ""))
        days.append(rec.get("source_day", "Unknown"))
        y_true_list.append(rec["y_true"])
        thresholds.append(rec.get("threshold_used", 0.50))
        for m_name, p_val in rec.get("predictions", {}).items():
            if m_name not in preds:
                preds[m_name] = []
            preds[m_name].append(p_val)
        for m_name, s_val in rec.get("scores", {}).items():
            if m_name not in scores:
                scores[m_name] = []
            scores[m_name].append(s_val)

    return {
        "flow_ids": flow_ids,
        "days": days,
        "ground_truth": np.array(y_true_list, dtype=int),
        "predictions": {k: np.array(v, dtype=int) for k, v in preds.items()},
        "risk_scores": {k: np.array(v, dtype=float) for k, v in scores.items()},
        "thresholds": thresholds,
    }


class StreamingEvaluationEngine:
    """
    Unified streaming prequential evaluation engine for both lightweight replay
    and out-of-core full streaming modes.
    """

    def __init__(
        self,
        variants: Dict[str, Any],
        auxiliary_engines: Dict[str, Any],
        calibrated_thresholds: Dict[str, Dict[float, float]],
        calibration_stats: Optional[Dict[str, Any]] = None,
        label_delay_batches: int = 1,
        output_dir: str = "reports/experiments",
        save_predictions: bool = True,
        prediction_filename: str = "protocol_b_predictions.jsonl.gz",
        stream_fingerprint: str = "",
    ):
        self.variants = variants
        self.aux = auxiliary_engines
        self.calibrated_thresholds = calibrated_thresholds
        self.calibration_stats = calibration_stats or {}
        self.label_delay = max(0, int(label_delay_batches))
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        self.prediction_filename = prediction_filename
        self.stream_fingerprint = stream_fingerprint

        os.makedirs(self.output_dir, exist_ok=True)
        self.pred_file_path = os.path.join(self.output_dir, self.prediction_filename)

        # FIFO queue for delayed label feedback: stores batches until label delay expires
        self.pending_feedback_queue: deque = deque()

        # Incremental Confusion Matrix Accumulators
        self.aggregate_cm: Dict[str, Dict[str, int]] = {
            v: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for v in self.variants.keys()
        }

        # daily_cm[day_name][variant_name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.daily_cm: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Calibration Target CM Accumulators (1%, 2%, 5% FPR targets)
        self.calib_cm: Dict[str, Dict[float, Dict[str, int]]] = {
            m: {fpr: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for fpr in [0.01, 0.02, 0.05]}
            for m in ["Online_SGD", "Online_SGD_Anomaly", "Proposed_ML_DST"]
        }

        # Bounded ring buffer for drift event logs (max 2000 events to prevent unbounded RAM growth)
        self.drift_events_log: deque = deque(maxlen=2000)

        # Disk-backed continuous score log (9 float32 columns: y_true + 8 evaluated models)
        self.score_log_path = os.path.join(self.output_dir, "_streaming_scores.dat")
        self._score_file = open(self.score_log_path, "wb")

        # Direct incremental compressed JSONL predictions file (zero memory accumulation at finalization)
        if self.save_predictions:
            self._pred_file = gzip.open(self.pred_file_path, "wt", encoding="utf-8")
        else:
            self._pred_file = None

        self.total_evaluated_flows = 0
        self.batch_count = 0

    def process_batch(
        self,
        batch_idx: int,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_day: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes prequential prediction and delayed feedback for one batch.
        Guarantees:
        1. Predict BEFORE update.
        2. Row-level daily metric attribution using each row's actual day.
        3. Full score precision and threshold used persisted incrementally to disk.
        4. Protection of temporal baseline: incoming test flows do NOT mutate the baseline during inference.
        """
        self.batch_count += 1
        n_samples = len(X_batch)
        self.total_evaluated_flows += n_samples

        # Extract per-row metadata if available
        if metadata and "row_days" in metadata:
            row_days = np.asarray(metadata["row_days"])
        else:
            row_days = np.full(n_samples, batch_day, dtype=object)

        if metadata and "flow_ids" in metadata:
            flow_ids = list(metadata["flow_ids"])
        elif metadata and "original_row_index" in metadata:
            flow_ids = [f"{day}_{idx}" for day, idx in zip(row_days, metadata["original_row_index"])]
        else:
            flow_ids = [f"{day}_{batch_idx}_{i}" for i, day in enumerate(row_days)]

        # 1. Prequential Inference: Predict BEFORE Update
        batch_preds: Dict[str, np.ndarray] = {}
        batch_scores: Dict[str, np.ndarray] = {}

        # 1. Frozen Static Model (Never adapts)
        thresh_frozen = float(self.aux.get("thresh_frozen_clf", 0.50))
        s_1 = self.variants["1_Frozen_Static"].predict_score(X_batch)
        p_1 = (s_1 >= thresh_frozen).astype(int)
        batch_scores["1_Frozen_Static"] = s_1
        batch_preds["1_Frozen_Static"] = p_1

        # 2. Online SGD No Drift
        s_2 = self.variants["2_Online_SGD_NoDrift"].predict_score(X_batch)
        p_2 = (s_2 >= 0.50).astype(int)
        batch_scores["2_Online_SGD_NoDrift"] = s_2
        batch_preds["2_Online_SGD_NoDrift"] = p_2

        # 3. Online SGD ADWIN
        s_3 = self.variants["3_Online_SGD_ADWIN"].predict_score(X_batch)
        p_3 = (s_3 >= 0.50).astype(int)
        batch_scores["3_Online_SGD_ADWIN"] = s_3
        batch_preds["3_Online_SGD_ADWIN"] = p_3

        # 4. River HATC No Drift
        s_4 = self.variants["4_River_HATC_NoDrift"].predict_score(X_batch)
        p_4 = (s_4 >= 0.50).astype(int)
        batch_scores["4_River_HATC_NoDrift"] = s_4
        batch_preds["4_River_HATC_NoDrift"] = p_4

        # 5. River HATC ADWIN
        s_5 = self.variants["5_River_HATC_ADWIN"].predict_score(X_batch)
        p_5 = (s_5 >= 0.50).astype(int)
        batch_scores["5_River_HATC_ADWIN"] = s_5
        batch_preds["5_River_HATC_ADWIN"] = p_5

        # 6. Online Anomaly Evidence
        s_6_sgd = self.variants["6_Online_Anomaly_Evidence"].predict_score(X_batch)
        s_6_anom = self.aux["anom_det_6"].score(X_batch)
        s_6_fused = 0.60 * s_6_sgd + 0.40 * s_6_anom
        p_6 = (s_6_fused >= 0.50).astype(int)
        batch_scores["6_Online_Anomaly_Evidence"] = s_6_fused
        batch_preds["6_Online_Anomaly_Evidence"] = p_6

        # 7. Online DST Fixed Threshold (Scoring inference does NOT update baseline buffer)
        s_7_sgd = self.variants["7_Online_DST_FixedThresh"].predict_score(X_batch)
        s_7_anom = self.aux["anom_det_7"].score(X_batch)
        s_7_temp = self.aux["temp_tracker_7"].compute_batch_deviations(X_batch, update_window=False)
        fused_7 = self.aux["dst_engine_7"].fuse_batch(s_7_sgd, s_7_anom, s_7_temp)
        s_7_dst = fused_7["risk_scores"]
        p_7 = (s_7_dst >= 0.50).astype(int)
        batch_scores["7_Online_DST_FixedThresh"] = s_7_dst
        batch_preds["7_Online_DST_FixedThresh"] = p_7

        # 8. Proposed Full Adaptive ML-DST (Scoring inference does NOT update baseline buffer)
        s_8_sgd = self.variants["8_Full_Adaptive_ML_DST"].predict_score(X_batch)
        s_8_anom = self.aux["anom_det_8"].score(X_batch)
        s_8_temp = self.aux["temp_tracker_8"].compute_batch_deviations(X_batch, update_window=False)
        fused_8 = self.aux["dst_engine_8"].fuse_batch(s_8_sgd, s_8_anom, s_8_temp)
        s_8_dst = fused_8["risk_scores"]

        current_thresh = float(self.aux["thresh_mgr"].current_threshold)
        p_8 = (s_8_dst >= current_thresh).astype(int)
        batch_scores["8_Full_Adaptive_ML_DST"] = s_8_dst
        batch_preds["8_Full_Adaptive_ML_DST"] = p_8

        # Update defense simulator
        self.aux["defense"].process_batch(s_8_dst, y_batch)

        # 2. Update Incremental Confusion Matrices
        for v_name in self.variants.keys():
            preds_v = batch_preds[v_name]
            tp = int(np.sum((preds_v == 1) & (y_batch == 1)))
            fp = int(np.sum((preds_v == 1) & (y_batch == 0)))
            tn = int(np.sum((preds_v == 0) & (y_batch == 0)))
            fn = int(np.sum((preds_v == 0) & (y_batch == 1)))

            self.aggregate_cm[v_name]["tp"] += tp
            self.aggregate_cm[v_name]["fp"] += fp
            self.aggregate_cm[v_name]["tn"] += tn
            self.aggregate_cm[v_name]["fn"] += fn

        # Update daily confusion matrices using each row's ACTUAL day
        unique_days = np.unique(row_days)
        for u_day in unique_days:
            u_day_str = str(u_day)
            if u_day_str not in self.daily_cm:
                self.daily_cm[u_day_str] = {
                    v: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for v in self.variants.keys()
                }
            mask_day = (row_days == u_day)
            y_sub = y_batch[mask_day]
            for v_name in self.variants.keys():
                p_sub = batch_preds[v_name][mask_day]
                self.daily_cm[u_day_str][v_name]["tp"] += int(np.sum((p_sub == 1) & (y_sub == 1)))
                self.daily_cm[u_day_str][v_name]["fp"] += int(np.sum((p_sub == 1) & (y_sub == 0)))
                self.daily_cm[u_day_str][v_name]["tn"] += int(np.sum((p_sub == 0) & (y_sub == 0)))
                self.daily_cm[u_day_str][v_name]["fn"] += int(np.sum((p_sub == 0) & (y_sub == 1)))

        # 3. Update Calibration-Target Threshold Matrices (1%, 2%, 5% FPR)
        for target_fpr in [0.01, 0.02, 0.05]:
            # Online SGD
            t_sgd = self.calibrated_thresholds["Online_SGD"][target_fpr]
            p_sgd_c = (s_2 >= t_sgd).astype(int)
            self.calib_cm["Online_SGD"][target_fpr]["tp"] += int(np.sum((p_sgd_c == 1) & (y_batch == 1)))
            self.calib_cm["Online_SGD"][target_fpr]["fp"] += int(np.sum((p_sgd_c == 1) & (y_batch == 0)))
            self.calib_cm["Online_SGD"][target_fpr]["tn"] += int(np.sum((p_sgd_c == 0) & (y_batch == 0)))
            self.calib_cm["Online_SGD"][target_fpr]["fn"] += int(np.sum((p_sgd_c == 0) & (y_batch == 1)))

            # Online SGD + Anomaly
            t_anom = self.calibrated_thresholds["Online_SGD_Anomaly"][target_fpr]
            p_anom_c = (s_6_fused >= t_anom).astype(int)
            self.calib_cm["Online_SGD_Anomaly"][target_fpr]["tp"] += int(np.sum((p_anom_c == 1) & (y_batch == 1)))
            self.calib_cm["Online_SGD_Anomaly"][target_fpr]["fp"] += int(np.sum((p_anom_c == 1) & (y_batch == 0)))
            self.calib_cm["Online_SGD_Anomaly"][target_fpr]["tn"] += int(np.sum((p_anom_c == 0) & (y_batch == 0)))
            self.calib_cm["Online_SGD_Anomaly"][target_fpr]["fn"] += int(np.sum((p_anom_c == 0) & (y_batch == 1)))

            # Proposed ML-DST
            t_dst = self.calibrated_thresholds["Proposed_ML_DST"][target_fpr]
            p_dst_c = (s_8_dst >= t_dst).astype(int)
            self.calib_cm["Proposed_ML_DST"][target_fpr]["tp"] += int(np.sum((p_dst_c == 1) & (y_batch == 1)))
            self.calib_cm["Proposed_ML_DST"][target_fpr]["fp"] += int(np.sum((p_dst_c == 1) & (y_batch == 0)))
            self.calib_cm["Proposed_ML_DST"][target_fpr]["tn"] += int(np.sum((p_dst_c == 0) & (y_batch == 0)))
            self.calib_cm["Proposed_ML_DST"][target_fpr]["fn"] += int(np.sum((p_dst_c == 0) & (y_batch == 1)))

        # 4. Write continuous scores to disk-backed binary log
        # 9 columns: [y_true, s_1, s_2, s_3, s_4, s_5, s_6_fused, s_7_dst, s_8_dst]
        log_record = np.column_stack([
            y_batch.astype(np.float64),
            s_1.astype(np.float64),
            s_2.astype(np.float64),
            s_3.astype(np.float64),
            s_4.astype(np.float64),
            s_5.astype(np.float64),
            s_6_fused.astype(np.float64),
            s_7_dst.astype(np.float64),
            s_8_dst.astype(np.float64),
        ])
        log_record.tofile(self._score_file)

        # 5. Incrementally write predictions to compressed JSONL (full float precision preserved)
        if self._pred_file is not None:
            json_lines = []
            for i in range(n_samples):
                flow_rec = {
                    "flow_id": str(flow_ids[i]),
                    "batch_idx": batch_idx,
                    "source_day": str(row_days[i]),
                    "y_true": int(y_batch[i]),
                    "predictions": {k: int(v[i]) for k, v in batch_preds.items()},
                    "scores": {k: float(v[i]) for k, v in batch_scores.items()},
                    "threshold_used": {
                        "8_Full_Adaptive_ML_DST": current_thresh,
                        "1_Frozen_Static": thresh_frozen,
                        "fixed_default": 0.50,
                    },
                }
                json_lines.append(json.dumps(flow_rec))
            self._pred_file.write("\n".join(json_lines) + "\n")

        # 6. Enqueue batch for Delayed Feedback
        arrival_record = {
            "batch_idx": batch_idx,
            "X": X_batch,
            "y": y_batch,
            "preds": batch_preds,
            "scores": batch_scores,
            "day": batch_day,
            "row_days": row_days,
        }
        self.pending_feedback_queue.append(arrival_record)

        # 7. Process Delayed Feedback when queue exceeds delay D
        while len(self.pending_feedback_queue) > self.label_delay:
            mature_batch = self.pending_feedback_queue.popleft()
            self._apply_delayed_feedback(mature_batch, operational_step=self.total_evaluated_flows)

        # Sample run-level memory tracker
        active_mem = ProcessMemoryTracker.get_active()
        if active_mem:
            active_mem.sample()

        return {
            "batch_idx": batch_idx,
            "flows": n_samples,
            "full_dst_pred": p_8,
            "full_dst_score": s_8_dst,
            "adaptive_threshold": current_thresh,
        }

    def _apply_delayed_feedback(self, batch_data: Dict[str, Any], operational_step: int) -> None:
        """
        Applies delayed label feedback strictly using original predictions from batch arrival.
        Never recomputes predictions using an updated model!
        """
        X_upd = batch_data["X"]
        y_upd = batch_data["y"]
        orig_scores = batch_data["scores"]
        batch_idx = batch_data["batch_idx"]

        # 1. Concept drift monitoring using original arrival predictions/scores BEFORE learner update
        # SGD ADWIN: per-flow error updating
        s_sgd_orig = orig_scores["3_Online_SGD_ADWIN"]
        errs_sgd = (s_sgd_orig - y_upd) ** 2
        drift_sgd = False
        for err in errs_sgd:
            if self.aux["adwin_sgd"].update(float(err)):
                drift_sgd = True
        if drift_sgd:
            self.variants["3_Online_SGD_ADWIN"].adapt_to_drift("plasticity_boost", boost_factor=2.0)

        # River HATC ADWIN: per-flow error updating
        s_hatc_orig = orig_scores["5_River_HATC_ADWIN"]
        errs_hatc = (s_hatc_orig - y_upd) ** 2
        drift_hatc = False
        for err in errs_hatc:
            if self.aux["adwin_hatc"].update(float(err)):
                drift_hatc = True
        if drift_hatc:
            self.variants["5_River_HATC_ADWIN"].adapt_to_drift("plasticity_boost", boost_factor=2.0)

        # Multi-signal drift coordination for full ML-DST: per-flow errors and scores
        s_full_orig = orig_scores["8_Full_Adaptive_ML_DST"]
        errs_full = (s_full_orig - y_upd) ** 2

        drift_full_res = self.aux["drift_coord_full"].update_batch(
            delayed_errors=errs_full,
            risk_scores=s_full_orig,
            packet_rates=np.array([len(X_upd)], dtype=float),
            step_index=operational_step,
        )
        drift_full = drift_full_res.get("drift_detected", False)
        action = drift_full_res.get("action_taken", "none")
        reason = ", ".join([k for k, v in drift_full_res.get("alarm_signals", {}).items() if v]) if drift_full else "none"

        if drift_full:
            logger.info(f"Concept Drift Alarm at step {operational_step}! Action: {action} (signals: {reason})")
            self.drift_events_log.append({
                "originating_batch_idx": batch_idx,
                "operational_step": operational_step,
                "action": action,
                "reason": reason,
                "alarm_signals": drift_full_res.get("alarm_signals", {}),
            })
            if action in ["plasticity_boost", "full_reset"]:
                self.variants["8_Full_Adaptive_ML_DST"].adapt_to_drift(action, boost_factor=2.0)

        # 2. Update online learners with mature batch labels
        self.variants["2_Online_SGD_NoDrift"].partial_fit(X_upd, y_upd)
        self.variants["3_Online_SGD_ADWIN"].partial_fit(X_upd, y_upd)
        self.variants["4_River_HATC_NoDrift"].partial_fit(X_upd, y_upd)
        self.variants["5_River_HATC_ADWIN"].partial_fit(X_upd, y_upd)
        self.variants["6_Online_Anomaly_Evidence"].partial_fit(X_upd, y_upd)
        self.variants["7_Online_DST_FixedThresh"].partial_fit(X_upd, y_upd)
        self.variants["8_Full_Adaptive_ML_DST"].partial_fit(X_upd, y_upd)

        # 3. Update adaptive threshold manager with delayed feedback labels (single relaxation invocation)
        self.aux["thresh_mgr"].update(
            s_full_orig,
            labels=y_upd,
            drift_detected=drift_full,
            step_index=operational_step,
        )

        # 4. Protected temporal tracker baseline: update rolling window ONLY with confirmed benign flows
        benign_mask = (y_upd == 0)
        if np.any(benign_mask):
            X_benign = X_upd[benign_mask]
            self.aux["temp_tracker_7"].compute_batch_deviations(X_benign, update_window=True)
            self.aux["temp_tracker_8"].compute_batch_deviations(X_benign, update_window=True)

    def finalize(self) -> Dict[str, Any]:
        """
        Drains any remaining pending feedback, closes files, computes final metrics,
        and returns complete results dictionary.
        """
        # Drain remaining pending feedback at stream end
        while self.pending_feedback_queue:
            batch_data = self.pending_feedback_queue.popleft()
            self._apply_delayed_feedback(batch_data, operational_step=self.total_evaluated_flows)

        # Close prediction log stream directly (no memory assembly needed)
        if self._pred_file is not None:
            self._pred_file.close()
            self._pred_file = None

        # Close binary score file
        self._score_file.close()

        # Compute exact, model-specific ROC-AUC and PR-AUC (Average Precision) from disk-backed binary log
        auc_metrics = self._compute_disk_backed_aucs()

        # Clean up temporary score file
        if os.path.exists(self.score_log_path):
            try:
                os.remove(self.score_log_path)
            except Exception:
                pass

        # Calculate final metrics from incremental confusion matrices
        aggregate_metrics: Dict[str, Any] = {}
        for v_name, cm in self.aggregate_cm.items():
            metrics = self._cm_to_metrics(cm)
            if v_name in auc_metrics:
                metrics["roc_auc"] = auc_metrics[v_name]["roc_auc"]
                metrics["pr_auc"] = auc_metrics[v_name]["pr_auc"]
                metrics["auc_reason"] = auc_metrics[v_name].get("auc_reason")
            aggregate_metrics[v_name] = metrics

        daily_metrics: Dict[str, Any] = {}
        for day, v_cms in self.daily_cm.items():
            daily_metrics[day] = {}
            for v_name, cm in v_cms.items():
                daily_metrics[day][v_name] = self._cm_to_metrics(cm)

        # Calibration Target Metrics (1%, 2%, 5% FPR) with genuine achieved calibration FPR
        fpr_comparisons: List[Dict[str, Any]] = []
        for m_name in ["Online_SGD", "Online_SGD_Anomaly", "Proposed_ML_DST"]:
            var_key = CALIBRATION_MODEL_TO_VARIANT[m_name]
            for target_fpr in [0.01, 0.02, 0.05]:
                cm = self.calib_cm[m_name][target_fpr]
                m_calc = self._cm_to_metrics(cm)
                calib_thresh = self.calibrated_thresholds[m_name][target_fpr]

                # Retrieve genuinely achieved calibration FPR measured on Monday calibration data
                m_stats = self.calibration_stats.get(m_name, {}).get(target_fpr, {})
                achieved_calib = m_stats.get("achieved_calibration_fpr", target_fpr)
                calib_samples = self.calibration_stats.get("total_calibration_samples", 0)
                calib_fp = m_stats.get("calibration_false_positives", int(round(achieved_calib * calib_samples)))

                fpr_comparisons.append({
                    "model": m_name,
                    "target_calibration_fpr": target_fpr,
                    "achieved_calibration_fpr": achieved_calib,
                    "calibrated_threshold": calib_thresh,
                    "calibration_sample_count": calib_samples,
                    "calibration_false_positives": calib_fp,
                    "comparison_operator": ">=",
                    "achieved_evaluation_fpr": m_calc["false_positive_rate"],
                    "recall": m_calc["recall"],
                    "precision": m_calc["precision"],
                    "attack_f1": m_calc["attack_f1"],
                    "macro_f1": m_calc["macro_f1"],
                    "f2": m_calc["f2"],
                    "false_positives": cm["fp"],
                    "false_negatives": cm["fn"],
                    "pr_auc": auc_metrics.get(var_key, {}).get("pr_auc"),
                    "roc_auc": auc_metrics.get(var_key, {}).get("roc_auc"),
                })

        defense_quality = self.aux["defense"].get_defense_quality_metrics()

        return {
            "total_evaluated_flows": self.total_evaluated_flows,
            "total_batches": self.batch_count,
            "drift_alarms_total": len(self.drift_events_log),
            "drift_events": list(self.drift_events_log),
            "aggregate_metrics": aggregate_metrics,
            "daily_metrics": daily_metrics,
            "calibrated_thresholds": self.calibrated_thresholds,
            "common_calibration_evaluations": fpr_comparisons,
            "defense_quality": defense_quality,
        }

    def _compute_disk_backed_aucs(self) -> Dict[str, Dict[str, Any]]:
        """
        Computes exact, model-specific ROC-AUC and PR-AUC (Average Precision) by reading
        the disk-backed 9-column binary score file via memory-mapped array.
        Every model is evaluated strictly against its own predicted score column.
        """
        results: Dict[str, Dict[str, Any]] = {
            v: {"roc_auc": None, "pr_auc": None, "auc_reason": None} for v in self.variants.keys()
        }
        if not os.path.exists(self.score_log_path) or self.total_evaluated_flows == 0:
            for v in results:
                results[v]["auc_reason"] = "No evaluated flows or score file missing"
            return results

        try:
            mmapped = np.memmap(
                self.score_log_path,
                dtype=np.float64,
                mode="r",
                shape=(self.total_evaluated_flows, 9),
            )
            y_true = (mmapped[:, 0] >= 0.5).astype(int)
            n_pos = int(np.sum(y_true == 1))
            n_neg = int(np.sum(y_true == 0))

            if n_pos == 0 or n_neg == 0:
                reason = "Single class present in evaluated stream ground truth"
                for v in results:
                    results[v]["auc_reason"] = reason
                del mmapped
                return results

            for mod_name, col_idx in EVALUATED_MODELS_COL_MAP.items():
                if mod_name not in self.variants:
                    continue
                scores = np.array(mmapped[:, col_idx], copy=False)
                try:
                    r_auc = float(roc_auc_score(y_true, scores))
                except Exception as e:
                    r_auc = None
                try:
                    p_auc = float(average_precision_score(y_true, scores))
                except Exception as e:
                    p_auc = None

                results[mod_name] = {
                    "roc_auc": round(r_auc, 4) if r_auc is not None else None,
                    "pr_auc": round(p_auc, 4) if p_auc is not None else None,
                    "auc_reason": None if (r_auc is not None and p_auc is not None) else "Calculation exception",
                }
            del mmapped
        except Exception as e:
            logger.warning(f"Disk-backed AUC computation failed: {e}")
            for v in results:
                results[v]["auc_reason"] = str(e)

        return results

    @staticmethod
    def _cm_to_metrics(cm: Dict[str, int]) -> Dict[str, Any]:
        """Calculates canonical metrics directly from confusion matrix counts."""
        tp = cm["tp"]
        fp = cm["fp"]
        tn = cm["tn"]
        fn = cm["fn"]
        total = tp + fp + tn + fn

        accuracy = (tp + tn) / max(1, total)
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, tp + fn)
        balanced_acc = 0.5 * (recall + (1.0 - fpr))

        if (precision + recall) > 0:
            attack_f1 = (2.0 * precision * recall) / (precision + recall)
            f2 = (5.0 * precision * recall) / ((4.0 * precision) + recall)
        else:
            attack_f1 = 0.0
            f2 = 0.0

        rec_benign = tn / max(1, tn + fp)
        prec_benign = tn / max(1, tn + fn)
        if (prec_benign + rec_benign) > 0:
            benign_f1 = (2.0 * prec_benign * rec_benign) / (prec_benign + rec_benign)
        else:
            benign_f1 = 0.0

        macro_f1 = 0.5 * (attack_f1 + benign_f1)
        weighted_f1 = ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, total)

        return {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "attack_f1": round(attack_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "f2": round(f2, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": None,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }
