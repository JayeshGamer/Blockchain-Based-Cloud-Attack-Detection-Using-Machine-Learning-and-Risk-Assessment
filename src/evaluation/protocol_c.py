"""
Protocol C: Controlled Concept-Drift Benchmark Engine.
Evaluates drift detection delay, false alarms, missed drifts, and post-drift recovery
under known, ground-truth change points on controlled synthetic-ordered streams.

Rigor Specifications:
1. Dedicated warm-up period (first 500 flows) fitted on disjoint benign baseline.
2. Warm-up flows strictly excluded from all scored evaluation metrics.
3. Operational alarm timestamps recorded when alarm becomes available under delayed feedback.
4. Detector-only timestamps recorded separately to measure pure detection vs operational latency.
5. Strict one-to-one drift matching with configurable tolerance window.
6. Zero warm-up/evaluation hash overlap.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.features.preprocessor import NetworkDataPreprocessor
from src.features.temporal import TemporalDeviationTracker
from src.models.online_classifier import OnlineSGDClassifier
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.drift.adwin import create_adwin_detector, MultiSignalDriftCoordinator
from src.dst.evidence_fusion import DSTEvidenceFusion
from src.risk.adaptive_threshold import AdaptiveThresholdManager
from src.utils.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_controlled_drift_stream(
    df: pd.DataFrame,
    warmup_flows: int = 500,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """
    Constructs a controlled evaluation stream with a dedicated disjoint warm-up period
    and 5 ground-truth change points.

    Returns:
        (warmup_df, true_drift_events, evaluated_stream_df)
    """
    np.random.seed(random_seed)

    label_col = "NormalizedLabel" if "NormalizedLabel" in df.columns else "Label"
    df_benign = df[df[label_col] == "BENIGN"].reset_index(drop=True)
    df_dos = df[df["AttackFamily"] == "DoS"].reset_index(drop=True)
    df_portscan = df[df["AttackFamily"] == "PortScan"].reset_index(drop=True)
    df_bruteforce = df[df["AttackFamily"] == "Brute Force"].reset_index(drop=True)

    # Dedicated disjoint warm-up split from benign traffic (zero overlap with evaluation stream)
    warmup_df = df_benign.iloc[:warmup_flows].copy().reset_index(drop=True)
    df_benign_eval = df_benign.iloc[warmup_flows:].copy().reset_index(drop=True)

    stream_blocks: List[pd.DataFrame] = []
    drift_points: List[Dict[str, Any]] = []

    # Segment 1: Stable Normal Baseline (500 flows: global t = 500 to 999) - 100% Benign
    b1 = df_benign_eval.sample(n=500, replace=True, random_state=random_seed)
    stream_blocks.append(b1)

    # Change Point 1: Sudden Prevalence Drift at global t = 1000 (0% -> 40% DoS)
    drift_points.append({
        "change_point_idx": 1000,
        "drift_type": "Sudden Attack Prevalence",
        "description": "Sudden surge of DoS Hulk attack traffic (40% prevalence)",
    })
    b2_benign = df_benign_eval.sample(n=600, replace=True, random_state=random_seed + 1)
    b2_dos = df_dos.sample(n=400, replace=True, random_state=random_seed + 1)
    b2 = pd.concat([b2_benign, b2_dos]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    stream_blocks.append(b2)

    # Change Point 2: Gradual Prevalence Drift at global t = 2000 (Linear increase from 10% to 50% Brute Force)
    drift_points.append({
        "change_point_idx": 2000,
        "drift_type": "Gradual Attack Prevalence",
        "description": "Gradual ramp of Brute Force attempts over 1000 flows",
    })
    g_rows = []
    for step in range(1000):
        p_atk = 0.10 + 0.40 * (step / 1000.0)
        if np.random.rand() < p_atk and len(df_bruteforce) > 0:
            row = df_bruteforce.sample(n=1, random_state=random_seed + step)
        else:
            row = df_benign_eval.sample(n=1, random_state=random_seed + step)
        g_rows.append(row)
    b3 = pd.concat(g_rows, ignore_index=True)
    stream_blocks.append(b3)

    # Change Point 3: Return to Normal Traffic at global t = 3000
    drift_points.append({
        "change_point_idx": 3000,
        "drift_type": "Return to Normal",
        "description": "Cessation of attack traffic; return to pure benign telemetry",
    })
    b4 = df_benign_eval.sample(n=1000, replace=True, random_state=random_seed + 3)
    stream_blocks.append(b4)

    # Change Point 4: Sudden Feature Distribution Drift at global t = 4000 (Covariate Shift)
    drift_points.append({
        "change_point_idx": 4000,
        "drift_type": "Sudden Feature Covariate Shift",
        "description": "Sudden change in traffic volume characteristics (packet size & rates doubled)",
    })
    b5 = df_benign_eval.sample(n=1000, replace=True, random_state=random_seed + 4).copy()
    feature_cols = [c for c in b5.columns if c not in ["source_file", "source_day", "original_row_index", "row_hash", "Label", "NormalizedLabel", "AttackFamily"]]
    rate_cols = [c for c in feature_cols if any(k in c.lower() for k in ["rate", "packet", "byte", "len"])]
    b5[rate_cols] = b5[rate_cols] * 2.5
    stream_blocks.append(b5)

    # Change Point 5: Unseen Attack Family Emergence at global t = 5000 (PortScan appears)
    drift_points.append({
        "change_point_idx": 5000,
        "drift_type": "Unseen Attack Emergence",
        "description": "Appearance of completely unseen PortScan reconnaissance attacks (50% prevalence)",
    })
    b6_ben = df_benign_eval.sample(n=500, replace=True, random_state=random_seed + 5)
    b6_ps = df_portscan.sample(n=500, replace=True, random_state=random_seed + 5)
    b6 = pd.concat([b6_ben, b6_ps]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    stream_blocks.append(b6)

    evaluated_stream = pd.concat(stream_blocks, ignore_index=True)
    return warmup_df, drift_points, evaluated_stream


def evaluate_one_to_one_drift_matching(
    operational_alarms: List[int],
    true_change_points: List[Dict[str, Any]],
    tolerance_window: int = 250,
    detector_alarms: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous one-to-one matching between true drift change points and alarms.
    Evaluates both operational alarm delay (including label latency) and detector delay.

    Strict partition guarantees:
      total_alarms = matched_alarms + duplicate_alarms + false_alarms
      true_change_points = matched_events + missed_events
    """
    true_points = sorted([cp["change_point_idx"] for cp in true_change_points])
    alarms = sorted(operational_alarms)

    matched_events: List[Dict[str, Any]] = []
    duplicate_alarms: List[int] = []
    false_alarms: List[int] = []
    matched_cp_map: Dict[int, int] = {}

    for i_alm, alarm in enumerate(alarms):
        candidate_cps = [cp for cp in true_points if cp <= alarm <= (cp + tolerance_window)]

        if not candidate_cps:
            false_alarms.append(alarm)
        else:
            cp = candidate_cps[0]
            if cp not in matched_cp_map:
                op_delay = alarm - cp
                matched_cp_map[cp] = alarm
                det_delay = (detector_alarms[i_alm] - cp) if (detector_alarms and i_alm < len(detector_alarms)) else op_delay
                matched_events.append({
                    "change_point": cp,
                    "operational_alarm": alarm,
                    "operational_delay": op_delay,
                    "detector_only_delay": det_delay,
                })
            else:
                duplicate_alarms.append(alarm)

    matched_count = len(matched_events)
    total_true = len(true_points)
    missed_count = total_true - matched_count
    detection_rate = float(matched_count / total_true) if total_true > 0 else 0.0

    op_delays = [m["operational_delay"] for m in matched_events]
    det_delays = [m["detector_only_delay"] for m in matched_events]

    mean_op_delay = float(np.mean(op_delays)) if op_delays else None
    med_op_delay = float(np.median(op_delays)) if op_delays else None
    mean_det_delay = float(np.mean(det_delays)) if det_delays else None
    med_det_delay = float(np.median(det_delays)) if det_delays else None

    assert len(alarms) == matched_count + len(duplicate_alarms) + len(false_alarms), (
        f"Alarm conservation violated: {len(alarms)} != {matched_count} + {len(duplicate_alarms)} + {len(false_alarms)}"
    )

    return {
        "true_drift_events": total_true,
        "detected_alarms_total": len(alarms),
        "true_alarms_matched": matched_count,
        "detection_rate": round(detection_rate, 4),
        "missed_events": missed_count,
        "missed_drifts": missed_count,
        "false_alarms": len(false_alarms),
        "duplicate_alarms": len(duplicate_alarms),
        "tolerance_window_flows": tolerance_window,
        # Operational delay (includes label latency)
        "mean_detection_delay_flows": round(mean_op_delay, 1) if mean_op_delay is not None else None,
        "median_detection_delay_flows": round(med_op_delay, 1) if med_op_delay is not None else None,
        # Detector-only delay (excluding label latency)
        "mean_detector_only_delay_flows": round(mean_det_delay, 1) if mean_det_delay is not None else None,
        "median_detector_only_delay_flows": round(med_det_delay, 1) if med_det_delay is not None else None,
        "detection_delays": op_delays,
        "matched_details": matched_events,
        "duplicate_alarm_steps": duplicate_alarms,
        "false_alarm_steps": false_alarms,
    }


def run_protocol_c(
    df: pd.DataFrame,
    warmup_flows: int = 500,
    batch_size: int = 50,
    label_delay_batches: int = 1,
    tolerance_window: int = 250,
    random_seed: int = 42,
    output_dir: str = "reports/experiments",
) -> Dict[str, Any]:
    """
    Executes Protocol C: Controlled Concept-Drift Benchmark with genuine delayed feedback.
    Enforces strict exclusion of warm-up data from scored metrics.
    Logs operational alarm availability indices to prevent backdating.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting Protocol C: Controlled Concept-Drift Benchmark ===")

    warmup_df, true_drift_events, eval_df = build_controlled_drift_stream(
        df, warmup_flows=warmup_flows, random_seed=random_seed
    )
    logger.info(
        f"Built Protocol C stream: Warm-up={len(warmup_df):,} flows (excluded from metrics) | "
        f"Evaluated Stream={len(eval_df):,} flows with {len(true_drift_events)} ground-truth change points."
    )

    prep = NetworkDataPreprocessor()
    X_warm_df, y_warm_bin, _, _ = prep.extract_and_clean_labels(warmup_df)
    prep.fit(X_warm_df)

    X_warmup = prep.transform(X_warm_df)
    y_warmup = y_warm_bin.values

    # Pre-extract evaluated stream features
    X_eval_df, y_eval_bin, _, _ = prep.extract_and_clean_labels(eval_df)
    X_eval = prep.transform(X_eval_df)
    y_eval = y_eval_bin.values

    n_eval = len(X_eval)

    # Initialize Online Learner, ADWIN, DST, and Adaptive Threshold strictly on warm-up data
    online_sgd = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_warmup, y_warmup)
    anom_det = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=30).fit(X_warmup)
    temp_tracker = TemporalDeviationTracker()
    temp_tracker.resolve_feature_indices(prep.feature_names_)

    drift_coord = MultiSignalDriftCoordinator(delta=0.002, prefer_river=True)
    dst_engine = DSTEvidenceFusion(uncertainty_budget=0.15)
    thresh_mgr = AdaptiveThresholdManager(initial_threshold=0.50, min_threshold=0.15, max_threshold=0.85)

    all_preds = []
    all_risks = []
    operational_alarms: List[int] = []
    detector_only_alarms: List[int] = []
    alarm_records: List[Dict[str, Any]] = []

    # Delayed queue: holds (xb, yb, preds, risks, global_start)
    delayed_queue = []

    # Evaluated stream begins at global index warmup_flows (500)
    for offset in range(0, n_eval, batch_size):
        global_start = warmup_flows + offset
        end_offset = min(offset + batch_size, n_eval)
        global_end = warmup_flows + end_offset

        xb = X_eval[offset:end_offset]
        yb = y_eval[offset:end_offset]

        # Predict Before Update Phase
        p_ml = online_sgd.predict_score(xb)
        p_anom = anom_det.score(xb)
        curr_t = thresh_mgr.get_threshold()
        p_temp = temp_tracker.compute_batch_deviations(
            xb, recent_risk_scores=p_ml, update_window=False, high_risk_threshold=curr_t
        )
        fused = dst_engine.fuse_batch(p_ml, p_anom, p_temp)
        risks = fused["risk_scores"]
        preds = (risks >= curr_t).astype(int)

        all_preds.extend(preds)
        all_risks.extend(risks)

        delayed_queue.append((xb, yb, preds, risks, global_start))

        # Delayed feedback processing
        if len(delayed_queue) > label_delay_batches:
            x_up, y_up, p_up, r_up, upd_global_start = delayed_queue.pop(0)

            err = (p_up != y_up).astype(float)
            drift_status = drift_coord.update_batch(
                delayed_errors=err,
                risk_scores=r_up,
                step_index=upd_global_start,
            )

            if drift_status["drift_detected"]:
                # Operational alarm availability is global_start (current arrival time)
                # Detector-only alarm is upd_global_start (originating flow time)
                operational_alarms.append(global_start)
                detector_only_alarms.append(upd_global_start)
                alarm_records.append({
                    "operational_alarm_idx": global_start,
                    "originating_flow_idx": upd_global_start,
                    "label_delay_flows": global_start - upd_global_start,
                    "signals": drift_status["alarm_signals"],
                    "action": drift_status["action_taken"],
                })
                online_sgd.adapt_to_drift(drift_status["action_taken"])

            thresh_mgr.update(
                risk_scores=r_up,
                drift_detected=drift_status["drift_detected"],
                delayed_labels=y_up,
                step_index=upd_global_start,
            )
            temp_tracker.compute_batch_deviations(
                x_up, recent_risk_scores=r_up, update_window=True, high_risk_threshold=thresh_mgr.get_threshold()
            )
            online_sgd.partial_fit(x_up, y_up)

    # Drain remaining delayed queue
    while delayed_queue:
        x_up, y_up, _, _, _ = delayed_queue.pop(0)
        online_sgd.partial_fit(x_up, y_up)

    # Evaluate detection metrics using one-to-one matching
    drift_metrics = evaluate_one_to_one_drift_matching(
        operational_alarms=operational_alarms,
        true_change_points=true_drift_events,
        tolerance_window=tolerance_window,
        detector_alarms=detector_only_alarms,
    )

    # Segment Performance Evaluation strictly on evaluated stream flows
    # Change points at global indices: 1000, 2000, 3000, 4000, 5000
    # Evaluated stream indices: 0..500 (global 500..1000), 500..1500 (global 1000..2000), etc.
    segment_metrics = {}
    segments = [
        ("Segment 1: Normal Baseline", 0, 500),
        ("Segment 2: Sudden DoS Surge", 500, 1500),
        ("Segment 3: Gradual Brute Force", 1500, 2500),
        ("Segment 4: Return to Normal", 2500, 3500),
        ("Segment 5: Covariate Shift", 3500, 4500),
        ("Segment 6: Unseen PortScan Emergence", 4500, 5500),
    ]

    for seg_name, s_start, s_end in segments:
        y_seg = y_eval[s_start:s_end]
        p_seg = np.array(all_preds[s_start:s_end])
        r_seg = np.array(all_risks[s_start:s_end])
        segment_metrics[seg_name] = compute_binary_metrics(y_seg, p_seg, r_seg)

    protocol_c_results = {
        "protocol": "Protocol C: Controlled Concept-Drift Benchmark",
        "timestamp": pd.Timestamp.now().isoformat(),
        "warmup_flows_excluded": warmup_flows,
        "evaluated_stream_flows": n_eval,
        "true_change_points": true_drift_events,
        "operational_alarms": operational_alarms,
        "detector_only_alarms": detector_only_alarms,
        "alarm_records": alarm_records,
        "drift_detection_performance": drift_metrics,
        "segment_metrics": segment_metrics,
    }

    out_json = os.path.join(output_dir, "protocol_c_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            protocol_c_results,
            f,
            indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else o.tolist() if hasattr(o, "tolist") else str(o),
        )
    logger.info(f"Saved Protocol C results to {out_json}")

    return protocol_c_results
