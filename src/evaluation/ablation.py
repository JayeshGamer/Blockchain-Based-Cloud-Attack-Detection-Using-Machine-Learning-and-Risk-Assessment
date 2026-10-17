"""
Phase 17: Fair Ablation Study Engine.
Evaluates 9 progressive architectural variants under strictly identical conditions:
  Variant A: Frozen static model (trained on Monday baseline only)
  Variant B: Online model without drift detection
  Variant C: Online model with drift detection but no active response
  Variant D: Online model with drift detection and active plasticity response
  Variant E: Online model plus unsupervised anomaly evidence
  Variant F: Online model plus temporal deviation evidence
  Variant G: Online model plus reliability-discounted DST fusion
  Variant H: Full model plus validated adaptive threshold (Mode 1: recall-constrained)
  Variant I: Full model plus hierarchical risk aggregation and simulated defense layer

Strict Fairness Guarantees:
- Identical dataset partitions and unified chronological stream ordering.
- Identical preprocessing and scaling (fitted exclusively on Monday baseline).
- Identical random seeds and initialization.
- Identical 1-batch delayed label feedback assumptions.
- Original predictions stored at arrival time; never recomputed on updated models.
- Independent mutable state for every variant.
"""

import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.data.ordering import order_chronological_stream
from src.features.preprocessor import NetworkDataPreprocessor
from src.features.temporal import TemporalDeviationTracker
from src.models.online_classifier import OnlineSGDClassifier
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.drift.adwin import create_adwin_detector, MultiSignalDriftCoordinator
from src.dst.evidence_fusion import DSTEvidenceFusion
from src.risk.adaptive_threshold import AdaptiveThresholdManager
from src.risk.hierarchical_risk import HierarchicalRiskEngine
from src.simulation.defense import SimulatedDefenseEngine
from src.utils.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_fair_ablation_study(
    df: pd.DataFrame,
    batch_size: int = 200,
    label_delay_batches: int = 1,
    random_seed: int = 42,
    output_dir: str = "reports/experiments",
) -> Dict[str, Any]:
    """
    Executes the 9-variant ablation study on the chronological stream.
    Applies unified chronological ordering to guarantee exact alignment with Protocol B.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting Fair Ablation Study (Variants A through I) ===")

    # 1. Apply unified chronological ordering
    df_ordered, stream_fingerprint = order_chronological_stream(df)
    logger.info(f"Unified chronological stream fingerprint: {stream_fingerprint[:16]}...")

    # Separate Monday baseline from streaming adaptation
    monday_mask = df_ordered["source_day"] == "Monday"
    df_monday = df_ordered[monday_mask].reset_index(drop=True)
    df_stream = df_ordered[~monday_mask].reset_index(drop=True)

    prep = NetworkDataPreprocessor(scaler_type="robust")
    X_mon_df, y_mon_bin, y_mon_multi, _ = prep.extract_and_clean_labels(df_monday)
    prep.fit(X_mon_df, y_mon_multi)

    X_monday = prep.transform(X_mon_df)
    y_monday = y_mon_bin.values

    # Pre-extract streaming features
    X_str_df, y_str_bin, _, _ = prep.extract_and_clean_labels(df_stream)
    X_stream = prep.transform(X_str_df)
    y_stream = y_str_bin.values
    days_stream = df_stream["source_day"].values

    variants = [
        "Variant A: Frozen Static Model",
        "Variant B: Online Model No Drift",
        "Variant C: Online Model + Drift (Passive)",
        "Variant D: Online Model + Drift (Active Response)",
        "Variant E: Online Model + Anomaly Evidence",
        "Variant F: Online Model + Temporal Evidence",
        "Variant G: Online Model + Discounted DST",
        "Variant H: Full Model + Adaptive Threshold",
        "Variant I: Full System + Hierarchy & Defense",
    ]

    # Initialize completely independent components per variant
    # Classifiers (independent instances)
    clf_a = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_b = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_c = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_d = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_e = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_f = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_g = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_h = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)
    clf_i = OnlineSGDClassifier(random_seed=random_seed).partial_fit(X_monday, y_monday)

    # Classifier-specific label-free cold-start threshold for Frozen Model
    s_mon_frozen = clf_a.predict_score(X_monday)
    thresh_frozen_clf = float(np.percentile(s_mon_frozen, 99.0))

    # Anomaly detectors (independent instances)
    anom_det_e = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=50).fit(X_monday)
    anom_det_g = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=50).fit(X_monday)
    anom_det_h = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=50).fit(X_monday)
    anom_det_i = AnomalyEvidenceDetector(random_seed=random_seed, n_estimators=50).fit(X_monday)

    # Temporal deviation trackers (independent instances)
    temp_tracker_f = TemporalDeviationTracker()
    temp_tracker_f.resolve_feature_indices(prep.feature_names_)

    temp_tracker_g = TemporalDeviationTracker()
    temp_tracker_g.resolve_feature_indices(prep.feature_names_)

    temp_tracker_h = TemporalDeviationTracker()
    temp_tracker_h.resolve_feature_indices(prep.feature_names_)

    temp_tracker_i = TemporalDeviationTracker()
    temp_tracker_i.resolve_feature_indices(prep.feature_names_)

    # DST fusion engines (independent instances)
    dst_engine_g = DSTEvidenceFusion(uncertainty_budget=0.15)
    dst_engine_h = DSTEvidenceFusion(uncertainty_budget=0.15)
    dst_engine_i = DSTEvidenceFusion(uncertainty_budget=0.15)

    # Drift detectors (independent instances)
    adwin_c = create_adwin_detector(delta=0.002, prefer_river=True)
    adwin_d = create_adwin_detector(delta=0.002, prefer_river=True)
    adwin_coord_h = MultiSignalDriftCoordinator(delta=0.002, prefer_river=True)
    adwin_coord_i = MultiSignalDriftCoordinator(delta=0.002, prefer_river=True)

    # Adaptive threshold managers (independent instances)
    thresh_mgr_h = AdaptiveThresholdManager(initial_threshold=0.50, min_threshold=0.15, max_threshold=0.85)
    thresh_mgr_i = AdaptiveThresholdManager(initial_threshold=0.50, min_threshold=0.15, max_threshold=0.85)

    hierarchy = HierarchicalRiskEngine(num_nodes=20, num_clusters=4, random_seed=random_seed)
    defense = SimulatedDefenseEngine()

    preds_dict: Dict[str, List[int]] = {v: [] for v in variants}
    scores_dict: Dict[str, List[float]] = {v: [] for v in variants}

    total_flows = len(X_stream)
    num_batches = int(np.ceil(total_flows / batch_size))

    # Delayed feedback queue: holds (xb, yb, b_idx, saved_preds, saved_scores, start)
    delayed_queue = []

    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, total_flows)
        xb = X_stream[start:end]
        yb = y_stream[start:end]
        day = days_stream[start]

        # Predict Before Update Phase
        # Variant A: Frozen Static
        sa = clf_a.predict_score(xb)
        pa = (sa >= thresh_frozen_clf).astype(int)

        # Variant B: Online No Drift
        sb = clf_b.predict_score(xb)
        pb = (sb >= 0.50).astype(int)

        # Variant C: Online + Drift Passive
        sc = clf_c.predict_score(xb)
        pc = (sc >= 0.50).astype(int)

        # Variant D: Online + Drift Active
        sd = clf_d.predict_score(xb)
        p_d = (sd >= 0.50).astype(int)

        # Variant E: Online + Anomaly
        se_ml = clf_e.predict_score(xb)
        se_anom = anom_det_e.score(xb)
        se = 0.60 * se_ml + 0.40 * se_anom
        pe = (se >= 0.50).astype(int)

        # Variant F: Online + Temporal
        sf_ml = clf_f.predict_score(xb)
        sf_temp = temp_tracker_f.compute_batch_deviations(xb, update_window=False)
        sf = 0.60 * sf_ml + 0.40 * sf_temp
        pf = (sf >= 0.50).astype(int)

        # Variant G: Online + Discounted DST
        sg_ml = clf_g.predict_score(xb)
        sg_anom = anom_det_g.score(xb)
        sg_temp = temp_tracker_g.compute_batch_deviations(xb, update_window=False)
        fused_g = dst_engine_g.fuse_batch(sg_ml, sg_anom, sg_temp)
        sg = fused_g["risk_scores"]
        pg = (sg >= 0.50).astype(int)

        # Variant H: Full Model + Adaptive Threshold
        sh_ml = clf_h.predict_score(xb)
        sh_anom = anom_det_h.score(xb)
        th_h = thresh_mgr_h.get_threshold()
        sh_temp = temp_tracker_h.compute_batch_deviations(
            xb, recent_risk_scores=sh_ml, update_window=False, high_risk_threshold=th_h
        )
        fused_h = dst_engine_h.fuse_batch(sh_ml, sh_anom, sh_temp)
        sh = fused_h["risk_scores"]
        ph = (sh >= th_h).astype(int)

        # Variant I: Full System + Hierarchy & Defense
        si_ml = clf_i.predict_score(xb)
        si_anom = anom_det_i.score(xb)
        th_i = thresh_mgr_i.get_threshold()
        si_temp = temp_tracker_i.compute_batch_deviations(
            xb, recent_risk_scores=si_ml, update_window=False, high_risk_threshold=th_i
        )
        fused_i = dst_engine_i.fuse_batch(si_ml, si_anom, si_temp)
        si = fused_i["risk_scores"]
        pi = (si >= th_i).astype(int)

        batch_p = {
            "Variant A: Frozen Static Model": pa,
            "Variant B: Online Model No Drift": pb,
            "Variant C: Online Model + Drift (Passive)": pc,
            "Variant D: Online Model + Drift (Active Response)": p_d,
            "Variant E: Online Model + Anomaly Evidence": pe,
            "Variant F: Online Model + Temporal Evidence": pf,
            "Variant G: Online Model + Discounted DST": pg,
            "Variant H: Full Model + Adaptive Threshold": ph,
            "Variant I: Full System + Hierarchy & Defense": pi,
        }
        batch_s = {
            "Variant A: Frozen Static Model": sa,
            "Variant B: Online Model No Drift": sb,
            "Variant C: Online Model + Drift (Passive)": sc,
            "Variant D: Online Model + Drift (Active Response)": sd,
            "Variant E: Online Model + Anomaly Evidence": se,
            "Variant F: Online Model + Temporal Evidence": sf,
            "Variant G: Online Model + Discounted DST": sg,
            "Variant H: Full Model + Adaptive Threshold": sh,
            "Variant I: Full System + Hierarchy & Defense": si,
        }

        for v in variants:
            preds_dict[v].extend(batch_p[v])
            scores_dict[v].extend(batch_s[v])

        # Hierarchy & Defense updates for Variant I
        for i_f, r_f in enumerate(si):
            flow_idx = start + i_f
            nid = hierarchy.assign_flow_to_node(flow_idx)
            hierarchy.update_node(
                nid,
                risk_score=r_f,
                belief=fused_i["beliefs"][i_f],
                plausibility=fused_i["plausibilities"][i_f],
                uncertainty=fused_i["uncertainties"][i_f],
            )
            node_obj = hierarchy.nodes[nid]
            defense.evaluate_flow_and_node(
                event_id=flow_idx,
                replay_index=flow_idx,
                source_day=day,
                node_id=nid,
                cluster_id=node_obj.cluster_id,
                ml_score=si_ml[i_f],
                anomaly_score=si_anom[i_f],
                temporal_score=si_temp[i_f],
                dst_belief=fused_i["beliefs"][i_f],
                dst_plausibility=fused_i["plausibilities"][i_f],
                dst_uncertainty=fused_i["uncertainties"][i_f],
                conflict=fused_i["conflicts"][i_f],
                current_threshold=th_i,
                drift_status=False,
                node_risk=node_obj.current_risk,
                ground_truth_is_attack=int(yb[i_f]),
            )

        # Delayed feedback queue (store original predictions and scores)
        delayed_queue.append((xb, yb, b_idx, batch_p, batch_s, start))

        if len(delayed_queue) > label_delay_batches:
            x_up, y_up, _, p_del, s_del, upd_start = delayed_queue.pop(0)

            clf_b.partial_fit(x_up, y_up)

            # Variant C passive drift check using saved original predictions
            err_c = (p_del["Variant C: Online Model + Drift (Passive)"] != y_up).astype(float)
            for ec in err_c:
                adwin_c.update(ec)
            clf_c.partial_fit(x_up, y_up)

            # Variant D active drift response using saved original predictions
            err_d = (p_del["Variant D: Online Model + Drift (Active Response)"] != y_up).astype(float)
            for ed in err_d:
                if adwin_d.update(ed):
                    clf_d.adapt_to_drift("plasticity_boost", boost_factor=2.0)
            clf_d.partial_fit(x_up, y_up)

            clf_e.partial_fit(x_up, y_up)

            # Variant F: isolated temporal tracker update
            temp_tracker_f.compute_batch_deviations(x_up, update_window=True)
            clf_f.partial_fit(x_up, y_up)

            # Variant G: isolated temporal tracker update
            temp_tracker_g.compute_batch_deviations(x_up, update_window=True)
            clf_g.partial_fit(x_up, y_up)

            # Variant H: isolated drift coordination, threshold update, and tracker update using saved predictions
            err_h = (p_del["Variant H: Full Model + Adaptive Threshold"] != y_up).astype(float)
            s_h_orig = s_del["Variant H: Full Model + Adaptive Threshold"]
            d_status_h = adwin_coord_h.update_batch(delayed_errors=err_h, risk_scores=s_h_orig, step_index=upd_start)
            if d_status_h["drift_detected"]:
                clf_h.adapt_to_drift(d_status_h["action_taken"])
            thresh_mgr_h.update(risk_scores=s_h_orig, drift_detected=d_status_h["drift_detected"], delayed_labels=y_up, step_index=upd_start)
            temp_tracker_h.compute_batch_deviations(
                x_up, recent_risk_scores=s_h_orig, update_window=True, high_risk_threshold=thresh_mgr_h.get_threshold()
            )
            clf_h.partial_fit(x_up, y_up)

            # Variant I: completely separate isolated drift coordination, threshold update, and tracker update using saved predictions
            err_i = (p_del["Variant I: Full System + Hierarchy & Defense"] != y_up).astype(float)
            s_i_orig = s_del["Variant I: Full System + Hierarchy & Defense"]
            d_status_i = adwin_coord_i.update_batch(delayed_errors=err_i, risk_scores=s_i_orig, step_index=upd_start)
            if d_status_i["drift_detected"]:
                clf_i.adapt_to_drift(d_status_i["action_taken"])
            thresh_mgr_i.update(risk_scores=s_i_orig, drift_detected=d_status_i["drift_detected"], delayed_labels=y_up, step_index=upd_start)
            temp_tracker_i.compute_batch_deviations(
                x_up, recent_risk_scores=s_i_orig, update_window=True, high_risk_threshold=thresh_mgr_i.get_threshold()
            )
            clf_i.partial_fit(x_up, y_up)

    # Drain remaining delayed batches
    while delayed_queue:
        x_up, y_up, _, _, _, _ = delayed_queue.pop(0)
        clf_b.partial_fit(x_up, y_up)
        clf_c.partial_fit(x_up, y_up)
        clf_d.partial_fit(x_up, y_up)
        clf_e.partial_fit(x_up, y_up)
        clf_f.partial_fit(x_up, y_up)
        clf_g.partial_fit(x_up, y_up)
        clf_h.partial_fit(x_up, y_up)
        clf_i.partial_fit(x_up, y_up)

    # Compute metrics for all 9 variants
    ablation_summary: Dict[str, Any] = {}
    for v in variants:
        p_arr = np.array(preds_dict[v])
        s_arr = np.array(scores_dict[v])
        ablation_summary[v] = compute_binary_metrics(y_stream, p_arr, s_arr)

    ablation_results = {
        "study": "Phase 17: Fair Ablation Study (Variants A through I)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "stream_fingerprint": stream_fingerprint,
        "stream_flows": total_flows,
        "warmup_flows": len(df_monday),
        "variants": ablation_summary,
    }

    out_path = os.path.join(output_dir, "ablation_study_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            ablation_results,
            f,
            indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else o.tolist() if hasattr(o, "tolist") else str(o),
        )
    logger.info(f"Saved Fair Ablation Study results to {out_path}")

    return ablation_results
