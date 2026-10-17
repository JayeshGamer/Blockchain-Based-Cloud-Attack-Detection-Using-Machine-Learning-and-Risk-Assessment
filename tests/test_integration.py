"""
Integration test for end-to-end Adaptive ML-DST pipeline.
Tests prequential stream replay, DST fusion, adaptive thresholding,
hierarchical risk aggregation, defense activation, and blockchain logging.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import NetworkDataPreprocessor
from src.models.online_classifier import OnlineClassifier
from src.models.anomaly_detector import AnomalyEvidenceDetector
from src.features.temporal import TemporalDeviationTracker
from src.dst.evidence_fusion import DSTEvidenceFusion
from src.risk.adaptive_threshold import AdaptiveThresholdManager
from src.risk.hierarchical_risk import HierarchicalRiskEngine
from src.simulation.defense import SimulatedDefenseEngine
from src.blockchain.adapter import BlockchainAuditAdapter
from src.drift.adwin import ADWIN


def test_end_to_end_streaming_pipeline():
    # Load small subset from processed sample
    df = pd.read_csv("data/processed/stratified_sample.csv.gz", nrows=500)

    # 1. Preprocessor initialization
    prep = NetworkDataPreprocessor()
    X_df, y_bin, y_multi, y_fam = prep.extract_and_clean_labels(df)

    # Initial training on first 100 flows (baseline warm-up)
    prep.fit(X_df.iloc[:100], y_multi.iloc[:100])
    X_warm = prep.transform(X_df.iloc[:100])
    y_warm = y_bin.iloc[:100].values

    # 2. Components
    online_clf = OnlineClassifier(random_seed=42)
    online_clf.partial_fit(X_warm, y_warm)

    anom_det = AnomalyEvidenceDetector(random_seed=42, n_estimators=20)
    benign_warm = X_warm[y_warm == 0]
    if len(benign_warm) > 0:
        anom_det.fit(benign_warm)

    drift_det = ADWIN(delta=0.01, min_window_length=5, grace_period=20)
    temporal_tracker = TemporalDeviationTracker(window_size=50)
    temporal_tracker.resolve_feature_indices(prep.feature_names_)

    dst_engine = DSTEvidenceFusion(uncertainty_budget=0.15)
    threshold_mgr = AdaptiveThresholdManager(initial_threshold=0.50, min_threshold=0.20, max_threshold=0.85)
    hierarchy = HierarchicalRiskEngine(num_nodes=5, num_clusters=2, random_seed=42)
    defense = SimulatedDefenseEngine()
    blockchain = BlockchainAuditAdapter(enabled=True, use_mock=True)

    # 3. Simulate streaming replay for remaining 400 flows in batches of 50
    stream_X_df = X_df.iloc[100:]
    stream_y = y_bin.iloc[100:].values

    batch_size = 50
    total_batches = len(stream_X_df) // batch_size
    alert_count = 0

    for b in range(total_batches):
        batch_slice = slice(b * batch_size, (b + 1) * batch_size)
        batch_df = stream_X_df.iloc[batch_slice]
        batch_y_true = stream_y[batch_slice]

        # Transform features
        X_batch = prep.transform(batch_df)

        # Step A: Online ML prediction
        probs = online_clf.predict_proba(X_batch)[:, 1]

        # Step B: Anomaly score
        anom_scores = anom_det.score(X_batch)

        # Step C: Temporal deviation
        temp_scores = temporal_tracker.compute_batch_deviations(X_batch, update_window=True)

        # Step D: DST Evidence Fusion
        fused = dst_engine.fuse_batch(probs, anom_scores, temp_scores)
        risk_scores = fused["risk_scores"]

        # Step E: Prequential evaluation & drift update
        # 0/1 prediction error using active threshold
        curr_thresh = threshold_mgr.get_threshold()
        preds = (risk_scores >= curr_thresh).astype(int)
        errors = (preds != batch_y_true).astype(float)

        drift_flag = False
        for err in errors:
            if drift_det.update(err):
                drift_flag = True

        # Update adaptive threshold
        updated_thresh = threshold_mgr.update(risk_scores, drift_detected=drift_flag)
        assert 0.20 <= updated_thresh <= 0.85

        # Step F: Update hierarchical topology and simulated defenses
        for i, r in enumerate(risk_scores):
            flow_idx = 100 + b * batch_size + i
            assigned_node = hierarchy.assign_flow_to_node(flow_idx)
            hierarchy.update_node(
                assigned_node,
                risk_score=r,
                belief=fused["beliefs"][i],
                plausibility=fused["plausibilities"][i],
                uncertainty=fused["uncertainties"][i],
            )

            if r >= updated_thresh:
                alert_count += 1
                node_obj = hierarchy.nodes[assigned_node]
                ev = defense.evaluate_node(
                    node_id=assigned_node,
                    cluster_id=node_obj.cluster_id,
                    node_risk=node_obj.current_risk,
                    current_threshold=updated_thresh,
                    replay_index=flow_idx,
                )
                # Commit high/medium alerts to blockchain
                if ev["action"] in ["RATE_LIMIT", "TRAFFIC_FILTER_ISOLATE"]:
                    blockchain.log_defense_event(
                        event_id=flow_idx,
                        evidence_payload=ev,
                        entity_id=assigned_node,
                        risk_category=2 if ev["action"] == "TRAFFIC_FILTER_ISOLATE" else 1,
                        action=ev["action"],
                    )

        # Aggregate hierarchy
        hierarchy_state = hierarchy.aggregate_hierarchy()
        defense.evaluate_global(
            global_risk=hierarchy_state["global_risk"],
            cluster_risks=hierarchy_state["cluster_risks"],
            replay_index=100 + (b + 1) * batch_size,
        )

        # Step G: Prequential model update (predict-then-update)
        online_clf.partial_fit(X_batch, batch_y_true)

    # Verification invariants
    assert online_clf.sample_count_ > 100
    assert len(defense.defense_events) > 0
    assert len(blockchain.mock_ledger.records) > 0
    print(f"Integration test completed: {alert_count} alerts, {len(blockchain.mock_ledger.records)} blockchain events recorded.")
