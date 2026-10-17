"""
FastAPI Backend for Adaptive ML-DST Cloud Attack Tolerance System.
Provides REST API endpoints for:
- Health & audit status
- Training & batch inference
- Live streaming controls (start/stop/status)
- Hierarchical risk telemetry (nodes, clusters, global)
- Defense alerts and event audit logs
- Real-time prequential metrics and experiment tracking
"""

import os
import time
import threading
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.data.audit import audit_dataset
from src.data.loader import split_data
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
from src.utils.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive ML-DST API",
    description="Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from src.inference.predictor import OnlineSGDPredictor

# Global System State Container
class SystemState:
    def __init__(self):
        self.config = load_config("configs/primary.yaml" if os.path.exists("configs/primary.yaml") else "configs/default.yaml")
        self.preprocessor = NetworkDataPreprocessor()
        self.online_clf = OnlineClassifier()
        self.anom_det = AnomalyEvidenceDetector()
        self.temporal_tracker = TemporalDeviationTracker()
        self.dst_engine = DSTEvidenceFusion()
        self.threshold_mgr = AdaptiveThresholdManager()
        self.hierarchy = HierarchicalRiskEngine()
        self.defense = SimulatedDefenseEngine()
        self.blockchain = BlockchainAuditAdapter()
        self.drift_det = ADWIN()

        # Primary saved model predictor
        self.model_path = "models/online_sgd_final_online.joblib"
        self.predictor: Optional[OnlineSGDPredictor] = None
        self.is_trained = False
        self._load_saved_predictor()

        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_stream_flag = False

        self.stream_stats = {
            "flows_processed": 0,
            "batches_processed": 0,
            "drift_alarms": 0,
            "alerts_triggered": 0,
            "current_threshold": 0.50,
            "start_time": 0.0,
            "elapsed_seconds": 0.0,
            "throughput_fps": 0.0,
        }

        self.evaluation_history = {
            "y_true": [],
            "y_pred": [],
            "risk_scores": [],
        }

        self.experiments_db: Dict[str, Any] = {}

    def _load_saved_predictor(self):
        """Attempts to load saved Online SGD model bundle. Returns False if absent without silent retraining."""
        if os.path.exists(self.model_path):
            try:
                self.predictor = OnlineSGDPredictor(model_path=self.model_path)
                self.is_trained = True
                logger.info(f"API loaded saved model from: {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")
                self.predictor = None
                self.is_trained = False
        else:
            self.predictor = None
            self.is_trained = False


state = SystemState()


# Pydantic Request/Response Models
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Numeric flow feature vector")
    threshold: Optional[float] = Field(None, description="Optional manual threshold override")
    operating_mode: Optional[str] = Field("balanced", description="Operating profile: 'balanced', 'high_recall', or 'volumetric'")


class BatchPredictRequest(BaseModel):
    flows: List[List[float]] = Field(..., description="Matrix of flow feature vectors")


class TrainRequest(BaseModel):
    sample_size: int = Field(5000, description="Training sample size for warm-up")
    random_seed: int = Field(42, description="Deterministic random seed")


# --- API Endpoints ---

@app.get("/health")
def get_health():
    """Health check endpoint."""
    return {
        "status": "HEALTHY",
        "service": "Adaptive ML-DST Engine",
        "version": "1.0.0",
        "is_trained": state.is_trained,
        "is_streaming": state.is_streaming,
        "flows_processed": state.stream_stats["flows_processed"],
        "blockchain_status": "MOCK_LEDGER" if state.blockchain.use_mock else "CONNECTED",
    }


@app.post("/dataset/audit")
def trigger_dataset_audit():
    """Returns the cached or newly generated dataset audit summary."""
    audit_file = "reports/dataset_audit/corrected_audit.json"
    if not os.path.exists(audit_file):
        audit_file = "reports/dataset_audit/audit_report.json"
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "total_rows" not in data and "total_raw_rows" in data:
                data["total_rows"] = data["total_raw_rows"]
            return data
    summary = audit_dataset("dataset", chunksize=50000)
    return summary


@app.post("/train")
def train_system(req: TrainRequest):
    """Initializes and trains baseline models, preprocessor, and anomaly detector."""
    sample_path = "data/processed/lightweight_sample.csv.gz"
    if not os.path.exists(sample_path):
        sample_path = "data/processed/stratified_sample.csv.gz"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Processed sample not found. Run data preparation first.")

    df = pd.read_csv(sample_path, nrows=req.sample_size)
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1, random_seed=req.random_seed)

    X_tr_df, y_tr_bin, y_tr_multi, _ = state.preprocessor.extract_and_clean_labels(train_df)
    state.preprocessor.fit(X_tr_df, y_tr_multi)

    X_train = state.preprocessor.transform(X_tr_df)
    y_train = y_tr_bin.values

    # Fit online classifier warm-up
    state.online_clf.partial_fit(X_train, y_train)

    # Fit anomaly detector on benign subset
    X_benign = X_train[y_train == 0]
    if len(X_benign) > 0:
        state.anom_det.fit(X_benign)

    # Initialize temporal tracker feature map
    state.temporal_tracker.resolve_feature_indices(state.preprocessor.feature_names_)

    state.is_trained = True
    return {
        "status": "TRAINED",
        "train_samples": len(X_train),
        "feature_count": len(state.preprocessor.feature_names_),
        "benign_samples": int(len(X_benign)),
        "attack_samples": int(len(X_train) - len(X_benign)),
    }


@app.get("/model/info")
def get_model_info():
    """Returns metadata, schema, and limitations for active saved model bundle."""
    meta_path = "models/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if state.predictor is not None:
        return {
            "checkpoint_stage": state.predictor.checkpoint_stage,
            "feature_count": len(state.predictor.feature_names),
            "threshold": state.predictor.threshold,
            "calibrated_thresholds": state.predictor.calibrated_thresholds,
        }
    raise HTTPException(
        status_code=404,
        detail="Model metadata not found at 'models/model_metadata.json'. Run 'python scripts/train_export.py' first.",
    )


@app.post("/predict")
def predict_flow(req: PredictRequest):
    """Predict attack probability and evaluate risk for a single network flow using saved bundle."""
    if not state.is_trained or state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model bundle not found at 'models/online_sgd_final_online.joblib'. Please run 'python scripts/train_export.py' to generate trained checkpoints.",
        )

    expected_len = len(state.predictor.feature_names)
    if len(req.features) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_len} features according to saved schema, got {len(req.features)}",
        )

    # Operating mode thresholds based on empirical calibration:
    # 'balanced': 0.50 (Recall 86.88%, FPR 2.66%)
    # 'high_recall': 0.28 (Recall 92.37%, FPR 5.89%)
    # 'volumetric': 0.40 (Recall 97.13%, FPR 2.87% on DoS/PortScan)
    mode_thresholds = {
        "balanced": 0.50,
        "high_recall": 0.28,
        "volumetric": 0.40,
    }
    active_thresh = req.threshold if req.threshold is not None else mode_thresholds.get(req.operating_mode or "balanced", 0.50)

    feat_dict = dict(zip(state.predictor.feature_names, req.features))
    res = state.predictor.predict_single(feat_dict, threshold_override=active_thresh)

    return {
        "prediction": res["predicted_label"],
        "is_attack": res["is_attack"],
        "attack_score": res["attack_score"],
        "risk_score": res["attack_score"],
        "active_threshold": res["threshold"],
        "operating_mode": req.operating_mode,
        "model_stage": res["model_stage"],
        "note": res["note"],
        "evidence": {
            "attack_score": res["attack_score"],
            "threshold": res["threshold"],
        },
    }


def _run_stream_worker():
    """Background streaming worker simulating live flow ingress."""
    logger.info("Background streaming simulation started.")
    sample_path = "data/processed/stratified_sample.csv.gz"
    df_stream = pd.read_csv(sample_path)
    X_df, y_bin, _, _ = state.preprocessor.extract_and_clean_labels(df_stream)

    batch_size = 100
    total_rows = len(X_df)
    state.stream_stats["start_time"] = time.time()

    for start_idx in range(0, total_rows, batch_size):
        if state.stop_stream_flag:
            break

        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = X_df.iloc[start_idx:end_idx]
        y_true_batch = y_bin.iloc[start_idx:end_idx].values

        X_batch = state.preprocessor.transform(batch_df)

        # 1. Sources
        probs = state.online_clf.predict_proba(X_batch)[:, 1]
        anoms = state.anom_det.score(X_batch)
        temps = state.temporal_tracker.compute_batch_deviations(X_batch, update_window=True)

        # 2. DST Fusion
        fused = state.dst_engine.fuse_batch(probs, anoms, temps)
        risks = fused["risk_scores"]

        # 3. Prequential prediction
        curr_thresh = state.threshold_mgr.get_threshold()
        preds = (risks >= curr_thresh).astype(int)
        errors = (preds != y_true_batch).astype(float)

        drift_detected = False
        for err in errors:
            if state.drift_det.update(err):
                drift_detected = True
                state.stream_stats["drift_alarms"] += 1

        updated_thresh = state.threshold_mgr.update(risks, drift_detected=drift_detected)
        state.stream_stats["current_threshold"] = updated_thresh

        # 4. Hierarchical simulation
        for i, r in enumerate(risks):
            flow_idx = start_idx + i
            node_id = state.hierarchy.assign_flow_to_node(flow_idx)
            state.hierarchy.update_node(
                node_id,
                risk_score=r,
                belief=fused["beliefs"][i],
                plausibility=fused["plausibilities"][i],
                uncertainty=fused["uncertainties"][i],
            )

            if r >= updated_thresh:
                state.stream_stats["alerts_triggered"] += 1
                node_obj = state.hierarchy.nodes[node_id]
                ev = state.defense.evaluate_node(
                    node_id=node_id,
                    cluster_id=node_obj.cluster_id,
                    node_risk=node_obj.current_risk,
                    current_threshold=updated_thresh,
                    replay_index=flow_idx,
                )
                if ev["action"] in ["RATE_LIMIT", "TRAFFIC_FILTER_ISOLATE"]:
                    state.blockchain.log_defense_event(
                        event_id=flow_idx,
                        evidence_payload=ev,
                        entity_id=node_id,
                        risk_category=2 if ev["action"] == "TRAFFIC_FILTER_ISOLATE" else 1,
                        action=ev["action"],
                    )

        h_state = state.hierarchy.aggregate_hierarchy()
        state.defense.evaluate_global(
            global_risk=h_state["global_risk"],
            cluster_risks=h_state["cluster_risks"],
            replay_index=end_idx,
        )

        # 5. Online model update
        state.online_clf.partial_fit(X_batch, y_true_batch)

        # Record metrics
        state.evaluation_history["y_true"].extend(y_true_batch.tolist())
        state.evaluation_history["y_pred"].extend(preds.tolist())
        state.evaluation_history["risk_scores"].extend(risks.tolist())

        state.stream_stats["flows_processed"] += len(X_batch)
        state.stream_stats["batches_processed"] += 1
        elapsed = max(0.001, time.time() - state.stream_stats["start_time"])
        state.stream_stats["elapsed_seconds"] = elapsed
        state.stream_stats["throughput_fps"] = round(state.stream_stats["flows_processed"] / elapsed, 1)

        time.sleep(0.05)  # Pace stream for live visual demonstration

    state.is_streaming = False
    logger.info("Streaming replay simulation finished.")


@app.post("/stream/start")
def start_streaming():
    """Start streaming replay simulation in the background."""
    if not state.is_trained:
        # Auto-train on small sample if not yet trained
        train_system(TrainRequest(sample_size=3000, random_seed=42))

    if state.is_streaming:
        return {"status": "ALREADY_RUNNING"}

    state.is_streaming = True
    state.stop_stream_flag = False
    state.stream_thread = threading.Thread(target=_run_stream_worker, daemon=True)
    state.stream_thread.start()
    return {"status": "STARTED"}


@app.post("/stream/stop")
def stop_streaming():
    """Stop the running stream simulation."""
    state.stop_stream_flag = True
    state.is_streaming = False
    return {"status": "STOPPING"}


@app.get("/stream/status")
def get_stream_status():
    """Get current streaming telemetry and throughput."""
    return {
        "is_streaming": state.is_streaming,
        "stats": state.stream_stats,
    }


@app.get("/risk/nodes")
def get_node_risks():
    """Get current risk telemetry of all simulated cloud nodes."""
    return state.hierarchy.get_highest_risk_nodes(top_k=state.hierarchy.num_nodes)


@app.get("/risk/clusters")
def get_cluster_risks():
    """Get current cluster risk levels."""
    return state.hierarchy.cluster_risks


@app.get("/risk/global")
def get_global_risk():
    """Get network-wide global cloud risk."""
    return {
        "global_risk": round(state.hierarchy.global_risk, 4),
        "active_threshold": round(state.threshold_mgr.get_threshold(), 4),
        "alarm_active": state.hierarchy.global_risk >= state.defense.critical_global_threshold,
    }


@app.get("/alerts")
def get_defense_alerts(limit: int = 50):
    """Get recent defense events and blockchain audit logs."""
    return {
        "action_counts": state.defense.action_counts,
        "recent_events": state.defense.get_recent_events(limit=limit),
        "total_blockchain_records": state.blockchain.mock_ledger.total_records,
    }


@app.get("/metrics")
def get_evaluation_metrics():
    """Calculate running prequential evaluation metrics on streaming history."""
    y_t = state.evaluation_history["y_true"]
    y_p = state.evaluation_history["y_pred"]
    r_s = state.evaluation_history["risk_scores"]

    if len(y_t) == 0:
        return {"status": "NO_STREAM_DATA_YET"}

    metrics = compute_binary_metrics(np.array(y_t), np.array(y_p), np.array(r_s))
    metrics["drift_alarms_total"] = state.stream_stats["drift_alarms"]
    metrics["flows_evaluated"] = len(y_t)
    metrics["current_threshold"] = state.stream_stats["current_threshold"]
    return metrics


@app.get("/drift/events")
def get_drift_events():
    """Retrieve all logged concept-drift events and adaptation alarms."""
    return {
        "total_alarms": state.stream_stats["drift_alarms"],
        "events": getattr(state, "drift_events_log", []),
    }


@app.get("/threshold/history")
def get_threshold_history(limit: int = 100):
    """Retrieve history of adaptive threshold calibrations and reasons."""
    return {
        "current_threshold": state.threshold_mgr.get_threshold(),
        "history": state.threshold_mgr.threshold_history[-limit:],
    }


@app.get("/experiments")
def list_experiments():
    """List all persisted experiment reports in reports/experiments/."""
    exp_dir = "reports/experiments"
    if not os.path.exists(exp_dir):
        return {"experiments": []}
    files = [f.replace(".json", "") for f in os.listdir(exp_dir) if f.endswith(".json")]
    return {"experiments": sorted(files)}


@app.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieve saved experiment metadata and metrics from disk."""
    exp_dir = "reports/experiments"
    fpath = os.path.join(exp_dir, f"{experiment_id}.json")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")


@app.get("/dashboard")
def serve_dashboard():
    """Serves the interactive SOC Web Dashboard."""
    dash_index = "dashboard/web/index.html"
    if os.path.exists(dash_index):
        return FileResponse(dash_index)
    raise HTTPException(status_code=404, detail="Dashboard index not found.")


# Mount static assets for web dashboard if present
if os.path.exists("dashboard/web"):
    app.mount("/static/web", StaticFiles(directory="dashboard/web"), name="web_dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
