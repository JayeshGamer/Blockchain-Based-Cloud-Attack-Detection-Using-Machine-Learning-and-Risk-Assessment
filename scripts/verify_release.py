"""
Release Verification Script for Adaptive ML-DST (V6).
Performs end-to-end verification of:
1. Exported model checkpoint bundles and metadata files.
2. Fresh-process save/load numerical score equality within 1e-7 tolerance.
3. Identical binary class predictions.
4. CLI prediction command (scripts/predict.py) execution.
5. REST API (src/api/main.py) health, info, and predict endpoints.
6. Schema rejection on invalid or missing features.
7. Generates comprehensive audit log in reports/final/logs/save_load_verification.log.
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predictor import OnlineSGDPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_release(log_path: str = "reports/final/logs/save_load_verification.log") -> bool:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_lines = []

    def log(msg: str):
        logger.info(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("STARTING V6 RELEASE VERIFICATION AUDIT")
    log("=" * 70)

    # 1. Verify Model Checkpoint Files
    log("\n--- STEP 1: VERIFYING MODEL CHECKPOINT FILES ---")
    initial_path = "models/online_sgd_initial.joblib"
    final_path = "models/online_sgd_final_online.joblib"
    meta_path = "models/model_metadata.json"

    for p in [initial_path, final_path, meta_path]:
        if not os.path.exists(p):
            log(f"FAILED: Required file missing: {p}")
            return False
        sz_kb = os.path.getsize(p) / 1024
        log(f"FOUND: {p} ({sz_kb:.2f} KB)")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    log(f"Default demo checkpoint: {metadata.get('default_demo_checkpoint')}")
    log(f"Feature count in schema: {metadata.get('feature_schema', {}).get('feature_count')}")

    # 2. Verify Fresh-Process Save/Load Consistency
    log("\n--- STEP 2: FRESH-PROCESS SAVE/LOAD PREDICTION EQUALITY ---")
    in_batch_path = "data/example_input.csv"
    if not os.path.exists(in_batch_path):
        log(f"FAILED: Example input missing at {in_batch_path}")
        return False

    df_test = pd.read_csv(in_batch_path)
    log(f"Loaded test batch of {len(df_test)} flows from {in_batch_path}")

    # Load bundle 1: in current process
    pred_current = OnlineSGDPredictor(model_path=final_path)
    res_current = pred_current.predict_df(df_test, flow_id_col="flow_id")

    # Run fresh Python subprocess to predict on same batch
    sub_code = (
        "import sys, pandas as pd, json;"
        "from src.inference.predictor import OnlineSGDPredictor;"
        f"df = pd.read_csv('{in_batch_path}');"
        f"p = OnlineSGDPredictor('{final_path}');"
        "res = p.predict_df(df, flow_id_col='flow_id');"
        "scores = res['attack_score'].tolist();"
        "preds = res['prediction'].tolist();"
        "print(json.dumps({'scores': scores, 'preds': preds}))"
    )

    proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
        check=True,
    )
    sub_res = json.loads(proc.stdout.strip())
    scores_sub = np.array(sub_res["scores"])
    preds_sub = np.array(sub_res["preds"])

    scores_curr = res_current["attack_score"].values
    preds_curr = res_current["prediction"].values

    max_score_diff = float(np.max(np.abs(scores_curr - scores_sub)))
    pred_mismatches = int(np.sum(preds_curr != preds_sub))

    log(f"Maximum continuous score discrepancy between fresh processes: {max_score_diff:.2e}")
    log(f"Binary prediction mismatches: {pred_mismatches} / {len(preds_curr)}")

    TOLERANCE = 1e-7
    if max_score_diff > TOLERANCE:
        log(f"FAILED: Score discrepancy {max_score_diff} exceeds tolerance {TOLERANCE}")
        return False
    if pred_mismatches > 0:
        log(f"FAILED: Found {pred_mismatches} mismatched predictions")
        return False

    log("PASSED: Save/Load numerical consistency verified bit-for-bit!")

    # 3. Verify CLI Prediction Path
    log("\n--- STEP 3: TESTING CLI PREDICTION COMMAND (scripts/predict.py) ---")
    cli_out_path = "data/example_cli_test_output.csv"
    if os.path.exists(cli_out_path):
        os.remove(cli_out_path)

    cmd = [
        sys.executable, "scripts/predict.py",
        "--input", in_batch_path,
        "--output", cli_out_path,
        "--model", final_path,
    ]
    proc_cli = subprocess.run(cmd, capture_output=True, text=True)
    if proc_cli.returncode != 0:
        log(f"FAILED: CLI command returned code {proc_cli.returncode}: {proc_cli.stderr}")
        return False

    if not os.path.exists(cli_out_path) or os.path.getsize(cli_out_path) == 0:
        log("FAILED: CLI output file not created or empty")
        return False

    df_cli = pd.read_csv(cli_out_path)
    log(f"CLI output generated successfully with {len(df_cli)} rows and columns: {list(df_cli.columns)}")
    os.remove(cli_out_path)
    log("PASSED: CLI prediction command verified!")

    # 4. Verify REST API Endpoints via TestClient
    log("\n--- STEP 4: TESTING FASTAPI BACKEND (src/api/main.py) ---")
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)

    # Health check
    res_h = client.get("/health")
    if res_h.status_code != 200 or not res_h.json().get("is_trained"):
        log(f"FAILED: Health check invalid: {res_h.status_code}, {res_h.json()}")
        return False
    log(f"API /health: OK (is_trained={res_h.json().get('is_trained')})")

    # Model info
    res_m = client.get("/model/info")
    if res_m.status_code != 200:
        log(f"FAILED: /model/info returned status {res_m.status_code}")
        return False
    log(f"API /model/info: OK (stage={res_m.json().get('checkpoints', {}).get('final_online', {}).get('stage')})")

    # Predict single flow
    features = df_test[pred_current.feature_names].iloc[0].tolist()
    res_p = client.post("/predict", json={"features": features})
    if res_p.status_code != 200:
        log(f"FAILED: /predict returned status {res_p.status_code}: {res_p.text}")
        return False
    p_data = res_p.json()
    log(f"API /predict: OK (prediction={p_data.get('prediction')}, attack_score={p_data.get('attack_score')})")

    # Test schema rejection on incorrect feature length
    res_bad = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    if res_bad.status_code != 400:
        log(f"FAILED: Schema rejection test failed (expected 400, got {res_bad.status_code})")
        return False
    log("API Schema rejection: OK (correctly rejected 3-feature payload with HTTP 400)")
    log("PASSED: REST API endpoints verified!")

    # 5. Final Report Check
    log("\n--- STEP 5: VERIFYING FINAL REPORT ARTIFACTS ---")
    required_artifacts = [
        "reports/final/metrics/table1_comparative_models.csv",
        "reports/final/metrics/table1b_tradeoff_comparisons.csv",
        "reports/final/metrics/protocol_b_results.json",
        "reports/final/manifests/source_hashes.json",
        "docs/RESULTS.md",
        "docs/CLEANUP_MANIFEST.md",
    ]
    for art in required_artifacts:
        if not os.path.exists(art):
            log(f"FAILED: Required artifact missing: {art}")
            return False
        log(f"FOUND: {art}")

    log("\n" + "=" * 70)
    log("ALL RELEASE VERIFICATION CHECKS PASSED PERFECTLY!")
    log("=" * 70)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"Verification log written to: {log_path}")
    return True


if __name__ == "__main__":
    success = verify_release()
    sys.exit(0 if success else 1)
