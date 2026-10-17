"""
Unit and integration tests for V6 Release Verification.
Validates:
1. Exported model existence, bundle schema, and successful loading.
2. Fresh-process save/load prediction equivalence.
3. Strict feature-schema validation (rejecting malformed/missing features).
4. No accidental retraining during prediction (frozen weights).
5. API loading, health check, and prediction endpoints.
6. Evaluation artifact consistency against protocol_b_results.json.
"""

import os
import sys
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.inference.predictor import OnlineSGDPredictor
from src.api.main import app


def test_exported_model_files_exist():
    """Verify that both initial and final online checkpoints and metadata exist."""
    assert os.path.exists("models/online_sgd_initial.joblib"), "Initial checkpoint missing"
    assert os.path.exists("models/online_sgd_final_online.joblib"), "Final online checkpoint missing"
    assert os.path.exists("models/model_metadata.json"), "Model metadata missing"
    assert os.path.getsize("models/online_sgd_initial.joblib") > 2000
    assert os.path.getsize("models/online_sgd_final_online.joblib") > 2000


def test_model_bundle_structure_and_schema():
    """Verify bundle keys, scaler state, classifier weights, and metadata consistency."""
    bundle = joblib.load("models/online_sgd_final_online.joblib")
    expected_keys = [
        "model", "preprocessor", "scaler", "impute_values",
        "feature_names", "class_mapping", "decision_threshold",
        "effective_config", "checkpoint_stage", "dataset_fingerprint"
    ]
    for k in expected_keys:
        assert k in bundle, f"Key '{k}' missing from bundle"

    assert len(bundle["feature_names"]) == 69
    assert hasattr(bundle["model"].model, "coef_")
    assert bundle["model"].model.coef_.shape[1] == 69


def test_no_accidental_retraining_during_inference():
    """Verify that calling predict does not alter classifier weights, intercept, or scaler."""
    predictor = OnlineSGDPredictor("models/online_sgd_final_online.joblib")
    initial_coef = predictor.model.model.coef_.copy()
    initial_intercept = predictor.model.model.intercept_.copy()

    # Predict on 50 samples
    df_sample = pd.read_csv("data/example_input.csv")
    preds = predictor.predict_df(df_sample)

    assert len(preds) == 50
    # Assert exact array equality (frozen weights)
    np.testing.assert_array_equal(predictor.model.model.coef_, initial_coef)
    np.testing.assert_array_equal(predictor.model.model.intercept_, initial_intercept)


def test_fresh_process_prediction_equivalence():
    """Verify that loaded predictor reproduces predictions with 0.0 numerical discrepancy across fresh processes."""
    p1 = OnlineSGDPredictor("models/online_sgd_final_online.joblib")
    df_sample = pd.read_csv("data/example_input.csv")
    res1 = p1.predict_df(df_sample, flow_id_col="flow_id")

    sub_code = (
        "import sys, pandas as pd, json;"
        "from src.inference.predictor import OnlineSGDPredictor;"
        "df = pd.read_csv('data/example_input.csv');"
        "p = OnlineSGDPredictor('models/online_sgd_final_online.joblib');"
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
    scores2 = np.array(sub_res["scores"])
    preds2 = np.array(sub_res["preds"])

    scores1 = res1["attack_score"].values
    preds1 = res1["prediction"].values

    max_diff = float(np.max(np.abs(scores1 - scores2)))
    assert max_diff < 1e-9, f"Discrepancy detected: {max_diff}"
    assert (preds1 == preds2).all()


def test_schema_validation_and_column_handling():
    """Verify that predictor handles extra columns gracefully and errors on missing columns."""
    predictor = OnlineSGDPredictor("models/online_sgd_final_online.joblib")
    df_sample = pd.read_csv("data/example_input.csv").head(5).copy()

    # Extra columns should be safely ignored
    df_sample["unrelated_arbitrary_metadata"] = 999.9
    res = predictor.predict_df(df_sample)
    assert len(res) == 5

    # Missing required columns should raise ValueError
    df_broken = df_sample.drop(columns=[predictor.feature_names[0]])
    with pytest.raises(ValueError, match="Missing features"):
        predictor.predict_df(df_broken)


def test_api_saved_model_inference():
    """Verify FastAPI integration with the saved model bundle."""
    client = TestClient(app)
    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["is_trained"] is True

    # Info check
    res_info = client.get("/model/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    feat_count = info_data.get("feature_schema", {}).get("feature_count", info_data.get("feature_count"))
    assert feat_count == 69

    # Predict valid vector
    valid_payload = {"features": [0.0] * 69}
    res_pred = client.post("/predict", json=valid_payload)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert pred_data["prediction"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= pred_data["attack_score"] <= 1.0

    # Predict invalid vector (wrong dimension -> HTTP 400)
    invalid_payload = {"features": [0.0, 1.0]}
    res_bad = client.post("/predict", json=invalid_payload)
    assert res_bad.status_code == 400


def test_report_artifact_consistency():
    """Verify that table1_comparative_models.csv perfectly matches protocol_b_results.json."""
    json_path = "reports/final/metrics/protocol_b_results.json"
    csv_path = "reports/final/metrics/table1_comparative_models.csv"

    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    df_csv = pd.read_csv(csv_path)
    # Check that the primary SGD detector in CSV matches JSON exactly
    sgd_csv = df_csv[df_csv["Model"] == "2_Online_SGD_NoDrift"].iloc[0]
    sgd_json = results["aggregate_metrics"]["2_Online_SGD_NoDrift"]

    assert abs(sgd_csv["Accuracy"] - sgd_json["accuracy"]) < 1e-4
    assert abs(sgd_csv["Recall (ADR)"] - sgd_json["recall"]) < 1e-4
    assert abs(sgd_csv["FPR"] - sgd_json["false_positive_rate"]) < 1e-4
    assert abs(sgd_csv["Attack_F1"] - sgd_json["attack_f1"]) < 1e-4
