"""
Task B: Secondary Multiclass and Attack Family Evaluation Engine.
Evaluates per-class detection performance across high-level attack families:
- BENIGN
- DoS
- DDoS
- PortScan
- Brute Force
- Web Attack
- Botnet
- Infiltration
- Exploit/Heartbleed
Produces per-class precision, recall, F1, and confusion matrix.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import split_data
from src.features.preprocessor import NetworkDataPreprocessor
from src.utils.metrics import compute_multiclass_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_task_b_multiclass(sample_path: str = "data/processed/lightweight_sample.csv.gz", random_seed: int = 42) -> None:
    logger.info("=== Starting Task B: Secondary Multiclass Evaluation ===")
    df = pd.read_csv(sample_path)

    train_df, val_df, test_df = split_data(df, test_size=0.25, val_size=0.15, random_seed=random_seed)

    prep = NetworkDataPreprocessor()
    X_tr_df, _, _, y_tr_fam = prep.extract_and_clean_labels(train_df)
    X_te_df, _, _, y_te_fam = prep.extract_and_clean_labels(test_df)

    prep.fit(X_tr_df)
    X_tr = prep.transform(X_tr_df)
    X_te = prep.transform(X_te_df)

    # Encode attack families
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on union of labels to ensure complete coverage
    all_fams = pd.concat([y_tr_fam, y_te_fam]).unique()
    le.fit(all_fams)

    y_tr_enc = le.transform(y_tr_fam)
    y_te_enc = le.transform(y_te_fam)
    classes = list(le.classes_)

    logger.info(f"Evaluating {len(classes)} Attack Families: {classes}")

    # 1. Incremental Online Multiclass Classifier
    online_mc = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        random_state=random_seed,
        warm_start=True,
    )
    online_mc.partial_fit(X_tr, y_tr_enc, classes=np.arange(len(classes)))

    # Prequential evaluation on test stream
    batch_size = 200
    n_test = len(X_te)
    y_pred_online = []

    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        xb = X_te[i:end]
        yb = y_te_enc[i:end]
        preds = online_mc.predict(xb)
        y_pred_online.extend(preds)
        online_mc.partial_fit(xb, yb)

    metrics_online = compute_multiclass_metrics(y_te_enc, np.array(y_pred_online), labels=np.arange(len(classes)))

    # 2. Static Multiclass Random Forest Benchmark
    rf_mc = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=random_seed, n_jobs=-1)
    rf_mc.fit(X_tr, y_tr_enc)
    y_pred_rf = rf_mc.predict(X_te)
    metrics_rf = compute_multiclass_metrics(y_te_enc, y_pred_rf, labels=np.arange(len(classes)))

    # Format per-class report table
    table_rows = []
    for idx, cls_name in enumerate(classes):
        rep_online = metrics_online["per_class_report"].get(str(idx), {})
        rep_rf = metrics_rf["per_class_report"].get(str(idx), {})
        sup = rep_online.get("support", 0)

        table_rows.append({
            "Attack Family": cls_name,
            "Support (Test Flows)": int(sup),
            "Reliability Status": "Reliable" if sup >= 20 else "Insufficient (<20 flows)",
            "Online Precision": round(float(rep_online.get("precision", 0.0)), 4),
            "Online Recall": round(float(rep_online.get("recall", 0.0)), 4),
            "Online F1": round(float(rep_online.get("f1-score", 0.0)), 4),
            "Static RF Precision": round(float(rep_rf.get("precision", 0.0)), 4),
            "Static RF Recall": round(float(rep_rf.get("recall", 0.0)), 4),
            "Static RF F1": round(float(rep_rf.get("f1-score", 0.0)), 4),
        })

    df_table = pd.DataFrame(table_rows)
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/experiments", exist_ok=True)

    csv_out = "reports/tables/table4_multiclass_families.csv"
    df_table.to_csv(csv_out, index=False)
    logger.info(f"Saved Task B table to {csv_out}")

    json_out = "reports/experiments/multiclass_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "online_metrics": metrics_online,
            "static_rf_metrics": metrics_rf,
            "classes": classes,
        }, f, indent=2)
    logger.info(f"Saved Task B JSON results to {json_out}")


if __name__ == "__main__":
    run_task_b_multiclass()
