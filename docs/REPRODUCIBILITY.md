# Reproducibility & Execution Guide: Adaptive ML-DST V6

This document provides step-by-step instructions to reproduce the training, evaluation, verification, and inference workflows for Adaptive ML-DST V6.

---

## 1. Environment Setup & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.7 64-bit on Windows and Linux)
- Node.js 18+ (optional, only needed for Next.js frontend dashboard)

### Clone & Install Dependencies
```bash
# Navigate to project root
cd d:\CNS_Project_And_Paper

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

The system expects the official **CICIDS2017** dataset CSV files in `dataset/` or preprocessed chronological samples in `data/processed/`.

```bash
# Verify raw dataset files and generate audit report
python scripts/prepare_data.py --audit

# Build the chronological streaming sample (8,445 Monday warmup, 41,555 streaming flows)
python scripts/prepare_data.py --sample-size 50000 --output data/processed/stratified_sample.csv.gz
```

*Note: Raw dataset files are kept in `dataset/` and excluded from git and release packages due to file size.*

---

## 3. Training & Model Export

Trains the Online SGD classifier strictly on Monday benign flows and executes chronological replay to produce both initial and final online checkpoints.

```bash
python scripts/train_export.py
```

### Outputs Generated:
- `models/online_sgd_initial.joblib` (~4.1 KB): Fitted strictly on Monday warm-up baseline (8,445 flows).
- `models/online_sgd_final_online.joblib` (~4.1 KB): Updated across 41,555 streaming flows with $D=1$ delayed feedback.
- `models/model_metadata.json` (~3.4 KB): Machine-readable metadata, feature schema (69 features), hashes, and defaults.
- `data/example_input.csv` & `data/example_output.csv`: Valid 50-flow batch demonstration.

---

## 4. Defensible Chronological Evaluation

Executes the standardized chronological prequential evaluation protocol across all comparative models under delayed feedback.

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

### Generated Artifacts:
- `reports/final/metrics/protocol_b_results.json`: Full benchmark metrics for all 8 models.
- `reports/final/metrics/table1_comparative_models.csv`: Primary comparative table.
- `reports/final/metrics/table1b_tradeoff_comparisons.csv`: Precision, recall, F1, F2, FPR, FNR trade-offs.
- `reports/final/metrics/table2_daily_breakdown.csv`: Per-day attack performance.
- `reports/final/metrics/table3_chronological_daily.csv`: Chronological accuracy across Monday–Friday.
- `reports/final/metrics/table4_drift_benchmark.csv`: Model comparison with and without ADWIN drift adaptation.
- `reports/final/figures/`: Publication-quality PNG plots (ROC curves, PR curves, score distributions, confusion matrices).
- `reports/final/manifests/source_hashes.json`: Cryptographic SHA-256 integrity hashes of all 55 source files.

---

## 5. Saved-Model Inference

### A. Command-Line Interface (CLI)
Predict on a batch of precomputed flow records from CSV without retraining:

```bash
# Predict on example input using default model (models/online_sgd_final_online.joblib)
python scripts/predict.py --input data/example_input.csv --output data/predictions.csv

# View first few predictions
# Columns: flow_id, attack_score, threshold, prediction, predicted_label
```

Options:
- `--model-path`: Specify alternative checkpoint (e.g. `models/online_sgd_initial.joblib`).
- `--threshold`: Override default 0.50 decision threshold (e.g. `--threshold 0.70`).

### B. REST API (FastAPI Backend)
Start the backend server:

```bash
# Run server on port 8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

#### Query API Endpoints:
```bash
# Health check (verifies model is loaded)
curl http://localhost:8000/health

# Model schema and metadata
curl http://localhost:8000/model/info

# Predict on single flow (pass 69 numerical flow features)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"features\": [0.0, 120.0, 5.0, ...]}"
```

---

## 6. Release Verification Audit

Run the automated 5-step release audit script to verify bundle integrity, fresh-process bit-level consistency, CLI execution, API endpoints, and artifact presence:

```bash
python scripts/verify_release.py
```

Expected output:
- Step 1: Model checkpoint files confirmed.
- Step 2: Fresh-process save/load prediction discrepancy = `0.00e+00` (identical predictions).
- Step 3: CLI prediction command verified on 50-flow batch.
- Step 4: FastAPI endpoints verified (`/health`, `/model/info`, `/predict`, schema rejection).
- Step 5: Final metrics and report artifacts confirmed.
- Status: `ALL RELEASE VERIFICATION CHECKS PASSED PERFECTLY!`

---

## 7. Running the Test Suite

Execute the pytest suite:

```bash
# Run all unit and integration tests
pytest tests/ -v
```
