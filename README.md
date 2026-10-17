# Blockchain-Based Cloud Attack Detection Using Machine Learning and Risk Assessment

### Adaptive ML-DST V6 — Streaming Intrusion Detection & Attack Tolerance

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-87-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible, research-grade evaluation and deployment package for **continuous, streaming network
intrusion detection in cloud environments**, with **hierarchical risk assessment**, **concept-drift
monitoring**, and a **blockchain-anchored audit trail**.

The project evaluates streaming network intrusion detection under realistic **prequential
(predict-before-update)** evaluation with **delayed label feedback ($D=1$)**, comparing standard
online classifiers against multi-source evidence fusion, and reports negative results alongside
positive ones.

---

## 📌 Primary Model Architecture & Scientific Status

- **Primary Streaming Model:** Industry-standard **Online SGD Logistic Regression**
  (`SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4)`) with `RobustScaler` preprocessing fit
  strictly on baseline Monday data. This is a standard convex linear model, **not a novel algorithm**.
- **Historical & Verified Performance (50,000-flow chronological stream; 41,555 evaluated flows):**
  - **Accuracy:** 93.56%
  - **Attack Recall:** 86.88%
  - **False Positive Rate (FPR):** 2.66%
  - **Attack F1-Score:** 0.9069
  - **ROC-AUC:** 0.9432 | **PR-AUC:** 0.9225
- **Evidence Fusion (ML-DST):** Retained as an experimental comparison. Under fixed 0.50 decision
  thresholding, Dempster-Shafer Theory (DST) achieves 42.69%–47.25% recall, underperforming standalone
  SGD (86.88%). **No claim is made that DST improves performance over the primary SGD detector.**

> **Reproducibility note:** [`reports/final/manifests/source_hashes.json`](reports/final/manifests/source_hashes.json)
> records the SHA-256 hash of every source file **as it existed during the evaluation run** that
> produced these results. It is a generated snapshot, not a guarantee about files edited afterwards —
> re-run `python scripts/evaluate.py` to regenerate it for the current tree. Line-ending normalisation
> is deliberately disabled in [`.gitattributes`](.gitattributes) so the committed bytes match the
> evaluated bytes on every platform.

---

## 📂 Project Structure

```
Blockchain-Based-Cloud-Attack-Detection/
├── README.md                          # Project documentation and entry points
├── LICENSE                            # MIT license
├── requirements.txt                   # Tested Python dependencies
├── pyproject.toml                     # Pytest and packaging configuration
├── .env.example                       # Configuration template
├── configs/
│   ├── primary.yaml                   # Production Online SGD configuration
│   ├── evaluation.yaml                # Standardized chronological evaluation schema
│   ├── lightweight.yaml               # Reduced-cost experiment profile
│   ├── full.yaml                      # Full multi-module pipeline configuration
│   └── default.yaml                   # Default pipeline configuration
├── src/
│   ├── data/                          # Streaming loader and dataset audit
│   ├── features/                      # Preprocessing (RobustScaler, median imputation)
│   ├── models/                        # Online SGD classifier, anomaly evidence detector, baselines
│   ├── inference/                     # Standalone predictor with frozen weights
│   ├── evaluation/                    # Prequential protocols A/B/C, ablation, metrics
│   ├── drift/                         # ADWIN concept drift detector
│   ├── dst/                           # Dempster-Shafer evidence fusion
│   ├── risk/                          # Hierarchical risk aggregation & adaptive thresholds
│   ├── simulation/                    # Simulated cloud defense mitigation
│   ├── blockchain/                    # Web3 audit trail adapter & mock ledger
│   ├── capture/                       # Live flow capture interface
│   ├── api/                           # FastAPI REST backend
│   └── utils/                         # Config validation & metric calculations
├── models/
│   ├── online_sgd_initial.joblib      # Baseline model fit on Monday warm-up (~4.1 KB)
│   ├── online_sgd_final_online.joblib # Final streaming model with D=1 feedback (~4.1 KB)
│   └── model_metadata.json            # Checkpoint metadata, schema & fingerprints
├── data/
│   ├── README.md                      # Dataset documentation and column schema
│   ├── raw/                           # (empty) drop raw captures here
│   ├── example_input.csv              # 50-flow validation batch (25 benign, 25 attack)
│   ├── example_output.csv             # Reference predictions for validation batch
│   └── processed/                     # Chronologically sorted streaming samples
├── dataset/
│   └── README.md                      # Raw CICIDS2017 download instructions (CSVs not tracked)
├── reports/
│   └── final/
│       ├── metrics/                   # Formatted CSV evaluation tables & JSON
│       ├── figures/                   # ROC, PR, confusion matrix & drift plots
│       ├── manifests/                 # SHA-256 source code hashes & resolved configs
│       └── logs/                      # Save/load and pytest execution logs
├── scripts/
│   ├── prepare_data.py                # Dataset audit and chronological sampler
│   ├── train_export.py                # Model training and checkpoint export
│   ├── evaluate.py                    # Standardized evaluation protocol runner
│   ├── predict.py                     # CLI saved-model inference on CSV
│   ├── run_experiments.py             # Batch experiment driver
│   ├── run_multiclass.py              # Multi-class attack-category experiment
│   ├── generate_plots.py              # Figure generation
│   ├── generate_report.py             # Report table generation
│   ├── simulate_live_attack.py        # End-to-end live-attack simulation
│   ├── verify_prediction_integrity.py # Checkpoint consistency checker
│   └── verify_release.py              # Automated 5-step release verification audit
├── tests/                             # 87 automated unit and regression tests
├── dashboard/                         # Streamlit + web visualization dashboard
├── contracts/
│   ├── RiskAuditLedger.sol            # Solidity audit ledger contract
│   └── test/                          # Contract tests
├── archive_obsolete/                  # Archived diagnostics & superseded versions
│   └── paper/                         # Draft IEEE paper (writing postponed)
└── docs/
    ├── METHODOLOGY.md                 # System mechanics, assumptions & contributions
    ├── RESULTS.md                     # Comprehensive benchmark metrics and trade-offs
    ├── LIMITATIONS.md                 # Honest boundaries, FPR costs & scope constraints
    ├── REPRODUCIBILITY.md             # End-to-end execution guide
    └── CLEANUP_MANIFEST.md            # Inventory of archived and deleted files
```

---

## 🗂️ Dataset

This project uses the **CICIDS2017** intrusion detection dataset from the Canadian Institute for
Cybersecurity (2,830,743 flows across 8 capture files, ~1.0 GB of CSV).

**The raw CSVs are not tracked by Git** — individual files exceed GitHub's 100 MB limit and the set
totals ~843 MB on disk. Download instructions and the exact expected filenames are in
[`dataset/README.md`](dataset/README.md).

**You do not need the raw data to run anything.** The evaluated 50,000-flow chronological sample
(8,445 Monday warm-up + 41,555 Tuesday–Friday streaming flows) ships as
[`data/processed/stratified_sample.csv.gz`](data/processed/stratified_sample.csv.gz), so evaluation,
inference, the API, and the full test suite all work on a fresh clone. The raw captures are only
needed to regenerate that sample from scratch with `scripts/prepare_data.py`.

---

## ⚡ Quick Start: Commands

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Verification Audit
Runs the 5-step automated audit confirming model integrity, bit-level save/load equivalence, CLI
execution, and API endpoints:
```bash
python scripts/verify_release.py
```

### 3. Predict from CSV (Command Line)
Predict attack scores and binary alerts using the saved model bundle (zero weight updates):
```bash
python scripts/predict.py --input data/example_input.csv --output data/predictions.csv
```

### 4. Start FastAPI Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
- Health Check: `GET http://localhost:8000/health`
- Model Metadata: `GET http://localhost:8000/model/info`
- Single Flow Predict: `POST http://localhost:8000/predict`

### 5. Run Chronological Evaluation
Reproduces all benchmark metrics and figures:
```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

### 6. Run Test Suite
```bash
pytest tests/ -v
```

---

## 📊 Headline Results (Protocol B, 41,555 flows, D=1 delayed feedback)

| Model | Accuracy | Recall (ADR) | Attack F1 | FPR | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Static (no learning) | 0.6290 | 0.0061 | 0.0118 | 0.0192 | 0.2477 | 0.2491 |
| **Online SGD (primary)** | **0.9356** | **0.8688** | **0.9069** | **0.0266** | **0.9432** | **0.9225** |
| Online SGD + ADWIN drift | 0.9335 | 0.8629 | 0.9036 | 0.0265 | 0.9385 | 0.9146 |
| Online Anomaly Evidence | 0.9328 | 0.8580 | 0.9022 | 0.0249 | 0.9625 | 0.9462 |
| Full Adaptive ML-DST (proposed) | 0.8599 | 0.6742 | 0.7765 | 0.0351 | 0.9500 | 0.9129 |
| ML-DST @ fixed 0.50 threshold | 0.7839 | 0.4269 | 0.5879 | 0.0143 | 0.9462 | 0.9038 |

Full tables, per-day breakdowns, and the calibration trade-off analysis are in
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## 📑 Detailed Documentation

- [METHODOLOGY.md](docs/METHODOLOGY.md): System workflow, standard vs. custom components, and evaluation assumptions.
- [RESULTS.md](docs/RESULTS.md): Numerical evaluation tables, confusion matrices, and model comparisons.
- [LIMITATIONS.md](docs/LIMITATIONS.md): Operational trade-offs, false alarms, missed attacks, and dataset scope.
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): Exact reproduction steps from scratch.
- [CLEANUP_MANIFEST.md](docs/CLEANUP_MANIFEST.md): Inventory of safely archived and removed legacy files.
- [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md): Condensed run guide.
- [VIVA_QA.md](VIVA_QA.md): Anticipated defense questions with grounded answers.
- [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md): Requirement-to-implementation-to-evidence mapping.

---

## ⚠️ Scope & Known Limitations

This is a research prototype, not a production security product. In particular:

- Evaluation uses **precomputed flow statistics** (CICFlowMeter-style features); raw-packet feature
  extraction is out of scope.
- The blockchain ledger is **demonstrated against a local mock / EVM testnet**; throughput, gas cost,
  and latency at multi-gigabit line rates are untested.
- Physical infrastructure response is **simulated**; no live OpenFlow reconfiguration is performed.
- The ML-DST fusion component **does not outperform** the standalone SGD detector on this dataset, and
  is reported as a negative result rather than a contribution.

See [LIMITATIONS.md](LIMITATIONS.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the full analysis.

---

## 📄 License

Released under the [MIT License](LICENSE). The reference publication
([`Base_Paper.pdf`](Base_Paper.pdf)) and the CICIDS2017 dataset remain under their original terms and
are **not** covered by the MIT license — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
