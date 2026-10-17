# Execution and Operation Guide
## Adaptive ML-DST: Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance

This guide provides end-to-end instructions for reproducing the experiments, running unit test verification, launching the REST API, and starting the interactive Streamlit monitoring dashboard.

---

### 1. Environment Setup and Prerequisites

#### Python Version
Python 3.10 to 3.12 (Python 3.12.7 recommended).

#### Virtual Environment Installation
```bash
# Clone or navigate to the repository
cd d:/CNS_Project_And_Paper

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip and install all core dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install river>=0.26.0
```

---

### 2. Execution Modes

The framework supports two configurations configured via YAML:

#### Mode 1: Lightweight Development Mode (Verified Fast Execution)
- **Config File:** `configs/lightweight.yaml`
- **Data Path:** `data/processed/lightweight_sample.csv.gz` (50,000 global deduplicated, minority-preserved flows)
- **Execution Time:** ~60 seconds for full pipeline (Protocol A + Protocol B + Protocol C + Ablation + Tables)
- **Command:**
```bash
python -m scripts.run_experiments --config configs/lightweight.yaml
```

#### Mode 2: Full Dataset Mode (Chunked Scalable Processing)
- **Config File:** `configs/full.yaml`
- **Data Path:** `dataset/*.csv` (All 8 raw CICIDS2017 CSV files, 2.83 million flows, processed via memory-safe chunks)
- **Chunk Size:** 50,000 flows
- **Command:**
```bash
# 1. Run global deduplication across all 8 files
python -m src.data.deduplication

# 2. Run full evaluation pipeline
python -m scripts.run_experiments --config configs/full.yaml
```

---

### 3. Running the Evaluation Suite

#### Step 1: Run Full Evaluation Pipeline (Protocols A, B, C & Fair Ablation)
```bash
python -m scripts.run_experiments
```
This script executes:
- **Protocol A:** Duplicate-Safe Conventional Evaluation (Train/Val/Test with 0 hash overlap).
- **Protocol B:** Chronological Prequential Evaluation with 1-batch delayed label feedback ($t+1$).
- **Protocol C:** Controlled Concept-Drift Benchmark across 5 change points.
- **Phase 17 Ablation Study:** Evaluates all 9 variants (A through I) under identical data conditions.
- **Outputs Produced:**
  - `reports/experiments/full_experiment_results.json`
  - `reports/experiments/protocol_a_results.json`
  - `reports/experiments/protocol_b_results.json`
  - `reports/experiments/protocol_c_results.json`
  - `reports/experiments/ablation_study_results.json`
  - `reports/tables/table1_comparative_models.csv`
  - `reports/tables/table2_ablation_study.csv`
  - `reports/tables/table3_chronological_daily.csv`
  - `reports/tables/table4_drift_benchmark.csv`

#### Step 2: Run Task B Secondary Multiclass Evaluation
```bash
python -m scripts.run_multiclass
```
- **Outputs Produced:**
  - `reports/tables/table4_multiclass_families.csv`
  - `reports/experiments/multiclass_results.json`

#### Step 3: Generate Real Experimental Figures (Zero Synthetic Data)
```bash
python -m scripts.generate_plots
```
- **Figures Produced:**
  - `reports/figures/fig1_dataset_distribution_and_deduplication.png`
  - `reports/figures/fig2_protocol_a_roc_pr_curves.png`
  - `reports/figures/fig3_protocol_a_confusion_matrices.png`
  - `reports/figures/fig4_chronological_streaming_trajectory.png`
  - `reports/figures/fig5_daily_fnr_and_recall.png`
  - `reports/figures/fig6_ablation_study_breakdown.png`
  - `reports/figures/fig7_controlled_drift_recovery.png`
  - `reports/figures/fig8_isolated_latency_and_throughput.png`

---

### 4. Running the Automated Test Suite

To verify mathematical correctness, zero data leakage, and drift detection invariants:
```bash
pytest tests/ -v
```
To save the complete test output report:
```bash
powershell -Command "pytest tests/ -v | Out-File -FilePath reports/verification/test_results.txt -Encoding utf8"
```

All 47 unit and integration tests must pass:
- `tests/test_leakage_and_splits.py`: 0 hash overlap, 0 label in features, training-only preprocessor fitting.
- `tests/test_adwin_rigor.py`: ADWIN stationary stream stability, sudden drift, gradual drift, River parity.
- `tests/test_dst_rigor.py`: Mass conservation, Shafer reliability discounting, Yager conflict handling, NaN rejection.
- `tests/test_threshold_rigor.py`: Recall $\ge 90\%$ constraint, $F_2$ optimization, protected benign buffer, attack burst freezing.
- `tests/test_temporal_rigor.py`: Past-only computation, protected baseline, deviation scaling.
- `tests/test_defense.py`: 7-tier staged countermeasures.
- `tests/test_api.py`: FastAPI health, audit, train, predict, and telemetry endpoints.

---

### 5. Launching the FastAPI Backend

The FastAPI service provides REST endpoints for live model inference, streaming orchestration, and hierarchical risk telemetry.

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Key Endpoints:
- `GET http://localhost:8000/health`: Health status, model status, and blockchain adapter status.
- `POST http://localhost:8000/dataset/audit`: Returns the corrected CICIDS2017 audit report.
- `POST http://localhost:8000/train`: Initializes and trains the baseline online classifier and benign anomaly detector.
- `POST http://localhost:8000/predict`: Evaluates risk for an ingress flow feature vector.
- `POST http://localhost:8000/stream/start`: Starts real-time prequential replay in the background.
- `POST http://localhost:8000/stream/stop`: Pauses streaming.
- `GET http://localhost:8000/risk/nodes`: Returns current risk and DST evidence for all 10 cloud nodes.
- `GET http://localhost:8000/risk/clusters`: Returns aggregated risk for 3 cloud compute clusters.
- `GET http://localhost:8000/risk/global`: Returns system-wide attack tolerance risk.
- `GET http://localhost:8000/alerts`: Returns recent simulated defensive actions.
- `GET http://localhost:8000/drift/events`: Returns genuine ADWIN concept-drift events.
- `GET http://localhost:8000/threshold/history`: Returns adaptive threshold progression log.
- `GET http://localhost:8000/experiments`: Returns all completed experiment IDs and metadata.

Interactive Swagger API Documentation:
`http://localhost:8000/docs`

---

### 6. Launching the Interactive Streamlit Dashboard

The Streamlit dashboard provides a high-fidelity, real-time command center for monitoring attack tolerance, concept-drift alarms, DST uncertainty, and staged countermeasures.

```bash
streamlit run dashboard/app.py
```
Open your browser to: `http://localhost:8501`

#### Dashboard Features:
1. **Execution Configuration:** Toggle between *Lightweight Development* and *Full Dataset* modes.
2. **Evaluation Protocol Selection:** View Protocol A (Conventional), Protocol B (Chronological Daily), Protocol C (Drift Benchmark), or Phase 17 Ablation.
3. **Drift Strategy Selector:** Switch between *Plasticity Boost*, *Partial Reset*, *Full Reset*, and *No Adaptation*.
4. **Interactive Stream Controls:** Start, pause, step, and reset chronological flow replay.
5. **Real-Time Telemetry:**
   - Attack Score vs. Pignistic Risk Score vs. Adaptive Threshold.
   - DST Evidence Breakdown (Belief, Plausibility, Uncertainty $\Theta$, Conflict $K$).
   - Hierarchical Cloud Map: Live status of 10 virtual nodes across 3 clusters.
   - Genuine River ADWIN Alarms Banner.
   - Staged Defense Log with automated audit records.
   - Prominent Recall & FNR Warnings whenever attack recall drops below 90%.
6. **Export Capabilities:** Download real experiment results in JSON and CSV formats.
